"""Streamlit inspector for robot101.data.combine_datasets outputs."""
from __future__ import annotations

import json
import os
import argparse
import hashlib
import io
import base64
from fractions import Fraction
from pathlib import Path, PurePosixPath

import av
import pandas as pd
import streamlit as st

from viewer_utils import episode_metadata, DebugOverlayRenderer


def safe_file(root, repo, revision, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe artifact path: {name}")
    if root:
        local = Path(root).expanduser().resolve() / name
        if local.is_file():
            return local
        manifest_path = Path(root).expanduser().resolve() / "manifest.json"
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            return Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))
        except EntryNotFoundError:
            manifest_path = Path(hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=revision))
    from huggingface_hub import hf_hub_download
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config", {}).get("lightweight"):
            for source in manifest["inputs"]:
                prefix = source["path"].rstrip("/") + "/"
                if name.startswith(prefix):
                    return Path(hf_hub_download(source["repo_id"], name[len(prefix):],
                                               repo_type="dataset", revision=source["revision"]))
    raise FileNotFoundError(f"Artifact not available: {name}")


@st.cache_data(show_spinner=False, max_entries=3, ttl=300)
def catalog(root, repo, requested_revision):
    if root:
        root = str(Path(root).expanduser().resolve())
        if not (Path(root) / "subtasks.jsonl").is_file():
            if (Path(root) / "manifest.json").is_file():
                raise ValueError("Combining has not finished: only the input manifest exists. "
                                 "Resume the combine_datasets command and wait for its 'Saved … episodes' message, "
                                 "then refresh this viewer. Uploading is optional; omit --push-to-hub for local use.")
            raise ValueError("This folder is not a completed combined dataset; select a folder containing subtasks.jsonl.")
        revision = "local"
        names = [p.relative_to(Path(root)).as_posix() for p in Path(root).rglob("*") if p.is_file()]
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.repo_info(repo, repo_type="dataset", revision=requested_revision, files_metadata=False)
        revision = info.sha
        names = [entry.rfilename for entry in info.siblings]
    path = safe_file(root, repo, revision, "subtasks.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return revision, names, rows


@st.cache_data(show_spinner=False, max_entries=12)
def video_metadata(root, repo, revision, prefix, destination_episode, names):
    info = json.loads(safe_file(root, repo, revision, f"{prefix}/meta/info.json").read_text())
    parts = sorted(n for n in names if n.startswith(f"{prefix}/meta/episodes/") and n.endswith(".parquet"))
    table = episode_metadata(parts, {destination_episode},
                             lambda n: pd.read_parquet(safe_file(root, repo, revision, n)))
    row = table.iloc[0]
    cameras = {}
    for camera in ("top", "wrist", "side"):
        key = f"observation.images.{camera}"
        if info["features"].get(key, {}).get("dtype") != "video":
            continue
        relative = info["video_path"].format(video_key=key,
            chunk_index=int(row[f"videos/{key}/chunk_index"]),
            file_index=int(row[f"videos/{key}/file_index"]))
        cameras[camera] = dict(path=f"{prefix}/{relative}", offset=float(row[f"videos/{key}/from_timestamp"]))
    return float(info["fps"]), cameras


@st.cache_data(show_spinner=False, max_entries=6)
def playback_clip(path, file_stamp, offset, fps, start, end, camera, render_key, _renderer):
    """Decode once and render an exact half-open interval into a cached H.264 preview."""
    rate = Fraction(str(fps)).limit_denominator(100000)
    buffer = io.BytesIO()
    with av.open(path) as source, av.open(buffer, mode="w", format="mp4") as output:
        video = source.streams.video[0]
        source.seek(max(0, int((offset + start / fps) / video.time_base)), stream=video, backward=True)
        decoded = (frame for frame in source.decode(video) if frame.time is not None)
        previous, current = None, next(decoded, None)
        encoder = None
        for index, episode_frame in enumerate(range(start, end)):
            target = offset + episode_frame / fps
            while current is not None and float(current.time) < target:
                previous, current = current, next(decoded, None)
            choices = [frame for frame in (previous, current) if frame is not None]
            nearest = min(choices, key=lambda frame: abs(float(frame.time) - target)) if choices else None
            if nearest is None or abs(float(nearest.time) - target) > 1.5 / fps:
                raise ValueError(f"Missing aligned {camera} frame {episode_frame}; preview not fabricated")
            rgb = nearest.to_ndarray(format="rgb24")
            if _renderer is not None:
                rgb = _renderer.render(rgb, episode_frame, camera)
            if encoder is None:
                encoder = output.add_stream("libx264", rate=rate)
                encoder.width, encoder.height = rgb.shape[1], rgb.shape[0]
                encoder.pix_fmt = "yuv420p"
                encoder.options = {"crf": "20", "preset": "veryfast"}
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, 1 / rate
            for packet in encoder.encode(frame):
                output.mux(packet)
        if encoder is None:
            raise ValueError("Empty subtask preview")
        for packet in encoder.encode():
            output.mux(packet)
    return buffer.getvalue()


def episode_player(clips, records, fps, selected_start=0):
    """Single shared playback clock, responsive cameras, and live annotations."""
    import streamlit.components.v1 as components

    timeline = [
        {"start": int(r["start_frame"]), "end": int(r["end_frame"]),
         "caption": r.get("detailed_caption") or r.get("brief_caption") or "No annotation available"}
        for r in records
    ]
    sources = [{"name": camera, "src": "data:video/mp4;base64," + base64.b64encode(clip).decode("ascii")}
               for camera, clip in clips]
    html = r'''<!doctype html><html><head><meta charset="utf-8"><style>
      * {box-sizing:border-box}
      html,body {margin:0;padding:0;background:transparent;color:#263342;font:15px system-ui,sans-serif;overflow:hidden}
      #app {display:flex;flex-direction:column;gap:12px;padding:0 0 3px}
      #annotation,.camera,#controls {background:#fff;border:1px solid #d8dee7;border-radius:8px}
      #annotation {padding:16px 18px;min-height:88px}
      #annotation-head {display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:8px}
      #caption-label {color:#0f52ba;font-size:11px;font-weight:700;letter-spacing:.04em}
      #segment {color:#687586;font-size:12px}
      #caption {font-size:15px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}
      #grid {display:grid;grid-template-columns:repeat(CAMERA_COUNT,minmax(0,1fr));gap:12px;align-items:start}
      .camera {min-width:0;overflow:hidden}
      .camera-label {padding:9px 12px;background:#f7f8fa;border-bottom:1px solid #e2e7ed;font-size:11px;font-weight:700;color:#536171;text-transform:uppercase;letter-spacing:.04em}
      video {display:block;width:100%;height:auto;object-fit:contain;background:#f1f3f5}
      #controls {display:flex;align-items:center;gap:12px;padding:10px 12px}
      button {background:#0f52ba;border:1px solid #0f52ba;border-radius:6px;color:white;padding:8px 16px;cursor:pointer}
      button:disabled {opacity:.55}
      button:hover:not(:disabled) {background:#0b438f;border-color:#0b438f}
      input[type=range] {min-width:30px;flex:1;accent-color:#0f52ba}
      #clock {white-space:nowrap;font-size:12px;color:#536171;font-variant-numeric:tabular-nums}
      #status {font-size:12px;color:#687586}
      #status:empty {display:none}
      @media(max-width:650px) {#grid{grid-template-columns:1fr}#controls{gap:7px}#annotation{padding:14px}}
    </style></head><body><main id="app">
      <section id="annotation"><div id="annotation-head"><span id="caption-label">GENERATED ACTION ANNOTATION</span><span id="segment"></span></div><div id="caption">Loading annotations…</div></section>
      <section id="grid"></section>
      <section id="controls"><button id="toggle" disabled>Loading…</button><input type="range" id="seek" min="0" max="1" value="0" step="1" disabled><span id="clock">0:00 / 0:00</span></section>
      <div id="status">Loading camera videos…</div>
    </main><script>
      const timeline=TIMELINE_DATA, sources=SOURCES_DATA, fps=FPS_DATA;
      const app=document.getElementById('app'),grid=document.getElementById('grid');
      const button=document.getElementById('toggle'),slider=document.getElementById('seek');
      const caption=document.getElementById('caption'),segment=document.getElementById('segment');
      const clock=document.getElementById('clock'),status=document.getElementById('status');
      let duration=0,ready=false,playing=false,scrubbing=false,seeking=false,seekSerial=0,lastIndex=-2;
      const videos=[];
      function resize(){
        const h=Math.ceil(app.getBoundingClientRect().height)+12;
        window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:h},'*');
      }
      new ResizeObserver(resize).observe(app);
      window.addEventListener('resize',resize);
      function fmt(t){let x=Math.floor(Math.max(0,t));return Math.floor(x/60)+':'+String(x%60).padStart(2,'0')}
      function updateCaption(t){
        const f=Math.floor(t*fps+1e-5);
        const i=timeline.findIndex(s=>s.start<=f&&f<s.end);
        if(i===lastIndex)return;
        lastIndex=i;const s=timeline[i];
        caption.textContent=s?s.caption:'No annotation for this interval';
        segment.textContent=s?`Frames ${s.start}–${s.end}`:'';
        resize();
      }
      function update(t){
        updateCaption(t);
        if(!scrubbing)slider.value=Math.round(t*fps);
        clock.textContent=fmt(t)+' / '+fmt(duration);
        button.textContent=playing?'Pause':'Play';
      }
      function pauseAll(){videos.forEach(v=>v.pause())}
      function fail(msg){playing=false;pauseAll();button.disabled=true;slider.disabled=true;status.textContent=msg;resize()}
      function playAll(){
        if(!ready||seeking||!playing)return;
        Promise.all(videos.map(v=>v.play())).then(()=>{status.textContent=''}).catch(()=>{
          playing=false;pauseAll();status.textContent='Click Play to start both videos.';update(videos[0].currentTime);
        });
      }
      function move(v,t){return new Promise((resolve,reject)=>{
        if(Math.abs(v.currentTime-t)<.002&&!v.seeking)return resolve();
        const done=()=>{v.removeEventListener('seeked',done);v.removeEventListener('error',bad);resolve()};
        const bad=()=>{v.removeEventListener('seeked',done);v.removeEventListener('error',bad);reject(new Error('Unable to seek video'))};
        v.addEventListener('seeked',done,{once:true});v.addEventListener('error',bad,{once:true});v.currentTime=t;
      })}
      async function seekAll(t){
        const id=++seekSerial;seeking=true;pauseAll();
        t=Math.max(0,Math.min(t,Math.max(0,duration-1/fps)));updateCaption(t);
        try{await Promise.all(videos.map(v=>move(v,t)));if(id!==seekSerial)return;seeking=false;update(t);playAll()}
        catch(e){if(id===seekSerial){seeking=false;fail(e.message)}}
      }
      const loads=sources.map(s=>new Promise((resolve,reject)=>{
        const box=document.createElement('div');box.className='camera';
        const title=document.createElement('div');title.className='camera-label';title.textContent=s.name+' camera';
        const video=document.createElement('video');video.muted=true;video.playsInline=true;video.preload='auto';
        box.append(title,video);grid.appendChild(box);videos.push(video);
        video.addEventListener('loadedmetadata',resize);
        video.addEventListener('canplay',()=>{if(video.readyState>=3)resolve()},{once:true});
        video.addEventListener('error',()=>reject(new Error(s.name+' camera could not load')),{once:true});
        video.src=s.src;video.load();
      }));
      Promise.all(loads).then(async()=>{
        duration=Math.min(...videos.map(v=>v.duration));
        if(!Number.isFinite(duration)||duration<=0)throw new Error('Video duration unavailable');
        ready=true;button.disabled=false;slider.disabled=false;slider.max=Math.max(0,Math.floor(duration*fps)-1);
        playing=false;await seekAll(START_DATA/fps);resize();
        status.textContent='Press Play to view the synchronized cameras.';
      }).catch(e=>fail(e.message));
      button.onclick=()=>{if(!ready)return;playing=!playing;if(playing){if(videos[0].ended)seekAll(0);else playAll()}else pauseAll();update(videos[0].currentTime)};
      slider.oninput=()=>{scrubbing=true;pauseAll();updateCaption(Number(slider.value)/fps)};
      slider.onchange=()=>{scrubbing=false;seekAll(Number(slider.value)/fps)};
      function tick(){
        if(ready&&!seeking&&!scrubbing){
          const t=videos[0].currentTime;
          if(playing&&!videos[0].paused){
            if(videos.slice(1).some(v=>Math.abs(v.currentTime-t)>.12)){
              seekAll(t);
            }else{
              videos.slice(1).forEach(v=>{let d=v.currentTime-t;v.playbackRate=d>.025?.98:d<-.025?1.02:1})
            }
          }
          if(videos[0].ended){playing=false;pauseAll()}
          update(t);
        }
        requestAnimationFrame(tick);
      }
      resize();tick();
    </script></body></html>'''
    html = (html.replace('TIMELINE_DATA', json.dumps(timeline).replace('<', '\\u003c'))
                .replace('SOURCES_DATA', json.dumps(sources).replace('<', '\\u003c'))
                .replace('FPS_DATA', str(float(fps)))
                .replace('START_DATA', str(int(selected_start)))
                .replace('CAMERA_COUNT', str(len(clips))))
    components.html(html, height=720, scrolling=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--root', help='Completed local combined dataset folder')
    inputs.add_argument('--repo-id', help='Combined Hugging Face dataset')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--transition-display-seconds', type=float, default=0.5)
    parser.add_argument('--show-masks', action='store_true')
    args, _ = parser.parse_known_args()
    if not 0 <= args.transition_display_seconds < float('inf'):
        parser.error('--transition-display-seconds must be finite and nonnegative')

    st.set_page_config(page_title='SO-101 Auto-Annotation', page_icon='🤖', layout='centered')
    st.html('''<style>
      .page-title{font-size:40px;line-height:1.2;text-decoration:underline;text-decoration-color:#0F52BA;text-decoration-thickness:3px;text-underline-offset:7px;margin:0 0 14px}
      .metric-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.65rem;margin:.25rem 0 .35rem}
      .metric-item{display:flex;flex-direction:column;align-items:flex-start;justify-content:center;gap:.35rem;min-width:0;min-height:94px;padding:.75rem .9rem;border:1px solid #d8dee7;border-radius:8px;background:#fff}
      .metric-value{color:#0F52BA;font-size:2rem;font-weight:650;line-height:1}
      .metric-label{color:#536171;font-size:.78rem;line-height:1.25}
      .metric-note{margin:0 0 .7rem;color:#687586;font-size:.72rem}
      @media(max-width:600px){.page-title{font-size:32px}}
      @media(max-width:600px){.metric-strip{grid-template-columns:repeat(2,minmax(0,1fr));gap:.4rem}.metric-item{min-height:78px;padding:.55rem}.metric-value{font-size:1.5rem}.metric-label{font-size:.66rem}}
      </style>''')
    st.markdown('<h1 class="page-title">SO-101 Auto-Annotation</h1>', unsafe_allow_html=True)
    st.write('Browse annotated SO-101 episodes with synchronized camera views and action captions.')
    source_col, repo_col = st.columns([3, 1])
    with source_col:
        st.caption('Source demonstrations: [felsager/community_dataset_v3_ee_smolVLA](https://huggingface.co/datasets/felsager/community_dataset_v3_ee_smolVLA)')
    with repo_col:
        st.link_button('GitHub repo', 'https://github.com/Kdaterao/so101-auto-annotation')

    st.divider()

    demo_tab, explanation_tab = st.tabs(['Interactive Demo', 'Pipeline Explanation'])
    with demo_tab:
        try:
            try:
                settings=dict(st.secrets)
            except Exception:
                settings={}
            if settings.get('HF_TOKEN'):
                os.environ['HF_TOKEN']=str(settings['HF_TOKEN'])
            repo=args.repo_id or os.environ.get('HF_DATASET_REPO','kdaterao/so101_spread73_v3_unified')
            if not args.repo_id:
                repo=settings.get('HF_DATASET_REPO',repo)
            default_root=Path('outputs/so101_spread73_v3_unified')
            root=args.root or (str(default_root) if not args.repo_id and (default_root/'subtasks.jsonl').is_file() else '')
            revision,names,rows=catalog(root,repo,args.revision)
            keys=sorted({(r['source_repo_id'],r['source_revision'],r['episode_index']) for r in rows})
            if not keys:
                st.info('No episodes found in the dataset.')
            else:
                multi=len({k[:2] for k in keys})>1
                default_key = next((key for key in keys if key[2] == 12), min(keys, key=lambda key: key[2]))
                selected=st.selectbox('Source episode',keys,index=keys.index(default_key),format_func=lambda k:f'Episode {k[2]}'+(f' · {k[0]}' if multi else ''))
                records=sorted([r for r in rows if (r['source_repo_id'],r['source_revision'],r['episode_index'])==selected],key=lambda r:r['start_frame'])
                record=next((r for r in records if r.get('grounding_path') and r.get('grounding_dataset_path')),None)
                if record is None:
                    st.info('No grounded video available for this episode.')
                else:
                    prefix=record['grounding_dataset_path']
                    fps,cameras=video_metadata(root,repo,revision,prefix,int(record['grounding_destination_episode']),tuple(names))
                    report=json.loads(safe_file(root,repo,revision,record['grounding_path']).read_text())
                    if not report.get('subtasks'):
                        st.info('No grounded subtasks available for this episode.')
                    else:
                        render_hash=hashlib.sha256(json.dumps(report,sort_keys=True).encode())
                        render_hash.update(f'{args.show_masks}:{args.transition_display_seconds}'.encode())
                        renderer=None
                        if not report.get('debug_overlays',False):
                            # Renderer needs its temporary mask files to stay alive until clips are encoded.
                            from contextlib import ExitStack
                            import tempfile,shutil
                            with ExitStack() as stack:
                                directory=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='robot101-masks-')))
                                if args.show_masks:
                                    for entry in report['subtasks']:
                                        info=entry.get('pov',{}).get('gripper_filter',{})
                                        if info.get('status')=='ok' and info.get('mask_file'):
                                            mask_file=info['mask_file']
                                            source=safe_file(root,repo,revision,f'{prefix}/point_tracks/{mask_file}')
                                            target=directory/mask_file
                                            target.parent.mkdir(parents=True,exist_ok=True)
                                            if not target.exists():
                                                shutil.copy2(source,target)
                                                render_hash.update(source.read_bytes())
                                renderer=DebugOverlayRenderer(report['subtasks'],directory,show_masks=args.show_masks,transition_seconds=args.transition_display_seconds,fps=fps)
                                chosen=[cam for cam in ('top','wrist') if cam in cameras] or list(cameras)[:2]
                                clips=[]
                                for cam in chosen:
                                    with st.spinner(f'Preparing {cam} camera…'):
                                        video=cameras[cam]
                                        path=safe_file(root,repo,revision,video['path'])
                                        data=playback_clip(str(path),(path.stat().st_size,path.stat().st_mtime_ns),video['offset'],fps,0,max(r['end_frame'] for r in report['subtasks'])+1,cam,render_hash.hexdigest(),renderer)
                                        clips.append((cam,data))
                                if clips:
                                    episode_player(clips,records,fps,0)
                                else:
                                    st.info('No camera streams available.')
                        else:
                            chosen=[cam for cam in ('top','wrist') if cam in cameras] or list(cameras)[:2]
                            clips=[]
                            for cam in chosen:
                                with st.spinner(f'Preparing {cam} camera…'):
                                    video=cameras[cam]
                                    path=safe_file(root,repo,revision,video['path'])
                                    data=playback_clip(str(path),(path.stat().st_size,path.stat().st_mtime_ns),video['offset'],fps,0,max(r['end_frame'] for r in report['subtasks'])+1,cam,render_hash.hexdigest(),None)
                                    clips.append((cam,data))
                            if clips:
                                episode_player(clips,records,fps,0)
                            else:
                                st.info('No camera streams available.')
        except Exception as exc:
            st.error(f'Unable to load demonstration: {exc}')
            with st.expander('Error details'):
                st.exception(exc)

    with explanation_tab:
        st.header('Project metrics')
        st.html('''<div class="metric-strip" aria-label="Project metrics">
          <div class="metric-item"><span class="metric-value">50</span><span class="metric-label">Robot demonstrations</span></div>
          <div class="metric-item"><span class="metric-value">~70%</span><span class="metric-label">Annotation accuracy</span></div>
          <div class="metric-item"><span class="metric-value">~90%</span><span class="metric-label">Grounding point accuracy</span></div>
        </div>
        <p class="metric-note">Approximate estimates based on manual review of 50 episodes.</p>''')
        st.divider()
        st.header('How episodes are segmented, annotated, and grounded')
        st.write('The pipeline starts with the source robot videos and produces time-aligned captions and spatial points for each episode.')
        st.markdown('**Segment → Caption → Ground → Aggregate and export**')

        st.markdown('#### 1. Segment episodes')
        st.write('V-JEPA 2.1 extracts visual features from sampled top-camera frames. Ward clustering groups adjacent spans into a temporal hierarchy, and an adaptive cut selects the subtask boundaries. Each interval maps back to source frames using half-open bounds: `[start, end)`.')

        st.markdown('#### 2. Generate captions')
        st.write('Perception-LM-3B describes actions across video spans. Qwen3-VL-8B captions objects and spatial relations from keyframes; synchronized wrist frames can provide extra context. These captions give the later grounding and aggregation passes a time-aligned description of what is visible.')

        st.markdown('#### 3. Ground gripper and task objects')
        st.write('Molmo2 proposes gripper and object points. SAM 2 segments the gripper in the POV view, and BootsTAPIR tracks selected points across each interval. Third-person views help locate task objects and destinations. Gripper-event splits can refine tracking intervals without changing the parent caption boundaries.')
        st.caption('Missing camera frames are left missing; the pipeline does not fabricate visual evidence.')

        st.markdown('#### 4. Standardize and package')
        st.write('After captioning and grounding, Claude Haiku uses the text descriptions and grounding diagnostics to standardize action, object, destination, and completion fields and flag conflicts. The combiner joins records by source dataset, revision, episode, and segment.')
        st.caption('Haiku checks text consistency; it does not inspect pixels or verify a mask or point. Review flags remain for human inspection.')

        st.divider()
        st.markdown('**What the dataset keeps**')
        st.write('The combiner links captions and grounding by source dataset, revision, episode, and segment. It preserves clean videos, robot state/action data, point-track and mask sidecars, and review flags so uncertain results can be checked by a person.')

if __name__=='__main__':
    main()
