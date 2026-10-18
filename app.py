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


def episode_player(clip, records, fps, selected_start):
    """Browser-time caption updates avoid a Streamlit rerun on every video frame."""
    import streamlit.components.v1 as components
    timeline = [dict(id=r["node_id"], start=r["start_frame"], end=r["end_frame"],
                     caption=r.get("detailed_caption") or r.get("brief_caption") or "No revised caption available") for r in records]
    data = json.dumps(timeline).replace("<", "\\u003c")
    source = base64.b64encode(clip).decode("ascii")
    html = '''<!doctype html><meta charset="utf-8">
<style>body{font:15px system-ui;margin:0;color:#ddd;background:#0e1117}
#caption{max-height:190px;overflow:auto;padding:10px;background:#1c2230;border-radius:6px;margin-bottom:8px}
p{margin:6px 0;white-space:pre-wrap}video{width:100%;display:block}select{max-width:100%;padding:6px;margin-bottom:8px}</style>
<div id="caption"><p id="rewritten"></p></div>
<video id="video" controls autoplay muted playsinline src="data:video/mp4;base64,VIDEO_DATA"></video>
<script>
const timeline=TIMELINE_DATA,fps=FPS_DATA,video=document.getElementById('video');
let last=null;
function update(){const frame=Math.floor(video.currentTime*fps+1e-5),index=timeline.findIndex(s=>frame>=s.start&&frame<s.end);
if(index===last)return;last=index;const s=timeline[index];
document.getElementById('rewritten').textContent=s?.caption||'No subtask annotation at this time';
}
video.onloadedmetadata=()=>{video.currentTime=START_DATA/fps;update();video.play().catch(()=>{});};
video.addEventListener('timeupdate',update);video.addEventListener('seeked',update);
function tick(){update();requestAnimationFrame(tick)}tick();
</script>'''
    html = (html.replace("TIMELINE_DATA", data).replace("FPS_DATA", str(float(fps)))
            .replace("START_DATA", str(int(selected_start))).replace("VIDEO_DATA", source))
    components.html(html, height=720, scrolling=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--root", help="Completed local combined dataset folder")
    inputs.add_argument("--repo-id", help="Combined Hugging Face dataset")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    default_root = Path("outputs/so101_spread73_v3_unified")
    root = args.root or (str(default_root) if not args.repo_id and (default_root / "subtasks.jsonl").is_file() else "")
    repo = args.repo_id or os.environ.get("HF_DATASET_REPO", "kdaterao/so101_spread73_v3_unified")
    st.set_page_config(page_title="Robot episodes", layout="wide", initial_sidebar_state="collapsed")
    try:
        settings = dict(st.secrets)
    except Exception:
        settings = {}
    if settings.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = str(settings["HF_TOKEN"])
    if not args.repo_id:
        repo = settings.get("HF_DATASET_REPO", repo)
    try:
        resolved, names, rows = catalog(root, repo, args.revision)
        keys = sorted({(r["source_repo_id"], r["source_revision"], r["episode_index"]) for r in rows})
        if not keys:
            st.info("No source episodes available.")
            return
        multiple_sources = len({k[:2] for k in keys}) > 1
        selected = st.selectbox("Source episode", keys, format_func=lambda k:
            f"Episode {k[2]}" + (f" · {k[0]} · {k[1][:10]}" if multiple_sources else ""))
        records = sorted([r for r in rows if (
            r["source_repo_id"], r["source_revision"], r["episode_index"]) == selected],
            key=lambda r: r["start_frame"])
        record = next((r for r in records if r.get("grounding_path") and r.get("grounding_dataset_path")), None)
        if record is None:
            st.info("No grounded video is available for this source episode.")
            return
        prefix = record["grounding_dataset_path"]
        fps, cameras = video_metadata(root, repo, resolved, prefix,
                                      int(record["grounding_destination_episode"]), tuple(names))
        report_path = safe_file(root, repo, resolved, record["grounding_path"])
        report = json.loads(report_path.read_text())
        if not report.get("subtasks"):
            st.info("No grounding intervals available for this episode.")
            return
        renderer = None
        render_hash = hashlib.sha256(json.dumps(report, sort_keys=True).encode())
        if not report.get("debug_overlays", False):
            import tempfile
            import shutil
            with tempfile.TemporaryDirectory(prefix="robot101-view-masks-") as directory:
                copied = set()
                for entry in report["subtasks"]:
                    info = entry.get("pov", {}).get("gripper_filter", {})
                    if info.get("status") == "ok" and info.get("mask_file"):
                        if info['mask_file'] in copied:
                            continue
                        copied.add(info['mask_file'])
                        source = safe_file(root, repo, resolved, f"{prefix}/point_tracks/{info['mask_file']}")
                        target = Path(directory) / info["mask_file"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                        render_hash.update(source.read_bytes())
                renderer = DebugOverlayRenderer(report["subtasks"], Path(directory))
        chosen = [camera for camera in ("top", "wrist") if camera in cameras]
        if not chosen:
            chosen = list(cameras)[:2]
        if not chosen:
            st.info("No camera videos are available for this episode.")
            return
        for column, camera in zip(st.columns(len(chosen)), chosen):
            with column:
                try:
                    video = cameras[camera]
                    with st.spinner(f"Preparing {camera} playback…"):
                        path = safe_file(root, repo, resolved, video["path"])
                        clip = playback_clip(str(path), (path.stat().st_size, path.stat().st_mtime_ns),
                            video["offset"], fps, 0, max(r["end_frame"] for r in report["subtasks"]) + 1,
                            camera, render_hash.hexdigest(), renderer)
                    episode_player(clip, records, fps, 0)
                except Exception as exc:
                    st.error(f"{camera}: {exc}")
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
