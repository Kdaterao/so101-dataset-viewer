"""Standalone metadata lookup and saved mask/point rendering. No model dependencies."""
from pathlib import Path
import cv2
import numpy as np


def episode_metadata(parts, needed, read):
    """Locate selected IDs in ordered LeRobot metadata shards by binary search."""
    import pandas as pd
    cache = {}
    selected = []
    for episode in sorted(needed):
        lo, hi = 0, len(parts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid not in cache:
                table = read(parts[mid])
                if table.empty:
                    raise ValueError(f"Empty episode metadata shard: {parts[mid]}")
                cache[mid] = (int(table.episode_index.min()), int(table.episode_index.max()),
                              table[table.episode_index.isin(needed)])
            first, last, rows = cache[mid]
            if episode < first:
                hi = mid - 1
            elif episode > last:
                lo = mid + 1
            else:
                matches = rows[rows.episode_index == episode]
                if len(matches) != 1:
                    raise ValueError(f"Missing or duplicate episode metadata for {episode}")
                selected.append(matches)
                break
        else:
            raise ValueError(f"Episode {episode} not found in ordered LeRobot metadata")
    return pd.concat(selected, ignore_index=True)


class DebugOverlayRenderer:
    def __init__(self, records, sidecar_root, *, show_masks=True, transition_seconds=None, fps=None):
        self.records = records
        self.show_masks = show_masks
        self.transition_frames = (max(0, round(transition_seconds * fps))
                                  if transition_seconds is not None and fps is not None else None)
        self.masks = []
        seen = set()
        for record in records:
            if not show_masks:
                break
            info = record["pov"].get("gripper_filter", {})
            identity = (info.get("mask_file"), tuple(info.get("window", [])))
            if info.get("mask_file") and identity in seen:
                continue
            seen.add(identity)
            packed = None
            if info.get("status") == "ok" and info.get("mask_file"):
                with np.load(Path(sidecar_root) / info["mask_file"], allow_pickle=False) as archive:
                    packed = (archive["packed_masks"].copy(), archive["shape"].copy(),
                              {int(frame): i for i, frame in enumerate(archive["frame_indices"])},
                              archive["valid_frames"].copy() if "valid_frames" in archive else None)
            self.masks.append((info, packed))

    @staticmethod
    def _dots(image, result, offset, color):
        tracks = np.asarray(result.get("tracks", []))
        visibility = np.asarray(result.get("visibility", []), dtype=bool)
        if tracks.ndim != 3 or visibility.shape != tracks.shape[:2] or not 0 <= offset < tracks.shape[1]:
            return
        h, w = image.shape[:2]
        for point in tracks[visibility[:, offset], offset]:
            if np.isfinite(point).all() and 0 <= point[0] < w and 0 <= point[1] < h:
                xy = tuple(np.rint(point).astype(int))
                cv2.circle(image, xy, 5, (0, 0, 0), -1)
                cv2.circle(image, xy, 3, color, -1)

    def render(self, rgb, frame, camera):
        image = rgb.copy()
        current = next((r for r in self.records if r["start_frame"] <= frame <= r["end_frame"]), None)
        if current is None:
            return image
        offset = frame-current["start_frame"]
        status = ""
        if camera == "wrist":
            status = "mask: outside window"
            for info, packed in self.masks:
                window = info.get("window")
                if window is None or not window[0] <= frame <= window[1]:
                    continue
                if packed is None:
                    status = "mask: REVIEW required"
                    continue
                bits, shape, indices, validity = packed
                if frame not in indices:
                    status = "mask: REVIEW required"
                    continue
                mask = np.unpackbits(bits[indices[frame]], axis=-1, count=int(shape[2])).astype(bool)
                if mask.shape != image.shape[:2]:
                    status = "mask: REVIEW required"
                    continue
                image[mask] = (0.6*image[mask]+0.4*np.array([255, 0, 255])).astype(np.uint8)
                status = "mask: SAM2" if validity is None or validity[indices[frame]] else "mask: REJECTED / REVIEW"
            if not self.show_masks:
                status = ""
            elif current["pov"].get("gripper_filter", {}).get("status") == "disabled":
                status = "mask: disabled"
            result = current["pov"]
            previous = result.get("previous_stage", {})
        else:
            result = current.get("third_person", {}).get(camera, {})
            previous = result.get("transition_from_previous", {})
        if self.transition_frames is None or offset < self.transition_frames:
            self._dots(image, previous, offset, (0, 255, 255))
        self._dots(image, result, offset, (0, 255, 0))
        text = f"DEBUG subtask {current['subtask']} {result.get('source', 'none')} {status}"
        cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
        cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        return image
