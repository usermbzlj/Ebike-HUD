"""High-value event ring buffer for alert pre/post clips (REC-005)."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class BufferedFrame:
    timestamp_ns: int
    index: int
    bgr: np.ndarray
    blur: float
    tracks: int


class EventClipBuffer:
    def __init__(self, pre_s: float = 2.0, post_s: float = 2.0, fps: float = 30.0) -> None:
        self.pre_s = pre_s
        self.post_s = post_s
        self.fps = fps
        self._pre: deque[BufferedFrame] = deque(maxlen=max(8, int(pre_s * fps) + 4))
        self._pending: list[dict[str, Any]] = []

    def push(self, frame: BufferedFrame) -> list[list[BufferedFrame]]:
        self._pre.append(frame)
        done: list[list[BufferedFrame]] = []
        still = []
        for item in self._pending:
            item["frames"].append(frame)
            if (frame.timestamp_ns - item["t0"]) / 1e9 >= self.post_s:
                done.append(item["frames"])
            else:
                still.append(item)
        self._pending = still
        return done

    def on_alert(self, timestamp_ns: int, meta: dict[str, Any]) -> None:
        frames = list(self._pre)
        self._pending.append({"t0": timestamp_ns, "frames": frames, "meta": meta})

    def flush(self) -> list[list[BufferedFrame]]:
        out = [item["frames"] for item in self._pending]
        self._pending = []
        return out


def write_event_clip(frames: list[BufferedFrame], out_path: Path, fps: float, meta: dict[str, Any] | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        (out_path.with_suffix(".json")).write_text(json.dumps({"empty": True, **(meta or {})}), encoding="utf-8")
        return out_path
    h, w = frames[0].bgr.shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(fr.bgr)
    vw.release()
    sidecar = {
        "n_frames": len(frames),
        "start_ns": frames[0].timestamp_ns,
        "end_ns": frames[-1].timestamp_ns,
        "meta": meta or {},
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return out_path
