"""Desktop session replay: seek any timestamp across video, IMU, tracks, alerts (REP-001/002)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar.session import iter_jsonl
from rpar.timebase import TimeInterpolator, percentile_intervals_ms


class SessionReplay:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.frames = list(iter_jsonl(self.root / "camera" / "frame_metadata.jsonl"))
        self.tracks = list(iter_jsonl(self.root / "perception" / "tracks.jsonl"))
        self.alerts = list(iter_jsonl(self.root / "events" / "alerts.jsonl"))
        self.diag = list(iter_jsonl(self.root / "diagnostics" / "runtime.jsonl"))
        self.gyro = list(iter_jsonl(self.root / "imu" / "gyro.jsonl"))
        self.accel = list(iter_jsonl(self.root / "imu" / "accelerometer.jsonl"))
        self._cap = None
        self._gyro_ip = None
        if self.gyro:
            t = np.array([g["timestamp_ns"] for g in self.gyro], dtype=np.int64)
            v = np.array([[g["x"], g["y"], g["z"]] for g in self.gyro], dtype=np.float64)
            self._gyro_ip = TimeInterpolator(t, v)
        overlay = self.root / "video" / "overlay_preview.mp4"
        raw = sorted((self.root / "video").glob("segment_*.mp4"))
        self.video_path = overlay if overlay.exists() else (raw[0] if raw else None)
        self.event_index = self._build_events()

    def _build_events(self) -> list[dict[str, Any]]:
        events = []
        for a in self.alerts:
            if not a.get("fired"):
                continue
            ts = int(a.get("timestamp_ns") or 0)
            idx = self.index_at_ns(ts)
            events.append({"index": idx, "timestamp_ns": ts, "phrase": a.get("phrase"), "track_id": a.get("track_id")})
        return events

    def index_at_ns(self, ts: int) -> int:
        if not self.frames:
            return 0
        best_i, best_d = 0, 10**18
        for i, f in enumerate(self.frames):
            d = abs(int(f.get("sensor_timestamp_ns") or 0) - ts)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def imu_series(self, index: int, window: int = 40) -> dict[str, Any]:
        if not self.frames:
            return {"gyro": [], "accel": []}
        ts = int(self.frames[index].get("sensor_timestamp_ns") or 0)
        gy = [g for g in self.gyro if abs(int(g["timestamp_ns"]) - ts) < 120_000_000][:window]
        ac = [g for g in self.accel if abs(int(g["timestamp_ns"]) - ts) < 120_000_000][:window]
        return {"gyro": gy, "accel": ac}

    def export_clip(self, out_path: Path, start_i: int, end_i: int) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self._cap is None:
            self.open()
        start_i = max(0, start_i)
        end_i = min(self.n_frames() - 1, end_i)
        fps = 30
        man = self.manifest or {}
        w = h = None
        frames = []
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_i)
            for i in range(start_i, end_i + 1):
                ok, bgr = self._cap.read()
                if not ok:
                    break
                frames.append(bgr)
                w, h = bgr.shape[1], bgr.shape[0]
        if not frames or w is None or h is None:
            raise FileNotFoundError("no video frames to export")
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            vw.write(fr)
        vw.release()
        sidecar = {
            "start": start_i,
            "end": end_i,
            "n": len(frames),
            "session": man.get("session_id"),
        }
        out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return out_path

    def open(self) -> None:
        if self.video_path:
            self._cap = cv2.VideoCapture(str(self.video_path))

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def n_frames(self) -> int:
        return len(self.frames)

    def at(self, index: int) -> dict[str, Any]:
        index = int(np.clip(index, 0, max(0, self.n_frames() - 1)))
        meta = self.frames[index]
        ts = int(meta.get("sensor_timestamp_ns") or 0)
        tracks = [t for t in self.tracks if int(t.get("source_frame_id", t.get("timestamp_ns", 0))) in {index, ts} or abs(int(t.get("timestamp_ns", 0)) - ts) < 2_000_000]
        if not tracks:
            # fallback: nearest timestamp bucket
            tracks = [t for t in self.tracks if abs(int(t.get("timestamp_ns", 0)) - ts) <= 40_000_000]
        alerts = [a for a in self.alerts if abs(int(a.get("timestamp_ns", 0)) - ts) <= 40_000_000]
        gyro = self._gyro_ip.at(ts).tolist() if self._gyro_ip is not None else [0, 0, 0]
        jpeg = b""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = self._cap.read()
            if ok:
                _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                jpeg = buf.tobytes()
        return {
            "index": index,
            "n_frames": self.n_frames(),
            "meta": meta,
            "tracks": tracks[:12],
            "alerts": alerts,
            "gyro": gyro,
            "imu_stats": percentile_intervals_ms([int(g["timestamp_ns"]) for g in self.gyro[:400]]) if self.gyro else {},
            "imu_series": self.imu_series(index),
            "events": self.event_index,
            "diag": self.diag[index] if index < len(self.diag) else {},
            "manifest": self.manifest,
            "jpeg": jpeg,
        }

    def hit_test(self, index: int, x: float, y: float) -> dict[str, Any] | None:
        from rpar.annotation import hit_test_tracks

        st = self.at(index)
        return hit_test_tracks(st.get("tracks") or [], x, y)


def scan_time_offset_ms(session_dir: Path, window_ms: float = 200.0) -> dict[str, Any]:
    """SYNC-006: scan ±200 ms for gyro-energy vs frame blur correlation."""
    root = Path(session_dir)
    frames = list(iter_jsonl(root / "camera" / "frame_metadata.jsonl"))
    diag = list(iter_jsonl(root / "diagnostics" / "runtime.jsonl"))
    gyro = list(iter_jsonl(root / "imu" / "gyro.jsonl"))
    if len(frames) < 8 or len(gyro) < 16:
        return {"ok": False, "reason": "insufficient_samples"}
    ft = np.array([int(f["sensor_timestamp_ns"]) for f in frames], dtype=np.int64)
    blur = np.array([float(d.get("blur", 0.0)) for d in diag[: len(frames)]], dtype=np.float64)
    if blur.size < ft.size:
        blur = np.pad(blur, (0, ft.size - blur.size))
    gt = np.array([int(g["timestamp_ns"]) for g in gyro], dtype=np.int64)
    ge = np.array([abs(g["x"]) + abs(g["y"]) + abs(g["z"]) for g in gyro], dtype=np.float64)
    g_ip = TimeInterpolator(gt, ge.reshape(-1, 1))
    best = None
    for off_ms in np.linspace(-window_ms, window_ms, 41):
        off = int(off_ms * 1e6)
        sampled = np.array([g_ip.at(int(t + off))[0] for t in ft])
        if sampled.std() < 1e-9 or blur.std() < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(sampled, blur[: sampled.size])[0, 1])
        if best is None or corr > best[0]:
            best = (corr, float(off_ms))
    return {"ok": True, "best_offset_ms": best[1] if best else 0.0, "correlation": best[0] if best else 0.0, "window_ms": window_ms}
