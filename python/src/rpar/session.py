"""Session bundle I/O, checksums, crash-safe segments (REC-001..007, NFR-010)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from rpar import SCHEMA_VERSION
from rpar.enums import PrivacyMode, RunMode
from rpar.models import AlertDecision, FrameMeta, MountProfile, TrackedRoadObject, _to_jsonable
from rpar.timebase import percentile_intervals_ms


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("wb" if False else "rb") as f:  # noqa: SIM115
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass
class SessionManifest:
    schema_version: str
    session_id: str
    device: dict[str, Any]
    camera_profile: str
    mount_profile_id: str
    calibration_hash: str
    model_packages: list[str]
    start_elapsed_realtime_ns: int
    end_elapsed_realtime_ns: int | None
    video_segments: int
    capability_report_hash: str
    privacy_mode: str
    run_mode: str
    config_snapshot: dict[str, Any]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionWriter:
    def __init__(self, root: Path, manifest: SessionManifest, mount: MountProfile) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.mount = mount
        self._closed = False
        self._seg_index = 0
        self._video: cv2.VideoWriter | None = None
        self._seg_frames = 0
        self._seg_limit = 300 * 60  # overridden by fps * segment_seconds later
        self._fps = 30
        self._size = (1920, 1080)
        self._init_dirs()

    def _init_dirs(self) -> None:
        for p in [
            self.root,
            self.root / "calibration",
            self.root / "video",
            self.root / "camera",
            self.root / "imu",
            self.root / "location",
            self.root / "perception",
            self.root / "events",
            self.root / "events" / "clips",
            self.root / "diagnostics",
        ]:
            p.mkdir(parents=True, exist_ok=True)
        (self.root / "calibration" / "mount_profile.json").write_text(
            json.dumps(self.mount.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._write_manifest()
        for name in [
            "camera/frame_metadata.jsonl",
            "imu/gyro.jsonl",
            "imu/accelerometer.jsonl",
            "imu/rotation_vector.jsonl",
            "location/location.jsonl",
            "perception/observations.jsonl",
            "perception/tracks.jsonl",
            "events/alerts.jsonl",
            "events/marks.jsonl",
            "diagnostics/runtime.jsonl",
            "diagnostics/events.jsonl",
        ]:
            (self.root / name).touch()

    def _write_manifest(self) -> None:
        tmp = self.root / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.root / "manifest.json")

    def open_segment(self, fps: int, size: tuple[int, int], segment_frames: int) -> None:
        self.close_segment()
        path = self.root / "video" / f"segment_{self._seg_index:03d}.mp4"
        tmp = self.root / "video" / f"segment_{self._seg_index:03d}.part.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video = cv2.VideoWriter(str(tmp), fourcc, fps, size)
        self._seg_frames = 0
        self._seg_limit = segment_frames
        self._fps = fps
        self._size = size
        self._part_path = tmp
        self._final_path = path

    def write_frame(
        self,
        bgr: np.ndarray,
        meta: FrameMeta,
        imu: dict[str, Any] | None,
        location: dict[str, Any] | None,
        tracks: list[TrackedRoadObject] | None,
        alerts: list[AlertDecision] | None,
        diagnostics: dict[str, Any] | None,
        observations: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._video is None:
            raise RuntimeError("segment not open")
        self._video.write(bgr)
        self._seg_frames += 1
        with (self.root / "camera" / "frame_metadata.jsonl").open("a", encoding="utf-8") as f:
            f.write(json_dumps(_to_jsonable(meta)) + "\n")
        if imu:
            kind = imu.get("sensor_type", "GYRO")
            mapping = {"GYRO": "gyro.jsonl", "ACCEL": "accelerometer.jsonl", "ROTATION_VECTOR": "rotation_vector.jsonl"}
            name = mapping.get(str(kind), "gyro.jsonl")
            with (self.root / "imu" / name).open("a", encoding="utf-8") as f:
                f.write(json_dumps(imu) + "\n")
        if location:
            with (self.root / "location" / "location.jsonl").open("a", encoding="utf-8") as f:
                f.write(json_dumps(location) + "\n")
        if observations:
            with (self.root / "perception" / "observations.jsonl").open("a", encoding="utf-8") as f:
                for o in observations:
                    f.write(json_dumps(o) + "\n")
        if tracks:
            with (self.root / "perception" / "tracks.jsonl").open("a", encoding="utf-8") as f:
                for t in tracks:
                    f.write(json_dumps(t.to_dict()) + "\n")
        if alerts:
            with (self.root / "events" / "alerts.jsonl").open("a", encoding="utf-8") as f:
                for a in alerts:
                    f.write(json_dumps(a.to_dict()) + "\n")
        if diagnostics:
            with (self.root / "diagnostics" / "runtime.jsonl").open("a", encoding="utf-8") as f:
                f.write(json_dumps(diagnostics) + "\n")
        if self._seg_frames >= self._seg_limit:
            self.close_segment()
            self.open_segment(self._fps, self._size, self._seg_limit)

    def write_imu(self, sample: dict[str, Any]) -> None:
        kind = str(sample.get("sensor_type", "GYRO"))
        mapping = {"GYRO": "gyro.jsonl", "ACCEL": "accelerometer.jsonl", "ROTATION_VECTOR": "rotation_vector.jsonl"}
        name = mapping.get(kind, "gyro.jsonl")
        with (self.root / "imu" / name).open("a", encoding="utf-8") as f:
            f.write(json_dumps(sample) + "\n")

    def write_event(self, event: dict[str, Any]) -> None:
        with (self.root / "diagnostics" / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json_dumps(event) + "\n")

    def write_mark(self, timestamp_ns: int, note: str, frame_id: int | None = None) -> None:
        with (self.root / "events" / "marks.jsonl").open("a", encoding="utf-8") as f:
            f.write(json_dumps({"timestamp_ns": timestamp_ns, "note": note, "frame_id": frame_id}) + "\n")

    def imu_interval_report(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, key in (
            ("gyro.jsonl", "gyro"),
            ("accelerometer.jsonl", "accel"),
            ("rotation_vector.jsonl", "rotation_vector"),
        ):
            path = self.root / "imu" / name
            ts: list[int] = []
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ts.append(int(row["timestamp_ns"]))
            out[key] = percentile_intervals_ms(ts)
        return out

    def close_segment(self) -> None:
        if self._video is None:
            return
        self._video.release()
        self._video = None
        if self._part_path.exists():
            self._part_path.replace(self._final_path)
        self.manifest.video_segments = self._seg_index + 1
        self._seg_index += 1
        self._write_manifest()

    def finalize(self, end_ns: int) -> Path:
        self.close_segment()
        self.manifest.end_elapsed_realtime_ns = end_ns
        intervals = self.imu_interval_report()
        (self.root / "diagnostics" / "imu_intervals.json").write_text(
            json.dumps(intervals, indent=2), encoding="utf-8"
        )
        self._write_manifest()
        write_checksums(self.root)
        self._closed = True
        return self.root


def write_checksums(root: Path) -> Path:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "checksums.sha256" and not p.name.endswith(".tmp"):
            rel = p.relative_to(root).as_posix()
            lines.append(f"{sha256_file(p)}  {rel}")
    out = root / "checksums.sha256"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_session(root: Path) -> dict[str, Any]:
    root = Path(root)
    report: dict[str, Any] = {"ok": True, "errors": [], "warnings": [], "files": {}}
    # recover truncated last segment even if the process died before manifest finalize
    for part in (root / "video").glob("*.part.mp4"):
        report["warnings"].append(f"unfinalized segment {part.name}")
        dest = Path(str(part).replace(".part.mp4", ".mp4.recovered"))
        shutil.copy2(part, dest)
        report["warnings"].append(f"copied to {dest.name} (marked recovered)")
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        report["ok"] = False
        report["errors"].append("missing manifest.json")
        return report
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report["manifest"] = manifest
    chk = root / "checksums.sha256"
    if chk.exists():
        for line in chk.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            path = root / rel
            if not path.exists():
                report["ok"] = False
                report["errors"].append(f"missing {rel}")
                continue
            actual = sha256_file(path)
            report["files"][rel] = actual == digest
            if actual != digest:
                report["ok"] = False
                report["errors"].append(f"checksum mismatch {rel}")
    else:
        report["warnings"].append("no checksums.sha256")
    cam = root / "camera" / "frame_metadata.jsonl"
    if cam.exists():
        ts = []
        for line in cam.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ts.append(json.loads(line).get("sensor_timestamp_ns") or json.loads(line).get("sensor_timestamp_ns", 0))
        report["camera_interval"] = percentile_intervals_ms([int(t) for t in ts if t])
    return report


def new_session_id(device_model: str = "PKC110") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{device_model}"


def estimate_hours_remaining(free_bytes: int, bitrate_mbps: float = 25.0) -> float:
    bytes_per_hour = bitrate_mbps * 1e6 / 8.0 * 3600.0
    return max(0.0, free_bytes / bytes_per_hour)


def export_bundle(src: Path, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(dest.with_suffix("")), "zip", root_dir=src)
    return Path(archive)


def copy_resumable(src: Path, dest: Path, chunk: int = 8 * 1024 * 1024) -> Path:
    """Resume a large copy from dest's current size (REC-006)."""
    src = Path(src)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = dest.stat().st_size if dest.exists() else 0
    with src.open("rb") as inf, dest.open("ab") as out:
        inf.seek(pos)
        while True:
            buf = inf.read(chunk)
            if not buf:
                break
            out.write(buf)
    return dest


def export_split_zip(src: Path, dest_prefix: Path, max_bytes: int = 512 * 1024 * 1024) -> list[Path]:
    """Pack a session into size-capped zip volumes for copy/resume (REC-006)."""
    import zipfile

    src = Path(src)
    dest_prefix = Path(dest_prefix)
    dest_prefix.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(src.rglob("*")) if p.is_file()]
    parts: list[Path] = []
    index = 1
    current: list[Path] = []
    size = 0
    def flush(batch: list[Path], n: int) -> Path:
        out = dest_prefix.parent / f"{dest_prefix.name}.part{n:02d}.zip"
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in batch:
                zf.write(f, f.relative_to(src).as_posix())
        return out

    for f in files:
        n = f.stat().st_size
        if current and size + n > max_bytes:
            parts.append(flush(current, index))
            index += 1
            current, size = [], 0
        current.append(f)
        size += n
    if current:
        parts.append(flush(current, index))
    return parts


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        yield from ()
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
