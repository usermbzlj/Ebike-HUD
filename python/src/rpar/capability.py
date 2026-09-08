"""Device capability report schema (CAP-001..007). Desktop can merge/display phone exports."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpar import SCHEMA_VERSION


def percentile(sorted_asc: list[float], p: float) -> float:
    if not sorted_asc:
        return 0.0
    i = int((p / 100.0) * (len(sorted_asc) - 1))
    i = max(0, min(len(sorted_asc) - 1, i))
    return float(sorted_asc[i])


def cpu_bench_window(duration_s: float = 0.12, n: int = 36) -> dict[str, Any]:
    """CAP-005 desktop CPU window. Phone 10 min bench is Android LiteRTBench."""
    import time

    a = [[float((i * n + j) % 17) for j in range(n)] for i in range(n)]
    times: list[float] = []
    first = 0.0
    t0 = time.perf_counter()
    end = t0 + max(0.02, duration_s)
    it = 0
    while time.perf_counter() < end or it < 2:
        t1 = time.perf_counter()
        c = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                acc = 0.0
                row = a[i]
                for j in range(n):
                    acc += row[j] * a[j][k]
                c[i][k] = acc
        ms = (time.perf_counter() - t1) * 1000.0
        if it == 0:
            first = ms
        else:
            times.append(ms)
        it += 1
        if it > 20_000:
            break
    times.sort()
    ran = time.perf_counter() - t0
    requested = 600.0
    status = "ok" if ran >= requested * 0.95 else "short_probe"
    return {
        "backend": "CPU",
        "status": "ok",
        "first_ms": first,
        "p50_ms": percentile(times, 50.0),
        "p95_ms": percentile(times, 95.0),
        "n_iters": len(times),
        "ran_s": ran,
        "stable_10min": {
            "status": status,
            "requested_s": requested,
            "ran_s": ran,
            "first_ms": first,
            "p50_ms": percentile(times, 50.0),
            "p95_ms": percentile(times, 95.0),
            "note": "Desktop stub is short_probe. PKC110 capability screen runs requested_s=600.",
        },
        "source": "desktop_stub",
    }


def _desktop_cpu_microbench() -> list[dict[str, Any]]:
    cpu = cpu_bench_window()
    return [
        cpu,
        {"backend": "GPU", "status": "unavailable_on_desktop"},
        {"backend": "NPU", "status": "unavailable_on_desktop"},
    ]


def desktop_capability_stub() -> dict[str, Any]:
    """Offline placeholder. Real Camera2/IMU/LiteRT numbers come from the Android probe."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "desktop_stub",
        "device": {
            "manufacturer": "runtime-detected",
            "model": "PKC110",
            "product": "Find X8 Pro",
            "os_version": "runtime-detected",
            "host_probe": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
        },
        "camera": {
            "hardware_level": "unavailable_on_desktop",
            "cameras": [
                {
                    "id": "rear_main",
                    "logical": True,
                    "facing": "BACK",
                    "outputs": [
                        {"format": "YUV_420_888", "sizes": ["1920x1080", "1280x720"], "fps": [30, 60]},
                        {"format": "PRIVATE", "sizes": ["1920x1080"], "fps": [30, 60]},
                    ],
                    "ois": "runtime-detected",
                    "video_stabilization": "runtime-detected",
                    "preview_stabilization": "runtime-detected",
                    "dynamic_range": ["SDR"],
                    "intrinsics": "runtime-detected",
                    "rolling_shutter": "runtime-detected",
                    "ae_target_fps_ranges": [{"min": 30, "max": 60}, {"min": 30, "max": 30}],
                    "concurrent_streams": [
                        {
                            "combo": "preview+yuv+record@1080p60",
                            "yuv": "1920x1080",
                            "record": "1920x1080",
                            "requested_fps": 60,
                            "max_fps_from_duration": 60,
                            "status": "candidate",
                        },
                        {
                            "combo": "preview+yuv+record@1080p30",
                            "yuv": "1920x1080",
                            "record": "1920x1080",
                            "requested_fps": 30,
                            "max_fps_from_duration": 60,
                            "status": "candidate",
                        },
                    ],
                }
            ],
            "concurrent_streams": [
                {
                    "combo": "preview+yuv+record@1080p60",
                    "status": "probe_on_device",
                    "requested_fps": 60,
                    "actual_fps": None,
                    "measured_yuv_fps": None,
                    "measured_n": 0,
                    "size": "1920x1080",
                    "record_ok": None,
                }
            ],
        },
        "sensors": {
            "gyro": {"max_hz": None, "actual_hz": None, "batch_delay_ms": None, "status": "probe_on_device"},
            "accel": {"max_hz": None, "actual_hz": None, "status": "probe_on_device"},
            "rotation_vector": {"max_hz": None, "actual_hz": None, "status": "probe_on_device"},
            "high_sampling_rate_permission": "probe_on_device",
        },
        "acceleration": {
            "candidates": ["CPU", "GPU", "NPU"],
            "note": "LiteRT windowed microbench on PKC110; desktop records a short CPU GEMM stub (stable_10min.short_probe until the phone 10 min run)",
            "results": _desktop_cpu_microbench(),
        },
        "arcore": {"available": "probe_on_device", "depth": "probe_on_device", "required": False},
        "audio": {"tts": True, "bluetooth": "probe_on_device", "routes": ["speaker", "headset"]},
        "availability_flags": {
            "ois": "unavailable_until_probe",
            "preview_stabilization": "unavailable_until_probe",
            "npu": "unavailable_until_probe",
        },
    }


def write_capability_report(path: Path, report: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report or desktop_capability_stub(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
