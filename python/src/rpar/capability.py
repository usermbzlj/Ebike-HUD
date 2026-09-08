"""Device capability report schema (CAP-001..007). Desktop can merge/display phone exports."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpar import SCHEMA_VERSION


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
                }
            ],
            "concurrent_streams": [
                {"combo": "preview+yuv+record", "status": "probe_on_device", "actual_fps": None}
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
            "note": "LiteRT microbench runs on PKC110 only",
            "results": [],
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
