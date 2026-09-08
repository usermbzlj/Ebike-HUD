"""CAM-012 crash / violent-vibration detector. Matches Android CrashDetect.kt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

GRAVITY = 9.81
RESIDUAL_THRESHOLD = 40.0
GYRO_THRESHOLD = 15.0


def accel_magnitude(ax: float, ay: float, az: float) -> float:
    return float((ax * ax + ay * ay + az * az) ** 0.5)


def linear_accel_residual(ax: float, ay: float, az: float, gravity: float = GRAVITY) -> float:
    return abs(accel_magnitude(ax, ay, az) - gravity)


def severe_impact(
    ax: float,
    ay: float,
    az: float,
    gyro_rad_s: float = 0.0,
    residual_threshold: float = RESIDUAL_THRESHOLD,
    gyro_threshold: float = GYRO_THRESHOLD,
) -> bool:
    return linear_accel_residual(ax, ay, az) >= residual_threshold or abs(gyro_rad_s) >= gyro_threshold


def scan_accel_jsonl(path: Path, limit: int = 50_000) -> dict[str, Any]:
    """Offline CAM-012 pass over a session accelerometer jsonl. Does not change alerts."""
    p = Path(path)
    hits: list[dict[str, Any]] = []
    n = 0
    if p.is_file():
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                if n > limit:
                    break
                row = json.loads(line)
                ax = float(row.get("x") or 0.0)
                ay = float(row.get("y") or 0.0)
                az = float(row.get("z") or 0.0)
                if severe_impact(ax, ay, az):
                    hits.append(
                        {
                            "timestamp_ns": row.get("timestamp_ns"),
                            "residual": linear_accel_residual(ax, ay, az),
                        }
                    )
    return {
        "n_samples": n,
        "n_hits": len(hits),
        "first_hit_ns": hits[0]["timestamp_ns"] if hits else None,
        "used_for_alert": False,
        "note": "CAM-012 offline scan. Phone runtime stops capture; this report is diagnostic only.",
    }
