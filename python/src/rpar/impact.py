"""Reserved M5 impact interface: IMU spike near an object, never used for alerts."""

from __future__ import annotations

from typing import Any


def estimate_impact_score(
    linear_accel: tuple[float, float, float] | None,
    speed_mps: float | None,
    distance_m: float | None,
    *,
    gravity: float = 9.81,
    min_speed_mps: float = 2.0,
    near_m: float = 6.0,
    spike_g: float = 3.5,
) -> float | None:
    """Return 0–1 when a vertical accel spike coincides with a near object; else None."""
    if linear_accel is None or speed_mps is None or speed_mps < min_speed_mps:
        return None
    if distance_m is None or distance_m > near_m:
        return None
    az = abs(float(linear_accel[2]) - gravity)
    if az < spike_g:
        return None
    return float(min(1.0, az / 12.0))


def harvest_impact_windows(accel_samples: list[dict[str, Any]], near_track_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """M5: IMU vertical spikes as weak-supervision windows, never as alerts."""
    out: list[dict[str, Any]] = []
    for s in accel_samples:
        z = float(s.get("z") or 0.0)
        az = abs(z - 9.81)
        if az < 3.5:
            continue
        out.append(
            {
                "timestamp_ns": s.get("timestamp_ns"),
                "impact_score": float(min(1.0, az / 12.0)),
                "near_track_ids": list(near_track_ids or []),
                "used_for_alert": False,
            }
        )
    return out
