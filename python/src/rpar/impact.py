"""M5 impact interface: same-frame spike plus visual→future IMU alignment. Never used for alerts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from rpar import SCHEMA_VERSION

RELIEF_GEOMETRIES = frozenset({"concave", "convex", "rough", "step"})
PATCH_SEMANTICS = frozenset({"repair_patch", "road_joint"})
_CONFIRMED = frozenset({"CONFIRMED", "ALERTED"})


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


def vertical_residual(z: float, gravity: float = 9.81) -> float:
    return abs(float(z) - gravity)


def impact_label(peak_ms2: float, speed_mps: float) -> str:
    """Speed-normalized vertical peak → none/weak/medium/strong. Faster rides expect more vibration."""
    scale = max(float(speed_mps) / 10.0, 0.35)
    adj = float(peak_ms2) / scale
    if adj < 2.0:
        return "none"
    if adj < 4.0:
        return "weak"
    if adj < 7.0:
        return "medium"
    return "strong"


def _first_confirmed(tracks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    first: dict[int, dict[str, Any]] = {}
    for t in tracks:
        if str(t.get("lifecycle_state") or "") not in _CONFIRMED:
            continue
        try:
            tid = int(t["track_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if tid not in first:
            first[tid] = t
    return first


def _window_ns(
    t0: int,
    distance_m: float | None,
    speed_mps: float,
    horizon_s: float,
) -> tuple[int, int, float | None]:
    """Prefer the expected pass instant (distance/speed); else the 0–horizon window."""
    if distance_m is not None and speed_mps > 0:
        t_pass = float(distance_m) / float(speed_mps)
        if 0.0 <= t_pass <= horizon_s + 0.75:
            half = 0.40
            start = t0 + int(max(0.0, t_pass - half) * 1e9)
            end = t0 + int((t_pass + half) * 1e9)
            return start, end, t_pass
    return t0, t0 + int(horizon_s * 1e9), None


def _stats(zs: list[float]) -> tuple[float, float]:
    if not zs:
        return 0.0, 0.0
    peak = max(zs)
    rms = (sum(z * z for z in zs) / len(zs)) ** 0.5
    return float(peak), float(rms)


def align_tracks_to_future_impact(
    tracks: list[dict[str, Any]],
    accel: list[dict[str, Any]],
    *,
    horizon_s: float = 3.0,
    gravity: float = 9.81,
    min_speed_mps: float = 2.0,
    speed_mps: float | None = None,
) -> list[dict[str, Any]]:
    """Weak-supervise each first-confirmed track with IMU in the 0–3 s (or pass) window.

    Spec M5: visual object + speed → future impact intensity. IMU is the label, never an alert.
    """
    first = _first_confirmed(tracks)
    accel_sorted = sorted(accel, key=lambda s: int(s.get("timestamp_ns") or 0))
    default_speed = float(speed_mps) if speed_mps is not None else 0.0
    rows: list[dict[str, Any]] = []
    for tid, t in first.items():
        raw_speed = t.get("speed_mps")
        spd = float(raw_speed) if raw_speed is not None else default_speed
        if spd < min_speed_mps:
            spd = default_speed if default_speed >= min_speed_mps else spd
        if spd < min_speed_mps:
            continue
        t0 = int(t.get("timestamp_ns") or 0)
        dist = t.get("distance_m")
        dist_f = float(dist) if dist is not None else None
        start_ns, end_ns, t_pass = _window_ns(t0, dist_f, spd, horizon_s)
        zs: list[float] = []
        for s in accel_sorted:
            ts = int(s.get("timestamp_ns") or 0)
            if start_ns <= ts <= end_ns:
                zs.append(vertical_residual(float(s.get("z") or 0.0), gravity))
        peak, rms = _stats(zs)
        geom = str(t.get("geometry_type") or "unknown")
        sem = str(t.get("semantic_type") or "unknown_anomaly")
        rows.append(
            {
                "track_id": tid,
                "timestamp_ns": t0,
                "horizon_s": horizon_s,
                "expected_pass_s": t_pass,
                "speed_mps": spd,
                "distance_m": dist_f,
                "peak_ms2": peak,
                "rms_ms2": rms,
                "n_imu": len(zs),
                "impact_label": impact_label(peak, spd),
                "used_for_alert": False,
                "geometry_type": geom,
                "semantic_type": sem,
            }
        )
    return rows


def separates_relief_from_patch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Offline M5 check: future IMU peak is higher on concave/convex/rough than on flat patches."""
    relief = [float(r["peak_ms2"]) for r in rows if str(r.get("geometry_type")) in RELIEF_GEOMETRIES]
    patch = [
        float(r["peak_ms2"])
        for r in rows
        if str(r.get("semantic_type")) in PATCH_SEMANTICS or str(r.get("geometry_type")) == "flat"
    ]
    mean_r = sum(relief) / len(relief) if relief else None
    mean_p = sum(patch) / len(patch) if patch else None
    delta = (mean_r - mean_p) if mean_r is not None and mean_p is not None else None
    ok = bool(relief and patch and delta is not None and delta > 0.5)
    return {
        "n_relief": len(relief),
        "n_patch": len(patch),
        "mean_relief_peak_ms2": mean_r,
        "mean_patch_peak_ms2": mean_p,
        "delta_ms2": delta,
        "pass": ok,
        "used_for_alert": False,
        "note": "Synthetic or session IMU; not a trained PKC110 impact net.",
    }


def align_session_impact(session_dir: Path, *, horizon_s: float = 3.0) -> dict[str, Any]:
    from rpar.session import iter_jsonl

    root = Path(session_dir)
    tracks = list(iter_jsonl(root / "perception" / "tracks.jsonl"))
    accel = list(iter_jsonl(root / "imu" / "accelerometer.jsonl"))
    loc = list(iter_jsonl(root / "location" / "location.jsonl"))
    speeds = [float(r["speed_mps"]) for r in loc if r.get("speed_mps") is not None]
    speeds.sort()
    speed = speeds[len(speeds) // 2] if speeds else None
    rows = align_tracks_to_future_impact(tracks, accel, horizon_s=horizon_s, speed_mps=speed)
    labels = dict(Counter(r["impact_label"] for r in rows))
    sep = separates_relief_from_patch(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "session": str(root),
        "horizon_s": horizon_s,
        "speed_mps": speed,
        "n_aligned": len(rows),
        "labels": labels,
        "used_for_alert": False,
        "separation": sep,
        "rows": rows,
    }


def write_impact_report(session_dir: Path, out_json: Path, *, horizon_s: float = 3.0) -> dict[str, Any]:
    import json

    report = align_session_impact(session_dir, horizon_s=horizon_s)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def synthetic_future_impact_proof(*, duration_s: float = 4.0, horizon_s: float = 3.0) -> dict[str, Any]:
    """Default synthetic world: IMU spikes when the ego passes relief, not flat patches."""
    from rpar.simulator import RoadSimulator, SimConfig

    sim = RoadSimulator(SimConfig(duration_s=duration_s, fps=30, blur_windows=[]))
    t0 = sim.t0_ns
    tracks: list[dict[str, Any]] = []
    for i, obj in enumerate(sim.objects, start=1):
        tracks.append(
            {
                "track_id": i,
                "timestamp_ns": t0,
                "lifecycle_state": "CONFIRMED",
                "semantic_type": obj.semantic.value,
                "geometry_type": obj.geometry.value,
                "distance_m": obj.y0_m,
                "speed_mps": sim.sim.speed_mps,
            }
        )
    accel: list[dict[str, Any]] = []
    n = int(duration_s * 200)
    for k in range(n):
        t = k / 200.0
        _, acc = sim.motion_at(t)
        accel.append({"timestamp_ns": t0 + int(t * 1e9), "z": acc[2]})
    rows = align_tracks_to_future_impact(tracks, accel, horizon_s=horizon_s, speed_mps=sim.sim.speed_mps)
    sep = separates_relief_from_patch(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "synthetic_pass_window",
        "horizon_s": horizon_s,
        "n_aligned": len(rows),
        "labels": dict(Counter(r["impact_label"] for r in rows)),
        "used_for_alert": False,
        "separation": sep,
        "rows": rows,
        "note": "Simulator IMU at object pass. Not PKC110 field evidence.",
    }
