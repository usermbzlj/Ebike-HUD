"""Monotonic timebase, interpolation and drop/reorder detection (SYNC-001..006)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


def ns_to_s(ts_ns: int) -> float:
    return ts_ns / NS_PER_S


def s_to_ns(t: float) -> int:
    return int(round(t * NS_PER_S))


@dataclass(slots=True)
class SyncAnomaly:
    timestamp_ns: int
    code: str
    detail: str


class TimeInterpolator:
    """Piecewise-linear interpolator on a strictly monotonic nanosecond axis."""

    def __init__(self, timestamps_ns: np.ndarray, values: np.ndarray) -> None:
        if timestamps_ns.size == 0:
            raise ValueError("empty interpolator")
        order = np.argsort(timestamps_ns)
        self._t = timestamps_ns[order].astype(np.int64, copy=False)
        self._v = np.asarray(values, dtype=np.float64)[order]

    def at(self, t_ns: int) -> np.ndarray:
        t = np.int64(t_ns)
        if t <= self._t[0]:
            return self._v[0].copy()
        if t >= self._t[-1]:
            return self._v[-1].copy()
        i = int(np.searchsorted(self._t, t, side="right") - 1)
        t0 = self._t[i]
        t1 = self._t[i + 1]
        alpha = (t - t0) / max(t1 - t0, 1)
        return (1.0 - alpha) * self._v[i] + alpha * self._v[i + 1]


class QuaternionInterpolator:
    def __init__(self, timestamps_ns: np.ndarray, quats_xyzw: np.ndarray) -> None:
        self._t = np.asarray(timestamps_ns, dtype=np.int64)
        q = np.asarray(quats_xyzw, dtype=np.float64)
        n = np.linalg.norm(q, axis=1, keepdims=True)
        n = np.clip(n, 1e-9, None)
        self._q = q / n
        for i in range(1, len(self._q)):
            if np.dot(self._q[i - 1], self._q[i]) < 0:
                self._q[i] *= -1

    def at(self, t_ns: int) -> np.ndarray:
        t = np.int64(t_ns)
        if t <= self._t[0]:
            return self._q[0].copy()
        if t >= self._t[-1]:
            return self._q[-1].copy()
        i = int(np.searchsorted(self._t, t, side="right") - 1)
        t0, t1 = self._t[i], self._t[i + 1]
        u = float((t - t0) / max(t1 - t0, 1))
        q0, q1 = self._q[i], self._q[i + 1]
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if dot > 0.9995:
            out = q0 + u * (q1 - q0)
            return out / np.linalg.norm(out)
        theta = np.arccos(dot)
        so = np.sin(theta)
        out = (np.sin((1 - u) * theta) * q0 + np.sin(u * theta) * q1) / so
        return out / np.linalg.norm(out)


def match_capture_results(
    frame_ts: list[int],
    result_ts: list[int],
    max_dt_ns: int = 2_000_000,
) -> tuple[list[tuple[int, int]], list[SyncAnomaly]]:
    """Match SENSOR_TIMESTAMP to CaptureResult timestamps (SYNC-002)."""
    anomalies: list[SyncAnomaly] = []
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    result_index = {t: i for i, t in enumerate(result_ts)}
    prev = None
    for i, ts in enumerate(frame_ts):
        if prev is not None and ts < prev:
            anomalies.append(SyncAnomaly(ts, "REORDER", f"frame {i} timestamp went backwards"))
        if prev is not None and ts == prev:
            anomalies.append(SyncAnomaly(ts, "DUPLICATE", f"frame {i} duplicated timestamp"))
        prev = ts
        if ts in result_index and result_index[ts] not in used:
            used.add(result_index[ts])
            pairs.append((i, result_index[ts]))
            continue
        best_j, best_dt = None, None
        for j, rts in enumerate(result_ts):
            if j in used:
                continue
            dt = abs(int(rts) - int(ts))
            if best_dt is None or dt < best_dt:
                best_dt, best_j = dt, j
        if best_j is not None and best_dt is not None and best_dt <= max_dt_ns:
            used.add(best_j)
            pairs.append((i, best_j))
        else:
            anomalies.append(SyncAnomaly(ts, "UNMATCHED_FRAME", f"frame {i} has no CaptureResult"))
    for j, rts in enumerate(result_ts):
        if j not in used:
            anomalies.append(SyncAnomaly(rts, "ORPHAN_RESULT", f"result {j} unmatched"))
    return pairs, anomalies


def detect_gaps(timestamps_ns: list[int], expected_hz: float, stall_factor: float = 4.0) -> list[SyncAnomaly]:
    if len(timestamps_ns) < 2:
        return []
    expected = NS_PER_S / max(expected_hz, 1e-3)
    out: list[SyncAnomaly] = []
    prev = timestamps_ns[0]
    for ts in timestamps_ns[1:]:
        dt = ts - prev
        if dt > expected * stall_factor:
            out.append(SyncAnomaly(ts, "SENSOR_STALL", f"gap {dt / NS_PER_MS:.1f} ms"))
        if dt < 0:
            out.append(SyncAnomaly(ts, "TIMESTAMP_JUMP", "negative dt"))
        prev = ts
    return out


def percentile_intervals_ms(timestamps_ns: list[int]) -> dict[str, float]:
    if len(timestamps_ns) < 2:
        return {"p5_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "hz": 0.0}
    dts = np.diff(np.asarray(timestamps_ns, dtype=np.int64)) / NS_PER_MS
    dts = dts[dts > 0]
    if dts.size == 0:
        return {"p5_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "hz": 0.0}
    return {
        "p5_ms": float(np.percentile(dts, 5)),
        "p50_ms": float(np.percentile(dts, 50)),
        "p95_ms": float(np.percentile(dts, 95)),
        "hz": 1000.0 / float(np.median(dts)),
    }


def alignment_error_p95_ms(camera_ts: list[int], imu_ts: list[int]) -> float:
    """Nearest IMU sample error for each camera timestamp (SYNC-003)."""
    if not camera_ts or not imu_ts:
        return float("inf")
    imu = np.asarray(imu_ts, dtype=np.int64)
    err = []
    for t in camera_ts:
        j = int(np.searchsorted(imu, t))
        candidates = []
        if j < len(imu):
            candidates.append(abs(int(imu[j]) - t))
        if j > 0:
            candidates.append(abs(int(imu[j - 1]) - t))
        err.append(min(candidates) / NS_PER_MS)
    return float(np.percentile(np.asarray(err), 95))
