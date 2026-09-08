"""Dual-scale pixel classifier trained on simulator classmaps (ML-005). No TensorFlow in pytest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar import SCHEMA_VERSION
from rpar.ml.train import write_run_card
from rpar.roi import FAR, NEAR, crop_xyxy
from rpar.segdecode import CLASS_ROAD, classmap_to_result
from rpar.simulator import RoadSimulator, SimConfig


def _xyrgb(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    b, g, r = cv2.split(img)
    return np.stack(
        [r.astype(np.float64) / 255.0, g.astype(np.float64) / 255.0, b.astype(np.float64) / 255.0, xs / max(w, 1), ys / max(h, 1)],
        axis=-1,
    ).reshape(-1, 5)


def _crop_pair(bgr: np.ndarray, classmap: np.ndarray, box, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box
    rgb = bgr[y0:y1, x0:x1]
    cm = classmap[y0:y1, x0:x1]
    if rgb.size == 0:
        rgb = bgr
        cm = classmap
    return (
        cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
        cv2.resize(cm, size, interpolation=cv2.INTER_NEAREST),
    )


def collect_dual_scale(sim: RoadSimulator, n: int = 10, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    h, w = sim.sim.height, sim.sim.width
    far_box = crop_xyxy(w, h, FAR)
    near_box = crop_xyxy(w, h, NEAR)
    for i in range(0, min(sim.n_frames(), n * stride), stride):
        frame, _ = sim.frame_at(i)
        cm = sim.classmap_at(i)
        far_im, far_cm = _crop_pair(frame.bgr, cm, far_box, (48, 24))
        near_im, near_cm = _crop_pair(frame.bgr, cm, near_box, (40, 32))
        xs.append(_xyrgb(far_im))
        ys.append(far_cm.reshape(-1))
        xs.append(_xyrgb(near_im))
        ys.append(near_cm.reshape(-1))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fit_one_vs_rest(x: np.ndarray, y: np.ndarray, n_class: int = 4) -> np.ndarray:
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    w = np.zeros((n_class, xb.shape[1]), dtype=np.float64)
    for k in range(n_class):
        t = (y == k).astype(np.float64)
        wk, *_ = np.linalg.lstsq(xb, t, rcond=None)
        w[k] = wk
    return w


def predict_labels(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    return np.argmax(xb @ weights.T, axis=1)


def road_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred == CLASS_ROAD
    g = gt == CLASS_ROAD
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union else 0.0


def train_dual_scale_classmap(seed: int = 3) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=1.2, blur_windows=[]))
    x, y = collect_dual_scale(sim, n=8, stride=1)
    idx = rng.choice(x.shape[0], size=min(4000, x.shape[0]), replace=False)
    weights = fit_one_vs_rest(x[idx], y[idx])
    pred = predict_labels(x[idx], weights)
    acc = float((pred == y[idx]).mean())
    frame, _ = sim.frame_at(min(4, sim.n_frames() - 1))
    cm = sim.classmap_at(min(4, sim.n_frames() - 1))
    h, w = frame.bgr.shape[:2]
    small = cv2.resize(frame.bgr, (80, 45), interpolation=cv2.INTER_AREA)
    pred_small = predict_labels(_xyrgb(small), weights).reshape(45, 80)
    pred_full = cv2.resize(pred_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    iou = road_iou(pred_full, cm)
    decoded = classmap_to_result(pred_full, frame)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "input": {"far": [48, 24], "near": [40, 32], "note": "dual-scale crops, not a single 640x640"},
        "pixel_acc": acc,
        "road_iou": iou,
        "n_pixels": int(idx.size),
        "weights": weights.tolist(),
        "decoded_obs": len(decoded.observations),
        "has_road_polygon": len(decoded.road_polygon) >= 3,
    }


def write_seg_bundle(out_dir: Path, seed: int = 3) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = train_dual_scale_classmap(seed=seed)
    (out_dir / "seg_weights.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    write_run_card(
        out_dir / "run_card.json",
        data_version="synthetic-classmap-v0.1",
        git_commit="local",
        seed=seed,
        hyperparameters={"estimator": "one_vs_rest_lstsq", "dual_scale": True},
        label_map={"bg": 0, "road": 1, "anomaly": 2, "occ": 3},
    )
    return model
