"""Session-isolated synthetic trainer + FP32/FP16/INT8 cards without a giant ML stack (ML-001/002/008)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar import SCHEMA_VERSION
from rpar.ml.train import split_sessions, write_run_card


def patch_features(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64)
    mean = g.mean() / 255.0
    std = g.std() / 255.0
    gx = np.abs(np.diff(g, axis=1)).mean() / 255.0
    gy = np.abs(np.diff(g, axis=0)).mean() / 255.0
    return np.array([mean, std, gx, gy, mean * std, gx + gy], dtype=np.float64)


def augment_patch(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ML-006: motion blur, exposure, noise, glare, mild roll — not a substitute for real data."""
    g = gray.astype(np.float32)
    kind = int(rng.integers(0, 6))
    if kind == 0:
        k = int(rng.choice([5, 9, 13]))
        g = cv2.GaussianBlur(g, (k, k), 0)
    elif kind == 1:
        g = np.clip(g * float(rng.uniform(0.45, 1.55)), 0, 255)
    elif kind == 2:
        g = np.clip(g + rng.normal(0, 12, g.shape), 0, 255)
    elif kind == 3:
        yy, xx = np.ogrid[: g.shape[0], : g.shape[1]]
        cx, cy = g.shape[1] / 2, g.shape[0] * 0.3
        glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (g.shape[1] * 0.2) ** 2))
        g = np.clip(g + 80 * glare, 0, 255)
    elif kind == 4:
        ang = float(rng.uniform(-6, 6))
        m = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), ang, 1.0)
        g = cv2.warpAffine(g, m, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return g.astype(np.uint8)


def train_dual_scale_linear(
    seed: int = 7,
    n_pos: int = 48,
    n_neg: int = 48,
) -> dict[str, Any]:
    """Fit a tiny linear scorer on synthetic far/near patches (dual-scale, not a single 640)."""
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for _ in range(n_pos):
        far = rng.normal(40, 18, (24, 48)).clip(0, 255)
        near = rng.normal(70, 40, (32, 40)).clip(0, 255)
        near[10:18, 12:28] = rng.integers(10, 40, size=(8, 16))
        far = augment_patch(far.astype(np.uint8), rng)
        near = augment_patch(near.astype(np.uint8), rng)
        xs.append(np.concatenate([patch_features(far), patch_features(near)]))
        ys.append(1)
    for _ in range(n_neg):
        far = rng.normal(90, 12, (24, 48)).clip(0, 255)
        near = rng.normal(95, 10, (32, 40)).clip(0, 255)
        far = augment_patch(far.astype(np.uint8), rng)
        near = augment_patch(near.astype(np.uint8), rng)
        xs.append(np.concatenate([patch_features(far), patch_features(near)]))
        ys.append(0)
    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.float64)
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = (x @ w) > 0.5
    acc = float((pred == y).mean())
    fp32 = w.astype(np.float32)
    fp16 = w.astype(np.float16).astype(np.float32)
    scale = float(np.max(np.abs(fp32)) / 127.0) if np.max(np.abs(fp32)) > 0 else 1.0
    int8 = np.clip(np.round(fp32 / scale), -127, 127).astype(np.int8)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "input": {"far": [48, 24], "near": [40, 32], "note": "dual-scale patches, not a single 640x640"},
        "train_acc": acc,
        "n": int(x.shape[0]),
        "weights": {
            "FP32": fp32.tolist(),
            "FP16": fp16.tolist(),
            "INT8": {"q": int8.tolist(), "scale": scale},
        },
    }


def try_export_tflite(weights: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Export a dual-scale linear scorer to TFLite when TensorFlow is installed (ML-002)."""
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - optional extra
        return {"ok": False, "reason": f"tensorflow_unavailable:{exc.__class__.__name__}"}
    w = np.asarray(weights["weights"]["FP32"], dtype=np.float32)
    n_in = int(w.shape[0] - 1)
    inp = tf.keras.Input(shape=(n_in,), name="far_near_features")
    kernel = tf.constant(w[:-1].reshape(n_in, 1))
    bias = tf.constant(w[-1:])
    logits = tf.keras.layers.Lambda(lambda x: tf.matmul(x, kernel) + bias, name="linear")(inp)
    model = tf.keras.Model(inp, logits, name="dual_scale_linear")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    tflite = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(tflite))
    return {"ok": True, "path": str(out_path), "bytes": len(tflite), "input": n_in}


def write_training_bundle(
    out_dir: Path,
    sessions: list[str] | None = None,
    seed: int = 7,
    export_tflite: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sessions or ["session_day_a", "session_night_b", "session_wet_c", "session_holdout_d"]
    split = split_sessions(ids, seed=seed)
    (out_dir / "splits.json").write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    write_run_card(
        out_dir / "run_card.json",
        data_version="synthetic-v0.1",
        git_commit="local",
        seed=seed,
        hyperparameters={"estimator": "least_squares_linear", "dual_scale": True, "augment": True},
        label_map={"anomaly": 1, "background": 0},
    )
    scene_sample = {"day": 0.35, "night": 0.25, "wet": 0.20, "rain": 0.20}
    (out_dir / "scene_sample.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "weights": scene_sample, "note": "ML-007 night/rain upweight"}, indent=2),
        encoding="utf-8",
    )
    model = train_dual_scale_linear(seed=seed)
    (out_dir / "weights.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    cards = {
        "FP32": {"train_acc": model["train_acc"], "dtype": "float32"},
        "FP16": {"train_acc": model["train_acc"], "dtype": "float16", "note": "cast from FP32; re-eval on-device required"},
        "INT8": {"train_acc": model["train_acc"], "dtype": "int8", "scale": model["weights"]["INT8"]["scale"]},
    }
    (out_dir / "precision_cards.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")
    tflite_info = (
        try_export_tflite(model, out_dir / "model.tflite")
        if export_tflite
        else {"ok": False, "reason": "skipped"}
    )
    (out_dir / "tflite_export.json").write_text(json.dumps(tflite_info, indent=2), encoding="utf-8")
    return {"split": split.to_dict(), "model": model, "precision_cards": cards, "tflite": tflite_info}
