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


def eval_classmap_slices(weights: np.ndarray) -> dict[str, Any]:
    from rpar.segengine import DualScaleSegEngine

    engine = DualScaleSegEngine(weights)
    mapping = {
        "day": SimConfig(width=320, height=180, fps=10, duration_s=0.4, blur_windows=[]),
        "night": SimConfig(width=320, height=180, fps=10, duration_s=0.4, night=True, blur_windows=[]),
        "wet": SimConfig(width=320, height=180, fps=10, duration_s=0.4, wet=True, blur_windows=[]),
    }
    slices: dict[str, Any] = {}
    for name, sc in mapping.items():
        sim = RoadSimulator(sc)
        frame, _ = sim.frame_at(0)
        pred = engine.classmap(frame.bgr)
        slices[name] = {"road_iou": round(road_iou(pred, sim.classmap_at(0)), 4)}
    return {"schema_version": SCHEMA_VERSION, "slices": slices}


def classmap_precision_cards(weights: np.ndarray) -> dict[str, Any]:
    """ML-008: FP32 / FP16 / INT8 cards on the same synthetic frame, not a PKC110 number."""
    from rpar.segengine import DualScaleSegEngine

    w = np.asarray(weights, dtype=np.float32)
    fp16 = w.astype(np.float16).astype(np.float32)
    scale = float(np.max(np.abs(w)) / 127.0) or 1.0
    w_int8 = np.clip(np.round(w / scale), -127, 127).astype(np.int8).astype(np.float32) * scale
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=0.3, blur_windows=[]))
    frame, _ = sim.frame_at(0)
    gt = sim.classmap_at(0)
    cards: dict[str, Any] = {}
    for name, ww in (("FP32", w), ("FP16", fp16), ("INT8", w_int8)):
        pred = DualScaleSegEngine(ww).classmap(frame.bgr)
        cards[name] = {
            "road_iou": round(road_iou(pred, gt), 4),
            "note": "synthetic frame; INT8 is weight-cast. Re-eval on PKC110 Camera2 before release.",
        }
    return {"schema_version": SCHEMA_VERSION, "cards": cards}


def collect_roi_rgb(size: tuple[int, int] = (96, 48), seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for night, wet in ((False, False), (True, False), (False, True)):
        sim = RoadSimulator(
            SimConfig(width=320, height=180, fps=10, duration_s=2.0, night=night, wet=wet, blur_windows=[])
        )
        for i in range(0, sim.n_frames(), 1):
            frame, _ = sim.frame_at(i)
            cm = sim.classmap_at(i)
            bgr = frame.bgr
            h, w = bgr.shape[:2]
            for box in (FAR, NEAR):
                x0, y0, x1, y1 = crop_xyxy(w, h, box)
                rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                lab = cm[y0:y1, x0:x1]
                xs.append(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
                ys.append(cv2.resize(lab, size, interpolation=cv2.INTER_NEAREST).astype(np.int32))
    return np.stack(xs), np.stack(ys)


def try_export_classmap_tflite(out_path: Path, seed: int = 3) -> dict[str, Any]:
    """Train a tiny conv classmap and export FP32 + dynamic-range INT8 TFLite (not loaded in pytest)."""
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "reason": f"tensorflow_unavailable:{exc.__class__.__name__}"}
    x, y = collect_roi_rgb(seed=seed)
    hh, ww = int(x.shape[1]), int(x.shape[2])
    inp = tf.keras.Input(shape=(hh, ww, 3), name="image")
    z = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    z = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(z)
    logits = tf.keras.layers.Conv2D(4, 1, padding="same", name="logits")(z)
    keras_model = tf.keras.Model(inp, logits, name="dual_scale_roadseg")
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.02),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    hist = keras_model.fit(x, y, epochs=12, batch_size=16, verbose=0)
    acc = float(hist.history["accuracy"][-1])
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    fp32 = converter.convert()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(fp32))
    int8_info: dict[str, Any] = {"ok": False}
    try:
        c2 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        c2.optimizations = [tf.lite.Optimize.DEFAULT]

        def _rep():
            for i in range(min(16, len(x))):
                yield [x[i : i + 1]]

        c2.representative_dataset = _rep
        qbytes = c2.convert()
        qpath = out_path.with_name("model_int8.tflite")
        qpath.write_bytes(bytes(qbytes))
        int8_info = {"ok": True, "path": str(qpath), "bytes": len(qbytes)}
    except Exception as exc:
        int8_info = {"ok": False, "reason": str(exc)[:240]}
    return {
        "ok": True,
        "path": str(out_path),
        "bytes": len(fp32),
        "train_acc": acc,
        "input_hw": [hh, ww],
        "n_crops": int(len(x)),
        "int8": int8_info,
        "note": "Synthetic classmaps only; not a PKC110-trained segmentation net.",
    }


def write_classmap_package(out_dir: Path, model: dict[str, Any], tflite_info: dict[str, Any]) -> Path:
    import hashlib

    from rpar.ml.train import write_model_package

    out_dir = Path(out_dir)
    write_model_package(out_dir, out_dir.name, engine="classmap")
    (out_dir / "seg_weights.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    src = tflite_info.get("path")
    if tflite_info.get("ok") and src and Path(src).exists():
        (out_dir / "model.tflite").write_bytes(Path(src).read_bytes())
        qsrc = (tflite_info.get("int8") or {}).get("path")
        if qsrc and Path(qsrc).exists():
            (out_dir / "model_int8.tflite").write_bytes(Path(qsrc).read_bytes())
    (out_dir / "MODEL_CARD.md").write_text(
        f"# {out_dir.name}\n\n"
        "Dual-scale classmap sidecar trained on **synthetic** simulator GT (ML-005). "
        "Not a production road-segmentation net and not evaluated on PKC110 Camera2 1080p60.\n"
        "Android HybridEngine keeps heuristic instances; classmap supplies road/occlusion "
        "and extra unknown_anomaly masks.\n",
        encoding="utf-8",
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["engine"] = "classmap"
    manifest["quantization"] = "fp32_plus_dynamic_int8"
    manifest["sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out_dir.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def write_seg_bundle(out_dir: Path, seed: int = 3, export_tflite: bool = False, package_dir: Path | None = None) -> dict[str, Any]:
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
    slices = eval_classmap_slices(np.asarray(model["weights"]))
    (out_dir / "scene_iou.json").write_text(json.dumps(slices, indent=2), encoding="utf-8")
    prec = classmap_precision_cards(np.asarray(model["weights"]))
    (out_dir / "precision_cards.json").write_text(json.dumps(prec, indent=2), encoding="utf-8")
    tflite_info: dict[str, Any] = {"ok": False, "reason": "skipped"}
    if export_tflite:
        tflite_info = try_export_classmap_tflite(out_dir / "model.tflite", seed=seed)
    (out_dir / "tflite_export.json").write_text(json.dumps(tflite_info, indent=2, default=str), encoding="utf-8")
    bundle = {**model, "slices": slices, "precision": prec, "tflite": tflite_info}
    dest = Path(package_dir) if package_dir else None
    if dest is not None:
        write_classmap_package(dest, model, tflite_info)
        bundle["package"] = str(dest)
    return bundle
