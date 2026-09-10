"""YOLO-World / fine-tuned YOLO bump sidecar. Lazy-loads ultralytics. Pytest must not import this module's model."""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from rpar.enums import InferenceBackend
from rpar.maskutil import rect_polygon
from rpar.ml.bump_prompts import (
    WORLD_INFER_PROMPTS,
    WORLD_PROMPTS,
    detection_to_obs,
    is_sunken_prompt,
    map_det_name,
    nms_xyxy,
    suppress_bumps_on_vehicles,
)
from rpar.models import FrameQualityMap, PerceptionResult, SynchronizedFrame

TEACHER_NAME = "yolov8m-worldv2.pt"
FINETUNED_NAMES = ("best.pt", "weights/best.pt", "last.pt")
COCO_VEHICLE_CLS = (2, 3, 5, 7)  # car, motorcycle, bus, truck


def repo_root(start: Path | None = None) -> Path:
    here = Path(start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "configs" / "rpar.defaults.yaml").is_file():
            return p
    return Path.cwd()


def default_teacher_path() -> Path:
    return repo_root() / "models" / "yolo-world" / TEACHER_NAME


def default_vocab_path() -> Path:
    return repo_root() / "models" / "yolo-world" / "rpar-bump-vocab.pt"


def default_finetuned_paths() -> list[Path]:
    pkg = repo_root() / "models" / "bump-world-0.1.0"
    return [pkg / name for name in FINETUNED_NAMES] + [
        pkg / "train" / "weights" / "best.pt",
        repo_root() / "artifacts" / "bump_train" / "train" / "weights" / "best.pt",
        default_vocab_path(),
    ]


def resolve_coco_vehicle_weights(path: Path | None = None) -> Path | None:
    """Local COCO detector used only to suppress vehicle false potholes. Never required."""
    cands: list[Path] = []
    if path is not None:
        cands.append(Path(path))
    env = os.environ.get("RPAR_COCO_WEIGHTS")
    if env:
        cands.append(Path(env))
    root = repo_root()
    cands.extend(
        [
            root / "models" / "yolo-coco" / "yolov8n.pt",
            root / "models" / "yolov8n.pt",
        ]
    )
    try:
        from ultralytics.utils import SETTINGS

        wdir = Path(str(SETTINGS.get("weights_dir") or ""))
        if wdir:
            cands.append(wdir / "yolov8n.pt")
    except Exception:
        pass
    for cand in cands:
        if cand.is_file() and cand.stat().st_size > 1_000_000:
            return cand
    return None


def resolve_bump_weights(path: Path | None = None) -> Path | None:
    if path is not None:
        p = Path(path)
        return p if p.is_file() and p.stat().st_size > 1_000_000 else None
    for cand in default_finetuned_paths():
        if cand.is_file() and cand.stat().st_size > 1_000_000:
            return cand
    teacher = default_teacher_path()
    if teacher.is_file() and teacher.stat().st_size > 1_000_000:
        return teacher
    return None


def _is_open_vocab(weights: Path) -> bool:
    n = weights.name.lower()
    return "worldv" in n or n.endswith("-world.pt") or "-world-" in n


def weights_available(path: Path | None = None) -> bool:
    return resolve_bump_weights(path) is not None


def ensure_bump_vocab() -> dict:
    """Freeze YOLO-World prompts into a local detect checkpoint (no CLIP at runtime)."""
    dest = default_vocab_path()
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return {"ok": True, "path": str(dest), "saved": False}
    teacher = default_teacher_path()
    if not teacher.is_file():
        return {"ok": False, "reason": "teacher_missing", "path": str(dest)}
    if find_spec("ultralytics") is None:
        return {"ok": False, "reason": "ultralytics_missing", "path": str(dest)}
    from ultralytics import YOLOWorld

    dest.parent.mkdir(parents=True, exist_ok=True)
    model = YOLOWorld(str(teacher))
    model.set_classes(list(WORLD_INFER_PROMPTS))
    model.save(str(dest))
    return {"ok": dest.is_file() and dest.stat().st_size > 1_000_000, "path": str(dest), "saved": True}


class WorldBumpEngine:
    """Open-vocabulary YOLO-World (or a fine-tuned 3-class YOLO) → RoadObservation. Not YOLOPv2."""

    replaces_bump_instances = True

    def __init__(self, weights: Path | None = None, conf: float = 0.08) -> None:
        if find_spec("ultralytics") is None:
            raise RuntimeError("ultralytics is not installed")
        resolved = resolve_bump_weights(weights)
        if resolved is None:
            raise FileNotFoundError("no local YOLO-World / bump weights")
        self.weights = resolved
        self.conf = float(conf)
        self.open_vocab = _is_open_vocab(resolved)
        if self.open_vocab:
            from ultralytics import YOLOWorld

            self._model = YOLOWorld(str(resolved))
            self._model.set_classes(list(WORLD_INFER_PROMPTS))
            vocab = default_vocab_path()
            if not vocab.is_file():
                try:
                    self._model.save(str(vocab))
                except Exception:
                    pass
        else:
            from ultralytics import YOLO

            self._model = YOLO(str(resolved))
        self.device = _infer_device()
        self.last_dets: list[dict] = []
        self.last_vehicle_boxes: list[tuple[float, float, float, float]] = []
        self._vehicle_model = None
        self._vehicle_load_failed = False

    def _load_vehicle_model(self):
        if self._vehicle_model is not None or self._vehicle_load_failed:
            return self._vehicle_model
        # Pytest never downloads or loads a COCO net.
        if os.environ.get("PYTEST_CURRENT_TEST"):
            self._vehicle_load_failed = True
            return None
        path = resolve_coco_vehicle_weights()
        try:
            from ultralytics import YOLO

            self._vehicle_model = YOLO(str(path) if path is not None else "yolov8n.pt")
        except Exception:
            self._vehicle_load_failed = True
            self._vehicle_model = None
        return self._vehicle_model

    def _detect_vehicles(self, bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        model = self._load_vehicle_model()
        if model is None:
            return []
        try:
            results = model.predict(
                source=bgr,
                conf=0.18,
                iou=0.50,
                verbose=False,
                device=self.device,
                imgsz=640,
                classes=list(COCO_VEHICLE_CLS),
                max_det=50,
            )
        except Exception:
            return []
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        out: list[tuple[float, float, float, float]] = []
        xyxy = boxes.xyxy
        for i in range(int(len(xyxy))):
            row = xyxy[i]
            out.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
        return out

    def detect_bgr(self, bgr: np.ndarray) -> list[dict]:
        results = self._model.predict(
            source=bgr,
            conf=self.conf,
            iou=0.50,
            verbose=False,
            device=self.device,
            imgsz=640,
        )
        self.last_vehicle_boxes = []
        if not results:
            self.last_dets = []
            return []
        r0 = results[0]
        names = getattr(r0, "names", None) or getattr(self._model, "names", {}) or {}
        boxes = getattr(r0, "boxes", None)
        dets: list[dict] = []
        if boxes is None:
            self.last_dets = []
            return []
        xyxy = boxes.xyxy
        cls_ids = boxes.cls
        confs = boxes.conf
        n = int(len(xyxy))
        for i in range(n):
            row = xyxy[i]
            box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            cls_id = int(cls_ids[i])
            if isinstance(names, dict):
                name = names.get(cls_id, names.get(str(cls_id), str(cls_id)))
            else:
                name = names[cls_id]
            name = str(name)
            if not name.strip():
                continue
            kind = map_det_name(name)
            if kind is None:
                continue
            dets.append(
                {
                    "name": name,
                    "class": kind,
                    "bbox": box,
                    "conf": float(confs[i]),
                    "sunken": is_sunken_prompt(name),
                }
            )
        dets = nms_xyxy(dets, iou_thr=0.50)
        if dets:
            self.last_vehicle_boxes = self._detect_vehicles(bgr)
            dets = suppress_bumps_on_vehicles(dets, self.last_vehicle_boxes)
        self.last_dets = dets
        return dets

    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult:
        t0 = perf_counter()
        dets = self.detect_bgr(frame.bgr)
        obs = []
        for d in dets:
            item = detection_to_obs(
                frame,
                str(d["name"]),
                tuple(d["bbox"]),
                float(d["conf"]),
                frame.bgr,
                quality,
                force_sunken=bool(d.get("sunken")),
            )
            if item is not None:
                obs.append(item)
        backend = InferenceBackend.GPU if self.device != "cpu" else InferenceBackend.CPU
        occ = [rect_polygon(*box) for box in self.last_vehicle_boxes]
        return PerceptionResult(
            timestamp_ns=frame.meta.sensor_timestamp_ns,
            source_frame_id=frame.meta.frame_id,
            road_polygon=[],
            occluded_polygons=occ,
            observations=obs,
            backend=backend,
            latency_ms=(perf_counter() - t0) * 1000.0,
            input_sizes=[(640, 640)],
            dual_scale=False,
        )

    def capability(self) -> dict[str, Any]:
        return {
            "backend": "yolo-world" if self.open_vocab else "yolo-bump",
            "outputs": "RoadObservation",
            "tensors_to_ui": False,
            "tasks": ["pothole", "manhole_cover", "speed_bump"],
            "not": ["drivable_area"],
            "open_vocab": self.open_vocab,
            "prompts": list(WORLD_PROMPTS) if self.open_vocab else list(("pothole", "speed_bump", "manhole_cover")),
            "weights": str(self.weights),
            "device": str(self.device),
        }

    def close(self) -> None:
        self._model = None
        self._vehicle_model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _infer_device() -> int | str:
    if find_spec("torch") is None:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        return "cpu"
    return "cpu"


def try_load_world_bump(weights: Path | None = None, conf: float = 0.08) -> WorldBumpEngine | None:
    path = resolve_bump_weights(weights)
    if path is None:
        return None
    if find_spec("ultralytics") is None:
        return None
    try:
        return WorldBumpEngine(path, conf=conf)
    except Exception:
        return None
