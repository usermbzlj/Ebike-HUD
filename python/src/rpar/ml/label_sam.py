"""SAM2 click/box refine + short-window tracking. Lazy-loads ultralytics. Pytest must not download weights."""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from rpar.ml.world_bump import repo_root

SAM_NAMES = ("sam2.1_t.pt", "sam2_t.pt", "sam2.1_b.pt")


def resolve_sam_weights(path: Path | None = None) -> Path | None:
    cands: list[Path] = []
    if path is not None:
        cands.append(Path(path))
    env = os.environ.get("RPAR_SAM_WEIGHTS")
    if env:
        cands.append(Path(env))
    root = repo_root()
    for name in SAM_NAMES:
        cands.append(root / "models" / "sam" / name)
        cands.append(root / name)
    try:
        from ultralytics.utils import SETTINGS

        wdir = Path(str(SETTINGS.get("weights_dir") or ""))
        if wdir:
            for name in SAM_NAMES:
                cands.append(wdir / name)
    except Exception:
        pass
    for cand in cands:
        if cand.is_file() and cand.stat().st_size > 1_000_000:
            return cand
    return None


def mask_to_xyxy(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    if mask is None or mask.size == 0:
        return None
    binary = mask > 0.5 if mask.dtype != np.bool_ else mask
    if binary.ndim == 3:
        binary = binary.squeeze()
    ys, xs = np.where(binary)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_ok(prev: tuple[float, float, float, float] | None, box: tuple[float, float, float, float], width: int, height: int) -> bool:
    area = box_area(box)
    if area < 64.0:
        return False
    if area > 0.28 * float(width * height):
        return False
    if box[2] <= box[0] + 4 or box[3] <= box[1] + 4:
        return False
    if prev is None:
        return True
    pa = box_area(prev)
    if pa > 0 and (area / pa > 4.5 or pa / area > 4.5):
        return False
    pcx, pcy = 0.5 * (prev[0] + prev[2]), 0.5 * (prev[1] + prev[3])
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    diag = float(np.hypot(width, height))
    if np.hypot(cx - pcx, cy - pcy) > 0.28 * diag:
        return False
    return True


def _result_box(result, width: int, height: int) -> tuple[float, float, float, float] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes):
        best = None
        best_area = -1.0
        xyxy = boxes.xyxy
        for i in range(int(len(xyxy))):
            row = xyxy[i]
            box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            a = box_area(box)
            if a > best_area:
                best, best_area = box, a
        return best
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is None:
        return None
    arr = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
    best = None
    best_area = -1.0
    for m in arr:
        box = mask_to_xyxy(m)
        if box is None:
            continue
        # Masks may be model-sized; scale into the source frame.
        mh, mw = m.shape[-2:]
        if mw != width or mh != height:
            sx, sy = width / float(mw), height / float(mh)
            box = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
        a = box_area(box)
        if a > best_area:
            best, best_area = box, a
    return best


class SamAssist:
    """Click/box → tight box on one frame; walk the same object a short window forward/back."""

    def __init__(self, weights: Path | None = None) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise RuntimeError("sam_disabled_in_pytest")
        if find_spec("ultralytics") is None:
            raise RuntimeError("ultralytics_missing")
        from ultralytics import SAM

        resolved = resolve_sam_weights(weights)
        name = str(resolved) if resolved is not None else "sam2.1_t.pt"
        self.model = SAM(name)
        dest = repo_root() / "models" / "sam" / Path(name).name
        src = Path(getattr(self.model, "ckpt_path", "") or name)
        if src.is_file() and src.stat().st_size > 1_000_000:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.resolve() != src.resolve():
                try:
                    dest.write_bytes(src.read_bytes())
                except Exception:
                    pass
        self.weights = dest if dest.is_file() else Path(name)
        self.device = 0
        try:
            import torch

            if not torch.cuda.is_available():
                self.device = "cpu"
        except Exception:
            self.device = "cpu"

    def prompt(
        self,
        bgr: np.ndarray,
        *,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        bbox: list[float] | tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float] | None:
        h, w = bgr.shape[:2]
        kw: dict[str, Any] = {"verbose": False, "device": self.device, "imgsz": 1024}
        if points:
            kw["points"] = [list(map(float, p)) for p in points]
            kw["labels"] = [int(x) for x in (labels or [1] * len(points))]
        if bbox is not None:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            kw["bboxes"] = [[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]]
        if "points" not in kw and "bboxes" not in kw:
            return None
        results = self.model.predict(bgr, **kw)
        if not results:
            return None
        box = _result_box(results[0], w, h)
        if box is None:
            return tuple(kw["bboxes"][0]) if "bboxes" in kw else None
        x0, y0, x1, y1 = box
        return (
            float(np.clip(x0, 0, w - 1)),
            float(np.clip(y0, 0, h - 1)),
            float(np.clip(x1, 1, w)),
            float(np.clip(y1, 1, h)),
        )

    def propagate(
        self,
        frames: list[np.ndarray],
        seed_idx: int,
        *,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        bbox: list[float] | tuple[float, float, float, float] | None = None,
    ) -> dict[int, tuple[float, float, float, float]]:
        if not frames:
            return {}
        seed = self.prompt(frames[seed_idx], points=points, labels=labels, bbox=bbox)
        if seed is None and bbox is not None:
            seed = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        if seed is None:
            return {}
        h, w = frames[seed_idx].shape[:2]
        out: dict[int, tuple[float, float, float, float]] = {seed_idx: seed}
        prev = seed
        for i in range(seed_idx + 1, len(frames)):
            nxt = self.prompt(frames[i], bbox=prev)
            if nxt is None or not box_ok(prev, nxt, w, h):
                break
            out[i] = nxt
            prev = nxt
        prev = seed
        for i in range(seed_idx - 1, -1, -1):
            nxt = self.prompt(frames[i], bbox=prev)
            if nxt is None or not box_ok(prev, nxt, w, h):
                break
            out[i] = nxt
            prev = nxt
        return out


_ASSIST: SamAssist | None = None
_ASSIST_ERROR: str | None = None


def sam_status() -> dict[str, Any]:
    loaded = _ASSIST is not None
    return {
        "ok": loaded and _ASSIST_ERROR is None,
        "loaded": loaded,
        "error": _ASSIST_ERROR,
        "weights": str(_ASSIST.weights) if _ASSIST is not None else None,
        "device": getattr(_ASSIST, "device", None) if _ASSIST is not None else None,
    }


def get_sam() -> SamAssist:
    global _ASSIST, _ASSIST_ERROR
    if _ASSIST is not None:
        return _ASSIST
    try:
        _ASSIST = SamAssist()
        _ASSIST_ERROR = None
    except Exception as exc:
        _ASSIST_ERROR = str(exc)
        raise
    return _ASSIST
