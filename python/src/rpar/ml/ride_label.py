"""Local VLM teachers for ride-clip boxes. Florence-2 grounds phrases; Qwen2.5-VL verifies crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar.ml.bump_prompts import map_det_name, nms_xyxy

FLORENCE_IDS = (
    "florence-community/Florence-2-base-ft",
    "florence-community/Florence-2-base",
    "microsoft/Florence-2-base-ft",
)
DINO_ID = "IDEA-Research/grounding-dino-tiny"
QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PHRASES = (
    "pothole",
    "large pothole",
    "asphalt hole",
    "speed bump",
    "speed hump",
    "manhole cover",
    "sunken manhole cover",
    "sewer cover",
)

LAST_LOAD_ERROR: dict[str, str] = {}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "configs" / "rpar.defaults.yaml").is_file():
            return p
    return Path.cwd()


def cache_dir() -> Path:
    d = _repo_root() / "models" / "vlm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gpu_gc() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _to_device(inputs: dict, device, dtype=None) -> dict:
    out = {}
    for k, v in inputs.items():
        if not hasattr(v, "to"):
            out[k] = v
            continue
        if dtype is not None and getattr(v, "is_floating_point", lambda: False)():
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v.to(device)
    return out


def _record_error(name: str, exc: BaseException) -> None:
    LAST_LOAD_ERROR[name] = f"{type(exc).__name__}: {exc}"[:800]
    print(f"[ride-label] {name} load failed: {LAST_LOAD_ERROR[name]}", flush=True)


class FlorenceGrounder:
    def __init__(self, model_id: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        last: Exception | None = None
        for cand in ((model_id,) if model_id else FLORENCE_IDS):
            if not cand:
                continue
            try:
                processor = AutoProcessor.from_pretrained(cand, cache_dir=str(cache_dir()))
                try:
                    from transformers import Florence2ForConditionalGeneration

                    model = Florence2ForConditionalGeneration.from_pretrained(
                        cand,
                        dtype=dtype,
                        cache_dir=str(cache_dir()),
                    )
                except Exception:
                    from transformers import AutoModelForCausalLM

                    model = AutoModelForCausalLM.from_pretrained(
                        cand,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        cache_dir=str(cache_dir()),
                    )
                self.processor = processor
                self.model = model.to(self.device)
                self.model.eval()
                self.model_id = cand
                self.dtype = dtype
                print(f"[ride-label] florence ready: {cand}", flush=True)
                return
            except Exception as exc:
                last = exc
                continue
        raise RuntimeError(last or "florence unavailable")

    def close(self) -> None:
        self.model = None
        self.processor = None
        _gpu_gc()

    def detect_bgr(self, bgr: np.ndarray) -> list[dict]:
        from PIL import Image

        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        h, w = bgr.shape[:2]
        phrases = ", ".join(PHRASES)
        prompt = "<CAPTION_TO_PHRASE_GROUNDING>" + phrases
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = _to_device(inputs, self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text,
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            image_size=(image.width, image.height),
        )
        block = parsed.get("<CAPTION_TO_PHRASE_GROUNDING>") or parsed.get(prompt) or {}
        if isinstance(block, dict):
            boxes = block.get("bboxes") or block.get("boxes") or []
            labels = block.get("labels") or []
        else:
            boxes, labels = [], []
        dets: list[dict] = []
        for box, lab in zip(boxes, labels):
            kind = map_det_name(str(lab))
            if kind is None:
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            dets.append({"name": str(lab), "class": kind, "bbox": (x0, y0, x1, y1), "conf": 0.35, "src": "florence"})
        return nms_xyxy(dets, iou_thr=0.50)


class GroundingDinoGrounder:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(DINO_ID, cache_dir=str(cache_dir()))
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID, cache_dir=str(cache_dir())).to(self.device)
        self.model.eval()
        self.text = "pothole. speed bump. manhole cover. sunken manhole cover."
        print(f"[ride-label] grounding-dino ready: {DINO_ID}", flush=True)

    def close(self) -> None:
        self.model = None
        self.processor = None
        _gpu_gc()

    def detect_bgr(self, bgr: np.ndarray) -> list[dict]:
        from PIL import Image

        import torch

        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=self.text, return_tensors="pt")
        inputs = _to_device(inputs, self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=0.22,
                text_threshold=0.20,
                target_sizes=[(image.height, image.width)],
            )
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=0.22,
                text_threshold=0.20,
                target_sizes=[(image.height, image.width)],
            )
        dets: list[dict] = []
        block = results[0] if results else {}
        boxes = block.get("boxes") or []
        labels = block.get("labels") or []
        scores = block.get("scores") or [0.3] * len(boxes)
        for box, lab, score in zip(boxes, labels, scores):
            kind = map_det_name(str(lab))
            if kind is None:
                continue
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            dets.append(
                {
                    "name": str(lab),
                    "class": kind,
                    "bbox": (x0, y0, x1, y1),
                    "conf": float(score),
                    "src": "grounding-dino",
                }
            )
        return nms_xyxy(dets, iou_thr=0.50)


class QwenVerifier:
    def __init__(self) -> None:
        import torch
        from transformers import AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(QWEN_ID, cache_dir=str(cache_dir()))
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QWEN_ID,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                cache_dir=str(cache_dir()),
            )
        except Exception:
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                QWEN_ID,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                cache_dir=str(cache_dir()),
            )
        if self.device != "cuda" and getattr(self.model, "device", None) is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[ride-label] qwen ready: {QWEN_ID}", flush=True)

    def close(self) -> None:
        self.model = None
        self.processor = None
        _gpu_gc()

    def accept_crop(self, bgr: np.ndarray) -> tuple[bool, str]:
        from PIL import Image

        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        prompt = (
            "这是电动车车把视角的路面局部。判断是否为：大坑pothole、减速带speed_bump、井盖manhole_cover。"
            "斑马线、水渍、修补缝、阴影、普通沥青请选 none。"
            '只输出JSON，例如 {"cls":"pothole"} 或 {"cls":"none"}。'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        device = getattr(self.model, "device", self.device)
        inputs = _to_device(inputs, device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=48, do_sample=False)
        raw = self.processor.batch_decode(out, skip_special_tokens=True)[0].lower()
        for kind in ("pothole", "speed_bump", "manhole_cover", "none"):
            if kind.replace("_", " ") in raw or kind in raw:
                return kind != "none", kind
        if "坑" in raw:
            return True, "pothole"
        if "减速" in raw:
            return True, "speed_bump"
        if "井盖" in raw:
            return True, "manhole_cover"
        return False, "none"


def try_load_florence() -> FlorenceGrounder | None:
    try:
        return FlorenceGrounder()
    except Exception as exc:
        _record_error("florence", exc)
        return None


def try_load_grounding_dino() -> GroundingDinoGrounder | None:
    try:
        return GroundingDinoGrounder()
    except Exception as exc:
        _record_error("grounding_dino", exc)
        return None


def try_load_vlm_boxer():
    """Florence first (phrase grounding); Grounding DINO if transformers native Florence fails."""
    boxer = try_load_florence()
    if boxer is not None:
        return boxer, "florence"
    boxer = try_load_grounding_dino()
    if boxer is not None:
        return boxer, "grounding-dino"
    return None, ""


def try_load_qwen() -> QwenVerifier | None:
    try:
        return QwenVerifier()
    except Exception as exc:
        _record_error("qwen", exc)
        return None


def crop_box(bgr: np.ndarray, bbox: tuple[float, float, float, float], pad: float = 0.25) -> np.ndarray:
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    x0 = int(max(0, x0 - pad * bw))
    y0 = int(max(0, y0 - pad * bh))
    x1 = int(min(w, x1 + pad * bw))
    y1 = int(min(h, y1 + pad * bh))
    if x1 <= x0 or y1 <= y0:
        return bgr
    return bgr[y0:y1, x0:x1]


def merge_dets(*groups: list[dict]) -> list[dict]:
    all_dets: list[dict] = []
    for g in groups:
        all_dets.extend(g or [])
    return nms_xyxy(all_dets, iou_thr=0.55)


def verify_dets(bgr: np.ndarray, dets: list[dict], verifier: QwenVerifier | None) -> list[dict]:
    if verifier is None:
        return dets
    kept: list[dict] = []
    for d in dets:
        crop = crop_box(bgr, tuple(d["bbox"]))
        ok, kind = verifier.accept_crop(crop)
        if not ok:
            continue
        mapped = map_det_name(kind) or d.get("class")
        if mapped:
            d = dict(d)
            d["class"] = mapped
            d["name"] = mapped
        kept.append(d)
    return kept


def label_status() -> dict[str, Any]:
    from importlib.util import find_spec

    return {
        "transformers": find_spec("transformers") is not None,
        "florence": list(FLORENCE_IDS),
        "qwen": QWEN_ID,
        "grounding_dino": DINO_ID,
        "cache": str(cache_dir()),
        "load_errors": dict(LAST_LOAD_ERROR),
    }
