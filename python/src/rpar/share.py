"""Share-pack export: face/plate blur + location coarsening (REC-007, SEC-005)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar.privacy import coarsen_location, sanitize_event, write_sanitized_jsonl
from rpar.session import write_checksums


def _haar(name: str):
    root = getattr(cv2, "data", None)
    cascades = getattr(root, "haarcascades", None) if root is not None else None
    if not cascades:
        return None
    path = Path(cascades) / name
    if not path.exists():
        return None
    clf = cv2.CascadeClassifier(str(path))
    return clf if not clf.empty() else None


def redact_bgr(bgr: np.ndarray) -> np.ndarray:
    """Blur likely faces and license-plate-shaped regions. Conservative: extra blur beats a leak."""
    out = bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    boxes: list[tuple[int, int, int, int]] = []
    face = _haar("haarcascade_frontalface_default.xml")
    if face is not None:
        for x, y, ww, hh in face.detectMultiScale(gray, 1.1, 4, minSize=(24, 24)):
            boxes.append((int(x), int(y), int(x + ww), int(y + hh)))
    # plate-like: high-contrast wide rectangles in the lower 70% of the frame
    edges = cv2.Canny(gray, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh < 8 or ww < 24:
            continue
        ar = ww / max(hh, 1)
        if 2.0 <= ar <= 6.5 and y > h * 0.28 and ww < w * 0.45:
            boxes.append((x, y, x + ww, y + hh))
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, x0 - 4), max(0, y0 - 4)
        x1, y1 = min(w, x1 + 4), min(h, y1 + 4)
        roi = out[y0:y1, x0:x1]
        if roi.size:
            k = max(9, (min(roi.shape[0], roi.shape[1]) // 2) | 1)
            out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def _redact_video(src: Path, dest: Path) -> bool:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dest.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ok_any = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(redact_bgr(frame))
        ok_any = True
    cap.release()
    vw.release()
    return ok_any


def export_share_bundle(src: Path, dest: Path) -> dict[str, Any]:
    """Write a SHARE_REDACTED copy that is distinct from the LOCAL_ONLY original."""
    src = Path(src)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("checksums.sha256"))
    loc = dest / "location" / "location.jsonl"
    if loc.exists():
        write_sanitized_jsonl(loc, loc, coarsen_location)
    for name in ("diagnostics/events.jsonl", "diagnostics/runtime.jsonl", "events/alerts.jsonl"):
        p = dest / name
        if p.exists():
            write_sanitized_jsonl(p, p, sanitize_event)
    video_dir = dest / "video"
    if video_dir.exists():
        for mp4 in list(video_dir.glob("*.mp4")):
            tmp = mp4.with_suffix(".redact.mp4")
            if _redact_video(mp4, tmp) and tmp.exists() and tmp.stat().st_size > 64:
                mp4.unlink()
                tmp.replace(mp4)
            elif tmp.exists():
                tmp.unlink()
    man_path = dest / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    manifest["privacy_mode"] = "SHARE_REDACTED"
    manifest["notes"] = (manifest.get("notes") or "") + " | share pack: faces/plates blurred, GPS coarsened"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_checksums(dest)
    return {"ok": True, "path": str(dest), "privacy_mode": "SHARE_REDACTED"}
