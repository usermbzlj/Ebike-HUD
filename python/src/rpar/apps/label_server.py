"""Interactive SAM2 labeling server. Writes YOLO txt the bump trainer already consumes."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rpar.ml.bump_dataset import default_dataset_dir, default_inbox, list_train_clips, reuse_raw_labels
from rpar.ml.label_store import clear_frame, commit_frame, frame_objects, labeled_frame_indices

STATIC = Path(__file__).resolve().parent / "static"


class PromptBody(BaseModel):
    idx: int
    class_name: str = "pothole"
    points: list[list[float]] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    bbox: list[float] | None = None


class CommitBody(BaseModel):
    idx: int
    class_name: str = "pothole"
    bbox: list[float]
    replace: bool = False


class TrackBody(BaseModel):
    idx: int
    class_name: str = "pothole"
    bbox: list[float]
    points: list[list[float]] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    back_s: float = 1.2
    fwd_s: float = 2.4
    track_fps: float = 8.0


class VideoCursor:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(self.path)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.lock = threading.Lock()
        self._pos = -1

    @property
    def clip(self) -> str:
        return self.path.stem

    def close(self) -> None:
        self.cap.release()

    def read_index(self, idx: int) -> np.ndarray | None:
        if self.n <= 0:
            return None
        idx = int(np.clip(idx, 0, self.n - 1))
        with self.lock:
            if self._pos == idx - 1:
                ok, bgr = self.cap.read()
                if ok:
                    self._pos = idx
                    return bgr
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, bgr = self.cap.read()
            if ok:
                self._pos = idx
                return bgr
            self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * idx / self.fps)
            ok, bgr = self.cap.read()
            self._pos = idx if ok else -1
            return bgr if ok else None

    def jpeg(self, idx: int, quality: int = 86) -> bytes:
        bgr = self.read_index(idx)
        if bgr is None:
            raise HTTPException(status_code=404, detail="frame")
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise HTTPException(status_code=500, detail="jpeg")
        return buf.tobytes()


class LabelState:
    def __init__(self, video: Path | None, dataset: Path) -> None:
        self.dataset = Path(dataset)
        self.inbox = default_inbox()
        self.lock = threading.Lock()
        self.cursor: VideoCursor | None = None
        if video is not None and Path(video).is_file():
            self.open_video(Path(video))
        else:
            clips = list_train_clips(self.inbox)
            if clips:
                self.open_video(clips[0])

    def open_video(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with self.lock:
            if self.cursor is not None:
                self.cursor.close()
            self.cursor = VideoCursor(path)
        return self.meta()

    def need(self) -> VideoCursor:
        if self.cursor is None:
            raise HTTPException(status_code=404, detail="no_video")
        return self.cursor

    def meta(self) -> dict[str, Any]:
        cur = self.need()
        return {
            "path": str(cur.path),
            "name": cur.path.name,
            "clip": cur.clip,
            "fps": cur.fps,
            "n": cur.n,
            "width": cur.w,
            "height": cur.h,
            "duration_s": cur.n / max(cur.fps, 1e-6),
            "dataset": str(self.dataset),
            "video_url": "/api/label/video",
        }

    def clips(self) -> list[dict[str, Any]]:
        rows = []
        for p in list_train_clips(self.inbox):
            rows.append({"path": str(p), "name": p.name, "clip": p.stem, "bytes": p.stat().st_size})
        cur = self.cursor
        if cur is not None and all(r["path"] != str(cur.path) for r in rows):
            rows.insert(0, {"path": str(cur.path), "name": cur.path.name, "clip": cur.clip, "bytes": cur.path.stat().st_size})
        return rows


STATE: LabelState | None = None


def _state() -> LabelState:
    if STATE is None:
        raise HTTPException(status_code=500, detail="not_booted")
    return STATE


def _window_indices(seed: int, n: int, fps: float, back_s: float, fwd_s: float, track_fps: float) -> list[int]:
    start = max(0, int(round(seed - back_s * fps)))
    end = min(n - 1, int(round(seed + fwd_s * fps)))
    step = max(1, int(round(fps / max(track_fps, 0.1))))
    idxs = list(range(start, end + 1, step))
    if seed not in idxs:
        idxs.append(seed)
        idxs.sort()
    return idxs


def create_label_app(video: Path | None = None, dataset: Path | None = None) -> FastAPI:
    global STATE
    STATE = LabelState(video, Path(dataset) if dataset else default_dataset_dir())
    app = FastAPI(title="RPAR Label", version="0.1.0")
    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = STATIC / "label.html"
        if not html.exists():
            return HTMLResponse("<h1>missing static/label.html</h1>", 500)
        return HTMLResponse(html.read_text(encoding="utf-8"))

    @app.get("/api/label/meta")
    def meta() -> dict[str, Any]:
        return _state().meta()

    @app.get("/api/label/clips")
    def clips() -> dict[str, Any]:
        st = _state()
        return {"inbox": str(st.inbox), "clips": st.clips()}

    @app.post("/api/label/open")
    def open_clip(payload: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(payload.get("path") or ""))
        return _state().open_video(path)

    @app.get("/api/label/video")
    def video_file() -> FileResponse:
        cur = _state().need()
        return FileResponse(cur.path, media_type="video/mp4", filename=cur.path.name)

    @app.get("/api/label/frame")
    def frame(idx: int) -> Response:
        return Response(content=_state().need().jpeg(idx), media_type="image/jpeg")

    @app.get("/api/label/objects")
    def objects(idx: int) -> dict[str, Any]:
        st = _state()
        cur = st.need()
        objs = frame_objects(st.dataset, cur.clip, idx, cur.w, cur.h)
        return {"idx": idx, "objects": objs}

    @app.get("/api/label/timeline")
    def timeline() -> dict[str, Any]:
        st = _state()
        cur = st.need()
        return {"idx": labeled_frame_indices(st.dataset, cur.clip)}

    @app.get("/api/label/status")
    def status() -> dict[str, Any]:
        from rpar.ml.label_sam import sam_status

        return sam_status()

    @app.post("/api/label/prompt")
    def prompt(body: PromptBody) -> dict[str, Any]:
        from rpar.ml.label_sam import get_sam

        st = _state()
        cur = st.need()
        bgr = cur.read_index(body.idx)
        if bgr is None:
            raise HTTPException(status_code=404, detail="frame")
        box = get_sam().prompt(bgr, points=body.points or None, labels=body.labels or None, bbox=body.bbox)
        if box is None:
            return {"ok": False, "reason": "no_mask", "idx": body.idx}
        return {"ok": True, "idx": body.idx, "class": body.class_name, "bbox": list(box)}

    @app.post("/api/label/commit")
    def commit(body: CommitBody) -> dict[str, Any]:
        st = _state()
        cur = st.need()
        bgr = cur.read_index(body.idx)
        if bgr is None:
            raise HTTPException(status_code=404, detail="frame")
        return commit_frame(
            st.dataset,
            clip=cur.clip,
            source_name=cur.path.name,
            frame_index=body.idx,
            bgr=bgr,
            objects=[{"class": body.class_name, "bbox": body.bbox, "source": "human"}],
            replace=body.replace,
        )

    @app.post("/api/label/clear")
    def clear(payload: dict[str, Any]) -> dict[str, Any]:
        st = _state()
        cur = st.need()
        idx = int(payload.get("idx") or 0)
        bgr = cur.read_index(idx)
        if bgr is None:
            raise HTTPException(status_code=404, detail="frame")
        return clear_frame(st.dataset, clip=cur.clip, source_name=cur.path.name, frame_index=idx, bgr=bgr)

    @app.post("/api/label/track")
    def track(body: TrackBody) -> dict[str, Any]:
        from rpar.ml.label_sam import get_sam

        st = _state()
        cur = st.need()
        idxs = _window_indices(body.idx, cur.n, cur.fps, body.back_s, body.fwd_s, body.track_fps)
        frames: list[np.ndarray] = []
        kept: list[int] = []
        for i in idxs:
            bgr = cur.read_index(i)
            if bgr is None:
                continue
            frames.append(bgr)
            kept.append(i)
        if not frames:
            raise HTTPException(status_code=404, detail="window")
        try:
            seed_local = kept.index(body.idx)
        except ValueError:
            seed_local = min(range(len(kept)), key=lambda j: abs(kept[j] - body.idx))
        local = get_sam().propagate(
            frames,
            seed_local,
            points=body.points or None,
            labels=body.labels or None,
            bbox=body.bbox,
        )
        written = []
        for local_i, box in local.items():
            src_idx = kept[local_i]
            out = commit_frame(
                st.dataset,
                clip=cur.clip,
                source_name=cur.path.name,
                frame_index=src_idx,
                bgr=frames[local_i],
                objects=[{"class": body.class_name, "bbox": list(box), "source": "human"}],
            )
            written.append({"idx": src_idx, "bbox": list(box), "n": out["n"]})
        return {"ok": True, "n": len(written), "frames": written}

    @app.post("/api/label/materialize")
    def materialize() -> dict[str, Any]:
        return reuse_raw_labels(_state().dataset)

    return app


def main(host: str = "127.0.0.1", port: int = 8766, *, video: Path | None = None, dataset: Path | None = None, open_browser: bool = True) -> None:
    import uvicorn

    app = create_label_app(video=video, dataset=dataset)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
