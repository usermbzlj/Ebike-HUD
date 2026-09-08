"""FastAPI research console: replay, live sim, calibration, capability, models."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rpar.capability import desktop_capability_stub, write_capability_report
from rpar.config import load_config
from rpar.enums import UiMode
from rpar.geometry import GeometryEngine
from rpar.golden import run_simulator_golden
from rpar.ml.train import write_model_package
from rpar.overlay import compose
from rpar.perception import HeuristicPerceptionEngine
from rpar.pipeline import RealtimePipeline
from rpar.simulator import RoadSimulator, SimConfig
from rpar.transforms import default_mount

ROOT = Path(__file__).resolve().parents[4]
STATIC = Path(__file__).resolve().parent / "static"
ART = ROOT / "artifacts"
MODELS = ROOT / "models"


class DemoState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        cfg_path = ROOT / "configs" / "rpar.defaults.yaml"
        self.cfg = load_config(cfg_path if cfg_path.exists() else None)
        self.ui = UiMode.RESEARCH
        self.alerts_enabled = True
        self.night = False
        self.sim: RoadSimulator | None = None
        self.pipe: RealtimePipeline | None = None
        self.index = 0
        self.last_jpeg: bytes | None = None
        self.last_view: dict[str, Any] | None = None
        self.timeline: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        sim_cfg = SimConfig(
            duration_s=5.0,
            night=self.night,
            blur_windows=[(1.6, 2.05)],
            glare_windows=[(3.2, 3.5)],
        )
        self.sim = RoadSimulator(sim_cfg)
        self.pipe = RealtimePipeline(
            self.cfg,
            HeuristicPerceptionEngine(self.cfg),
            GeometryEngine(self.sim.mount, self.cfg.geometry, self.sim.k),
        )
        self.pipe.set_alerts_enabled(self.alerts_enabled)
        self.index = 0
        self.timeline = []
        self.last_jpeg = None
        self.last_view = None

    def step_one(self) -> dict[str, Any]:
        assert self.sim and self.pipe
        if self.index >= self.sim.n_frames():
            self.index = 0
            self.pipe.reset()
        frame, gt = self.sim.frame_at(self.index)
        view = self.pipe.step(frame, ui_mode=self.ui)
        vis = compose(frame.bgr, view, self.ui)
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        self.last_jpeg = buf.tobytes()
        payload = {
            "index": self.index,
            "t": gt["t"],
            "gt": gt,
            "status": view.status.value,
            "status_copy": view.status_copy,
            "speed_kmh": view.speed_kmh,
            "infer_fps": view.infer_fps,
            "ar_fps": view.ar_fps,
            "latency_p95_ms": view.latency_p95_ms,
            "backend": view.backend.value,
            "model_version": view.model_version,
            "blur": view.blur,
            "glare": view.glare,
            "alerts": [a.to_dict() for a in view.alerts],
            "tracks": [t.to_dict() for t in view.tracks],
            "n_frames": self.sim.n_frames(),
            "ui": self.ui.value,
            "alerts_enabled": self.alerts_enabled,
            "night": self.night,
            "geometry_valid": self.pipe.geometry.valid,
        }
        self.last_view = payload
        self.timeline.append(payload)
        self.index += 1
        return payload


STATE = DemoState()


def create_app() -> FastAPI:
    app = FastAPI(title="Road Perception AR Console", version="0.1.0")
    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = STATIC / "index.html"
        if not html.exists():
            return HTMLResponse("<h1>missing static/index.html</h1>", 500)
        return HTMLResponse(html.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "rpar-console"}

    @app.post("/api/reset")
    def reset(night: bool = False) -> dict[str, Any]:
        with STATE.lock:
            STATE.night = night
            STATE.reset()
        return {"ok": True, "n_frames": STATE.sim.n_frames() if STATE.sim else 0}

    @app.post("/api/mode")
    def mode(ui: str = "RESEARCH", alerts: bool = True) -> dict[str, Any]:
        with STATE.lock:
            STATE.ui = UiMode.RIDING if ui.upper() == "RIDING" else UiMode.RESEARCH
            STATE.alerts_enabled = bool(alerts)
            if STATE.pipe:
                STATE.pipe.set_alerts_enabled(STATE.alerts_enabled)
        return {"ui": STATE.ui.value, "alerts": STATE.alerts_enabled}

    @app.get("/api/step")
    def step() -> JSONResponse:
        with STATE.lock:
            payload = STATE.step_one()
        return JSONResponse(payload)

    @app.get("/api/frame.jpg")
    def frame_jpg() -> Response:
        with STATE.lock:
            if STATE.last_jpeg is None:
                STATE.step_one()
            data = STATE.last_jpeg or b""
        return Response(data, media_type="image/jpeg")

    @app.get("/api/state")
    def state() -> JSONResponse:
        with STATE.lock:
            if STATE.last_view is None:
                STATE.step_one()
            return JSONResponse(STATE.last_view or {})

    @app.get("/api/capability")
    def capability() -> dict[str, Any]:
        return desktop_capability_stub()

    @app.post("/api/golden")
    def golden() -> dict[str, Any]:
        ART.mkdir(parents=True, exist_ok=True)
        return run_simulator_golden(ART / "golden", STATE.cfg)

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return STATE.cfg.snapshot()

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        MODELS.mkdir(parents=True, exist_ok=True)
        pkgs = []
        for p in MODELS.iterdir():
            man = p / "manifest.json"
            if man.exists():
                pkgs.append(json.loads(man.read_text(encoding="utf-8")))
        return {"packages": pkgs}

    @app.post("/api/models/ensure-default")
    def ensure_model() -> dict[str, Any]:
        path = write_model_package(MODELS / "heuristic-cv-0.1.0", "heuristic-cv-0.1.0")
        return {"ok": True, "path": str(path)}

    @app.get("/api/mount")
    def mount() -> dict[str, Any]:
        return default_mount().to_dict()

    return app


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    ART.mkdir(parents=True, exist_ok=True)
    write_capability_report(ART / "capability_desktop.json")
    write_model_package(MODELS / "heuristic-cv-0.1.0", "heuristic-cv-0.1.0")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
