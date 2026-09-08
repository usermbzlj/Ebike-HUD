"""FastAPI research console: replay, live sim, calibration, capability, models."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rpar.annotation import tracks_to_annotation_task, tracks_to_cvat_xml, apply_revision
from rpar.capability import desktop_capability_stub, write_capability_report
from rpar.config import load_config
from rpar.enums import UiMode
from rpar.geometry import GeometryEngine
from rpar.capture import record_simulated_session
from rpar.error_cases import harvest_error_cases
from rpar.golden import run_acceptance_suite, run_oracle_golden, run_simulator_golden
from rpar.ml.active import write_active_queue
from rpar.ml.eval import write_eval_bundle
from rpar.ml.synth_train import write_training_bundle
from rpar.ml.train import write_model_package
from rpar.overlay import compose
from rpar.perception import HeuristicPerceptionEngine
from rpar.pipeline import RealtimePipeline
from rpar.replay import SessionReplay, scan_time_offset_ms
from rpar.session import verify_session
from rpar.share import export_share_bundle
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
        self.replay: SessionReplay | None = None
        self.replay_index = 0
        self.last_replay_jpeg: bytes | None = None
        self.paused = False
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
        if self.paused and self.last_view is not None:
            payload = dict(self.last_view)
            payload["paused"] = True
            return payload
        if self.index >= self.sim.n_frames():
            self.index = 0
            self.pipe.reset()
        frame, gt = self.sim.frame_at(self.index)
        view = self.pipe.step(frame, ui_mode=self.ui)
        vis = compose(frame.bgr, view, self.ui, night=self.night)
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
            "latency_p50_ms": view.latency_p50_ms,
            "queue_depth": view.queue_depth,
            "dropped_infer": view.dropped_infer,
            "dual_scale": view.dual_scale,
            "input_far": list(view.input_far),
            "input_near": list(view.input_near),
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
            "paused": self.paused,
            "degrade_reason": view.quality.degrade_reason if view.quality else None,
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

    @app.post("/api/pause")
    def pause(on: bool | None = None) -> dict[str, Any]:
        with STATE.lock:
            STATE.paused = (not STATE.paused) if on is None else bool(on)
        return {"paused": STATE.paused}

    @app.post("/api/screenshot")
    def screenshot() -> dict[str, Any]:
        ART.mkdir(parents=True, exist_ok=True)
        dest = ART / "screenshots"
        dest.mkdir(parents=True, exist_ok=True)
        with STATE.lock:
            data = STATE.last_jpeg
        if not data:
            raise HTTPException(status_code=404, detail="no_frame")
        path = dest / f"frame_{STATE.index:05d}.jpg"
        path.write_bytes(data)
        return {"ok": True, "path": str(path)}

    @app.post("/api/mark")
    def mark(note: str = "manual") -> dict[str, Any]:
        ART.mkdir(parents=True, exist_ok=True)
        with STATE.lock:
            ts = STATE.last_view.get("t") if STATE.last_view else None
            frame_id = STATE.index
            if STATE.replay is not None:
                marks = STATE.replay.root / "events" / "marks.jsonl"
                marks.parent.mkdir(parents=True, exist_ok=True)
                marks.open("a", encoding="utf-8").write(
                    json.dumps({"timestamp_ns": None, "t": ts, "frame_id": frame_id, "note": note}) + "\n"
                )
                path = marks
            else:
                path = ART / "marks.jsonl"
                path.open("a", encoding="utf-8").write(
                    json.dumps({"t": ts, "frame_id": frame_id, "note": note}) + "\n"
                )
        return {"ok": True, "path": str(path), "note": note}

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

    @app.post("/api/accept")
    def accept() -> dict[str, Any]:
        ART.mkdir(parents=True, exist_ok=True)
        oracle = run_oracle_golden(ART / "acceptance" / "oracle", STATE.cfg)
        scenes = run_acceptance_suite(ART / "acceptance" / "scenes", STATE.cfg)
        return {"oracle": oracle, "scenes": scenes}

    @app.post("/api/session/record")
    def session_record() -> dict[str, Any]:
        ART.mkdir(parents=True, exist_ok=True)
        path = record_simulated_session(ART / "sessions")
        with STATE.lock:
            if STATE.replay:
                STATE.replay.close()
            STATE.replay = SessionReplay(path)
            STATE.replay.open()
            STATE.replay_index = 0
        return {"ok": True, "path": str(path), "verify": verify_session(path), "n_frames": STATE.replay.n_frames()}

    @app.get("/api/replay/state")
    def replay_state(i: int = 0) -> JSONResponse:
        with STATE.lock:
            if STATE.replay is None:
                raise HTTPException(status_code=404, detail="no session recorded")
            payload = STATE.replay.at(i)
            STATE.last_replay_jpeg = payload.pop("jpeg", b"")
            STATE.replay_index = i
        payload.pop("jpeg", None)
        return JSONResponse(payload)

    @app.get("/api/replay/frame.jpg")
    def replay_jpg() -> Response:
        with STATE.lock:
            data = STATE.last_replay_jpeg or b""
        return Response(data, media_type="image/jpeg")

    @app.get("/api/replay/offset")
    def replay_offset() -> dict[str, Any]:
        with STATE.lock:
            if STATE.replay is None:
                return {"ok": False, "reason": "no_session"}
            return scan_time_offset_ms(STATE.replay.root)

    @app.get("/api/replay/events")
    def replay_events() -> dict[str, Any]:
        with STATE.lock:
            if STATE.replay is None:
                return {"events": []}
            return {"events": STATE.replay.event_index}

    @app.post("/api/replay/clip")
    def replay_clip(start: int = 0, end: int = 30) -> dict[str, Any]:
        ART.mkdir(parents=True, exist_ok=True)
        with STATE.lock:
            if STATE.replay is None:
                raise HTTPException(status_code=404, detail="no session recorded")
            path = STATE.replay.export_clip(ART / "clips" / "replay_clip.mp4", start, end)
        return {"ok": True, "path": str(path)}

    @app.post("/api/session/share")
    def session_share() -> dict[str, Any]:
        with STATE.lock:
            if STATE.replay is None:
                raise HTTPException(status_code=404, detail="no session recorded")
            src = STATE.replay.root
        dest = ART / "share" / src.name
        return export_share_bundle(src, dest)

    @app.post("/api/eval")
    def eval_bundle() -> dict[str, Any]:
        ART.mkdir(parents=True, exist_ok=True)
        return write_eval_bundle(ART / "eval", STATE.cfg)

    @app.post("/api/annotate")
    def annotate() -> dict[str, Any]:
        with STATE.lock:
            if STATE.replay is None:
                raise HTTPException(status_code=404, detail="no session recorded")
            src = STATE.replay.root / "perception" / "tracks.jsonl"
            root = STATE.replay.root
        out = ART / "annotation_task.json"
        task = tracks_to_annotation_task(src, out)
        xml = tracks_to_cvat_xml(src, ART / "cvat.xml")
        queue = write_active_queue(root, ART / "active_queue.json")
        return {"ok": True, "path": str(out), "n": len(task.get("items") or []), "cvat": str(xml), "active": queue}

    @app.get("/api/replay/hit")
    def replay_hit(x: float = 0, y: float = 0, i: int = 0) -> dict[str, Any]:
        with STATE.lock:
            if STATE.replay is None:
                raise HTTPException(status_code=404, detail="no session recorded")
            hit = STATE.replay.hit_test(i, x, y)
        return {"ok": hit is not None, "track": hit}

    @app.post("/api/annotate/revise")
    def annotate_revise(track_id: int = 0, frame_id: int = 0, semantic_type: str = "") -> dict[str, Any]:
        path = ART / "annotation_task.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="annotate first")
        patch = {"semantic_type": semantic_type} if semantic_type else {}
        apply_revision(path, track_id, frame_id, patch)
        return {"ok": True}

    @app.post("/api/train-synth")
    def train_synth() -> dict[str, Any]:
        ART.mkdir(parents=True, exist_ok=True)
        bundle = write_training_bundle(ART / "train_synth")
        return {"ok": True, "train_acc": bundle["model"]["train_acc"], "split": bundle["split"]}

    @app.post("/api/error-lib")
    def error_lib() -> dict[str, Any]:
        with STATE.lock:
            if STATE.replay is None:
                raise HTTPException(status_code=404, detail="no session recorded")
            src = STATE.replay.root
        dest = ART / "error_lib" / src.name
        return harvest_error_cases(src, dest)

    return app


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    ART.mkdir(parents=True, exist_ok=True)
    write_capability_report(ART / "capability_desktop.json")
    write_model_package(MODELS / "heuristic-cv-0.1.0", "heuristic-cv-0.1.0")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
