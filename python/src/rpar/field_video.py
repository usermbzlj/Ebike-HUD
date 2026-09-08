"""Discover and run the local Video/ field clips (spec 1.2 / 11.4). MP4s stay untracked."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar import SCHEMA_VERSION
from rpar.config import RparConfig, load_config
from rpar.enums import INFO_LAYER_SEMANTICS, UiMode
from rpar.golden import run_video_file
from rpar.perception import load_engine, load_field_engine

# Spec §1.2 sample durations. Copies may be transcoded (fps/size differ; duration is the fingerprint).
_SPEC_CLIPS = (
    {"alias": "day_25007", "spec_name": "25007.mp4", "duration_s": 68.824, "lighting": "day"},
    {"alias": "night_25013", "spec_name": "25013.mp4", "duration_s": 55.014, "lighting": "night"},
)


def repo_video_dir(start: Path | None = None) -> Path:
    here = Path(start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        cand = p / "Video"
        if cand.is_dir():
            return cand
    return Path.cwd() / "Video"


def _rel_path(path: Path, video_dir: Path | None = None) -> str:
    root = (video_dir or repo_video_dir()).resolve().parent
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path, limit: int = 2_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        buf = f.read(limit)
        h.update(buf)
        if path.stat().st_size > limit:
            f.seek(max(0, path.stat().st_size - 65536))
            h.update(f.read())
            h.update(str(path.stat().st_size).encode())
    return h.hexdigest()


def probe_clip(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "ok": False, "error": "unreadable"}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    dur = n / fps if fps else 0.0
    means: list[float] = []
    for idx in [0, n // 4, n // 2, 3 * n // 4, max(0, n - 1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, im = cap.read()
        if ok:
            means.append(float(im.mean()))
    cap.release()
    luma = float(np.mean(means)) if means else 0.0
    lighting = "night" if luma < 105 else "day"
    alias = "extra"
    spec_name = None
    for spec in _SPEC_CLIPS:
        if abs(dur - spec["duration_s"]) <= 1.0:
            alias = spec["alias"]
            spec_name = spec["spec_name"]
            lighting = spec["lighting"]
            break
    return {
        "ok": True,
        "path": str(path),
        "relpath": _rel_path(path),
        "name": path.name,
        "alias": alias,
        "spec_name": spec_name,
        "width": w,
        "height": h,
        "fps": round(fps, 3),
        "frames": n,
        "duration_s": round(dur, 3),
        "mean_luma": round(luma, 2),
        "lighting": lighting,
        "bytes": path.stat().st_size,
        "sha256_prefix": _sha256(path),
        "note": "Likely a transcoded copy if size/fps differ from the 1080p60 / 1080p SDR originals.",
    }


def list_clips(video_dir: Path | None = None) -> list[dict[str, Any]]:
    d = Path(video_dir) if video_dir else repo_video_dir()
    if not d.is_dir():
        return []
    return [probe_clip(p) for p in sorted(d.glob("*.mp4")) if p.is_file()]


def write_catalog(video_dir: Path | None = None) -> dict[str, Any]:
    d = Path(video_dir) if video_dir else repo_video_dir()
    clips = list_clips(d)

    def public(clip: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in clip.items() if k != "path"}

    published = [public(c) for c in clips]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "directory": "Video",
        "n_clips": len(published),
        "clips": published,
        "spec_map": {
            "25007.mp4": next((c for c in published if c.get("alias") == "day_25007"), None),
            "25013.mp4": next((c for c in published if c.get("alias") == "night_25013"), None),
        },
    }
    out = d / "catalog.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def extract_preview(path: Path, dest: Path, at_ratio: float = 0.35) -> Path | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, int(n * at_ratio))))
    ok, im = cap.read()
    cap.release()
    if not ok:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), im, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return dest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def probe_seg_clip(path: Path, cfg: RparConfig | None = None, max_frames: int = 8) -> dict[str, Any]:
    """Run hybrid classmap sidecar on a few real frames; no geometric GT."""
    cfg = cfg or load_config()
    pkg = _repo_root() / "models" / "roadseg-synth-0.1.0"
    engine = load_engine(cfg, pkg if (pkg / "seg_weights.json").exists() else None)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "reason": "unreadable"}
    from rpar.models import FrameMeta, SynchronizedFrame

    n = 0
    road_frac = []
    n_obs = 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    while n < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        ts = 1_000_000_000_000 + n * 33_000_000
        frame = SynchronizedFrame(
            meta=FrameMeta(
                frame_id=n,
                sensor_timestamp_ns=ts,
                image_timestamp_ns=ts,
                exposure_time_ns=None,
                iso=None,
                focal_length_mm=None,
                focus_distance_diopters=None,
                af_state=None,
                ae_state=None,
                awb_state=None,
                crop_region=None,
                stabilization_mode=None,
                width=w,
                height=h,
                availability={"exposure": False, "iso": False},
            ),
            bgr=bgr,
            pose=None,
            angular_velocity=None,
            linear_accel=None,
            location=None,
            speed_mps=10.0,
        )
        result = engine.infer(frame)
        n_obs += len(result.observations)
        if result.road_polygon:
            mask = np.zeros((h, w), np.uint8)
            pts = np.array(result.road_polygon, dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillConvexPoly(mask, pts, 1)
                road_frac.append(float(mask.mean()))
        n += 1
    cap.release()
    cap_info = engine.capability()
    engine.close()
    return {
        "ok": n > 0,
        "frames": n,
        "hybrid": bool(cap_info.get("hybrid")),
        "mean_road_frac": float(np.mean(road_frac)) if road_frac else 0.0,
        "n_observations": n_obs,
        "package": str(pkg) if (pkg / "seg_weights.json").exists() else None,
    }


def _sync_frame(bgr: np.ndarray, idx: int, w: int, h: int):
    from rpar.models import FrameMeta, SynchronizedFrame

    ts = 1_000_000_000_000 + idx * 33_000_000
    return SynchronizedFrame(
        meta=FrameMeta(
            frame_id=idx,
            sensor_timestamp_ns=ts,
            image_timestamp_ns=ts,
            exposure_time_ns=None,
            iso=None,
            focal_length_mm=None,
            focus_distance_diopters=None,
            af_state=None,
            ae_state=None,
            awb_state=None,
            crop_region=None,
            stabilization_mode=None,
            width=w,
            height=h,
            availability={"exposure": False, "iso": False},
        ),
        bgr=bgr,
        pose=None,
        angular_velocity=None,
        linear_accel=None,
        location=None,
        speed_mps=10.0,
    )


def render_yolop_stills(path: Path, dest: Path, engine, ratios: tuple[float, ...] = (0.28, 0.48, 0.68)) -> list[str]:
    """Direct YOLOPv2 wash overlays (green road + vehicle boxes) at a few timestamps."""
    sidecar = getattr(engine, "sidecar", None)
    if sidecar is None or not hasattr(sidecar, "last_da_mask"):
        return []
    from rpar.ml.yolopv2 import paint_drivable

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for r in ratios:
        idx = max(0, min(n - 1, int(n * r)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        result = sidecar.infer(_sync_frame(bgr, idx, w, h))
        painted = paint_drivable(bgr, sidecar.last_da_mask, sidecar.last_boxes, result.road_polygon)
        dest_path = dest / f"yolop_{idx:04d}.jpg"
        cv2.imwrite(str(dest_path), painted, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        out.append(str(dest_path))
    cap.release()
    return out


_INFO_SEM = {s.value for s in INFO_LAYER_SEMANTICS}


def m2_clip_gates(clip: dict[str, Any]) -> dict[str, Any]:
    """Appendix B software proxies for a single field clip. Not GT first-confirm or PKC110 10 h."""
    run = clip.get("run") if isinstance(clip.get("run"), dict) else {}
    alias = str(clip.get("alias") or "")
    lighting = str(clip.get("lighting") or run.get("lighting") or "unknown")
    frames = int(run.get("frames") or 0)
    overlay_ok = bool(run.get("overlay_ok"))
    n_conf = int(run.get("n_confirmed_tracks") or 0)
    n_alerts = int(run.get("n_alerts_fired") or 0)
    n_rough = int(run.get("n_rough_broken_confirmed") or 0)
    sem = run.get("confirmed_semantics") if isinstance(run.get("confirmed_semantics"), dict) else {}
    n_info = int(run.get("n_info_confirmed") or 0)
    if not n_info and sem:
        n_info = sum(int(v) for k, v in sem.items() if str(k) in _INFO_SEM)
    first_med = run.get("first_confirm_median_m")
    n_solid = max(0, n_conf - n_info)
    road_share = float(run.get("road_frame_share") or 0.0)
    per_min = float(run.get("confirmed_per_min") or 0.0)
    is_night = alias == "night_25013"
    is_day = alias == "day_25007"
    checks: dict[str, bool] = {
        "overlay_ok": overlay_ok,
        "has_frames": frames > 0,
    }
    if is_night:
        checks["night_no_solid_damage"] = n_solid == 0 and n_rough == 0
        checks["night_no_alerts"] = n_alerts == 0
        checks["night_no_rough_broken"] = n_rough == 0
    if is_day:
        checks["day_road_polygon"] = road_share >= 0.05
    passed = all(checks.values()) if checks else False
    return {
        "alias": alias,
        "lighting": lighting,
        "checks": checks,
        "pass": passed,
        "n_confirmed_tracks": n_conf,
        "n_info_confirmed": n_info,
        "n_solid_confirmed": n_solid,
        "n_alerts_fired": n_alerts,
        "confirmed_per_min": per_min,
        "road_frame_share": road_share,
        "first_confirm_median_m": first_med,
        "first_confirm_is_gt": bool(run.get("first_confirm_is_gt")),
        "spec_solid_fp_per_min": 1.0 if is_night else (0.5 if is_day else None),
        "note": "Night solid-damage==0 is an asphalt-noise proxy; info-layer puddle/gravel is allowed. Predicted first-confirm is not geometric GT.",
    }


def m2_field_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    gates = [m2_clip_gates(r) for r in results if r.get("ok") or r.get("run")]
    runnable = [g for g in gates if g.get("alias")]
    return {
        "schema_version": SCHEMA_VERSION,
        "gates": runnable,
        "pass": all(g["pass"] for g in runnable) if runnable else False,
        "n_clips": len(runnable),
        "not_claimed": [
            "first_confirm_distance_with_geometric_GT",
            "PKC110_Camera2_1080p60_SDR",
            "10h_false_voice_rate",
            "60min_thermal_no_crash",
        ],
        "note": "M2 software gates on local Video/ transcodes. Spec 11.2 distance/recall still need PKC110 + GT.",
    }


def run_field_videos(
    video_dir: Path | None = None,
    out_dir: Path | None = None,
    cfg: RparConfig | None = None,
    max_frames: int = 180,
    *,
    prefer_yolop: bool = False,
    ui_mode: UiMode | None = None,
) -> dict[str, Any]:
    """Run the field pipeline on every local clip. YOLOPv2 is opt-in so pytest stays fast."""
    cfg = cfg or load_config()
    mode = ui_mode or UiMode.RIDING
    d = Path(video_dir) if video_dir else repo_video_dir()
    out_dir = Path(out_dir) if out_dir else Path("artifacts") / "field_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = write_catalog(d)
    live = {c["name"]: c for c in list_clips(d)}
    engine = load_field_engine(cfg, prefer_yolop=prefer_yolop)
    results: list[dict[str, Any]] = []
    hybrid = False
    try:
        hybrid = bool(engine.capability().get("hybrid")) if hasattr(engine, "capability") else False
        for clip in catalog["clips"]:
            if not clip.get("ok"):
                results.append(clip)
                continue
            src = Path(live[clip["name"]]["path"])
            alias = clip["alias"]
            dest = out_dir / alias
            dest.mkdir(parents=True, exist_ok=True)
            extract_preview(src, dest / "preview.jpg")
            metrics = run_video_file(src, dest, cfg, max_frames=max_frames, engine=engine, ui_mode=mode)
            yolop_stills = render_yolop_stills(src, dest, engine) if prefer_yolop else []
            row = {
                **clip,
                "run": metrics,
                "yolop_stills": yolop_stills,
                "seg": probe_seg_clip(src, cfg, max_frames=min(8, max_frames)),
            }
            (dest / "metrics.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            results.append(row)
    finally:
        engine.close()
    m2 = m2_field_report(results)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "catalog": catalog,
        "max_frames": max_frames,
        "prefer_yolop": prefer_yolop,
        "ui_mode": mode.value,
        "hybrid": hybrid,
        "results": results,
        "n_ok": sum(1 for r in results if r.get("run", {}).get("frames", 0) > 0),
        "m2": m2,
    }
    (out_dir / "m2_report.json").write_text(json.dumps(m2, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "field_video.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle
