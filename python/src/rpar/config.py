"""Versioned configuration (MOD-005). No perception thresholds live as UI literals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "1.0"


@dataclass
class QualityConfig:
    laplacian_usable: float = 18.0
    motion_blur_block: float = 0.62
    glare_block: float = 0.45
    underexposure_block: float = 0.72
    overexposure_block: float = 0.55
    min_road_visible: float = 0.18
    max_selected_age_ms: float = 100.0
    buffer_seconds: float = 0.5
    tile_cols: int = 8
    tile_rows: int = 6
    lens_drop_blob_min: int = 12


@dataclass
class TrackingConfig:
    confirm_window_s: float = 0.40
    min_confirm_hits: int = 3
    candidate_max_age_s: float = 0.35
    low_quality_hold_s: float = 0.45
    passed_remove_s: float = 0.50
    iou_match: float = 0.25
    center_match_px: float = 80.0
    process_noise: float = 18.0
    meas_noise: float = 6.0
    unknown_anomaly_extra_hits: int = 2


@dataclass
class GeometryConfig:
    corridor_half_width_m: float = 0.85
    across_min_width_m: float = 1.6
    near_boundary_m: float = 2.2
    far_display_m: float = 30.0
    distance_mae_near_m: float = 2.5
    distance_mae_far_m: float = 5.0
    min_speed_for_ttc_mps: float = 2.0
    pitch_health_deg: float = 8.0
    roll_health_deg: float = 10.0
    hide_distance_if_invalid: bool = True


@dataclass
class AlertConfig:
    enabled_default: bool = True
    score_threshold: float = 0.62
    global_cooldown_s: float = 2.2
    class_cooldown_s: float = 4.0
    min_severity: int = 2
    min_path_relevance: float = 0.45
    min_visibility: float = 0.40
    min_effective: float = 0.48
    realert_severity_jump: int = 1
    pause_on_degraded: bool = True


@dataclass
class RenderConfig:
    riding_max_labels: int = 5
    ar_target_fps: int = 60
    ar_min_fps: int = 30
    candidate_alpha: float = 0.35
    confirmed_alpha: float = 0.85
    overlay_error_px: float = 8.0
    night_brightness: float = 0.72


@dataclass
class CameraConfig:
    width: int = 1920
    height: int = 1080
    target_fps: int = 60
    fallback_fps: int = 30
    dynamic_range: str = "SDR"
    prefer_preview_plus_yuv_plus_record: bool = True
    segment_seconds: int = 300
    bitrate_mbps: float = 25.0
    lock_physical_camera: bool = True


@dataclass
class RuntimeConfig:
    infer_fps: float = 12.0
    quality_fps: float = 30.0
    tracking_fps: float = 30.0
    e2e_p95_ms: float = 150.0
    thermal_min_infer_fps: float = 8.0
    memory_target_mb: float = 1536.0
    startup_preview_s: float = 5.0


@dataclass
class ModelPackageRef:
    package_id: str = "heuristic-cv-0.1.0"
    engine: str = "heuristic"
    input_far: tuple[int, int] = (768, 384)
    input_near: tuple[int, int] = (640, 480)
    sha256: str = ""
    compatible_app: str = ">=0.1.0"


@dataclass
class RparConfig:
    schema_version: str = SCHEMA_VERSION
    quality: QualityConfig = field(default_factory=QualityConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelPackageRef = field(default_factory=ModelPackageRef)
    default_mount_id: str = "left_handlebar_v1"

    def snapshot(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _from_dict(data: dict[str, Any]) -> RparConfig:
    cfg = RparConfig()
    if "quality" in data:
        cfg.quality = QualityConfig(**{**cfg.quality.__dict__, **data["quality"]})
    if "tracking" in data:
        cfg.tracking = TrackingConfig(**{**cfg.tracking.__dict__, **data["tracking"]})
    if "geometry" in data:
        cfg.geometry = GeometryConfig(**{**cfg.geometry.__dict__, **data["geometry"]})
    if "alert" in data:
        cfg.alert = AlertConfig(**{**cfg.alert.__dict__, **data["alert"]})
    if "render" in data:
        cfg.render = RenderConfig(**{**cfg.render.__dict__, **data["render"]})
    if "camera" in data:
        cam = {**cfg.camera.__dict__, **data["camera"]}
        cfg.camera = CameraConfig(**cam)
    if "runtime" in data:
        cfg.runtime = RuntimeConfig(**{**cfg.runtime.__dict__, **data["runtime"]})
    if "model" in data:
        mdl = {**cfg.model.__dict__, **data["model"]}
        if "input_far" in mdl:
            mdl["input_far"] = tuple(mdl["input_far"])
        if "input_near" in mdl:
            mdl["input_near"] = tuple(mdl["input_near"])
        cfg.model = ModelPackageRef(**mdl)
    if "default_mount_id" in data:
        cfg.default_mount_id = data["default_mount_id"]
    if "schema_version" in data:
        cfg.schema_version = data["schema_version"]
    return cfg


def default_config() -> RparConfig:
    return RparConfig()


def load_config(path: str | Path | None = None) -> RparConfig:
    cfg = default_config()
    if path is None:
        here = Path(__file__).resolve().parents[3] / "configs" / "rpar.defaults.yaml"
        path = here if here.exists() else None
    if path is None:
        return cfg
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _from_dict(_merge(cfg.snapshot(), raw))


def save_config(cfg: RparConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.snapshot(), sort_keys=False, allow_unicode=True), encoding="utf-8")
