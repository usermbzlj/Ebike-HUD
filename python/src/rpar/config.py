"""Versioned configuration (MOD-005). Every perception / policy threshold lives here.

`configs/rpar.defaults.yaml` is the single source of truth. The Android asset
`rpar.defaults.json` is generated from it (`rpar export-android-config`) and a test
keeps the two in sync.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "1.1"


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
    # Occluder polygons covering this share of the frame put the HUD in OCCLUDED.
    occlusion_status_ratio: float = 0.08
    # Above this share no new tracks may spawn (spec appendix B).
    occlusion_block_ratio: float = 0.40
    # Tiles that stay motionless this long while the scene moves are lens contamination.
    lens_static_seconds: float = 2.0
    lens_static_tiles: int = 3
    # Consecutive unusable seconds before PERCEPTION_LIMITED.
    bad_streak_limited_s: float = 0.45


@dataclass
class PerceptionConfig:
    # Asphalt luma (0-255) at which the night score's luma term reaches 0.5; see perception.night_score.
    night_road_luma: float = 100.0
    # Night score (luma + colour-cast + dark-sky cues) at or above which conservative night rules apply.
    night_score_threshold: float = 0.6
    # Minimum dark-blob area (px) at 1080p; scaled with ROI resolution.
    pothole_min_area_px: float = 220.0
    pothole_min_area_night_px: float = 1400.0
    pothole_circularity: float = 0.62
    # Any dark blob needs this much contrast to count as an anomaly at all...
    anomaly_contrast: float = 12.0
    # ...and this much (a shadowed interior) before it is called a pothole. Fresh repair
    # patches are ~30-40 grey levels darker than asphalt; holes are 70+.
    pothole_contrast: float = 45.0
    pothole_contrast_night: float = 22.0
    # A cover is "sunken" when its centre is this much darker than the rim.
    manhole_sunken_delta: float = 12.0
    # Occluder (vehicle) blob must cover this share of the frame.
    occluder_min_frac: float = 0.012
    # Normalized far / near crops of the driving corridor (x0, y0, x1, y1).
    far_roi: tuple[float, float, float, float] = (0.18, 0.32, 0.82, 0.62)
    near_roi: tuple[float, float, float, float] = (0.08, 0.50, 0.92, 1.00)
    # Near crop is downscaled to at most this width; far crop keeps native pixel density.
    near_max_width: int = 960


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
    bump_confirm_hits: int = 2


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
    # Assumed contact-point localisation error (px at 1080p) used for distance confidence.
    contact_sigma_px: float = 4.0
    # Look-ahead (m) beyond which path relevance decays to zero.
    relevance_horizon_m: float = 40.0


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
    # Pothole / speed bump / sunken cover score threshold (no other gate is bypassed).
    # A confirmed MEDIUM pothole in the corridor with good visibility scores ~0.22-0.35;
    # a 0.35-confidence or off-corridor one stays below 0.2.
    bump_score_threshold: float = 0.22
    # Risk weights indexed by Severity: none, light, medium, heavy, unknown.
    severity_weights: tuple[float, float, float, float, float] = (0.05, 0.3, 0.7, 1.0, 0.22)


@dataclass
class RenderConfig:
    riding_max_labels: int = 5
    ar_target_fps: int = 60
    ar_min_fps: int = 30
    candidate_alpha: float = 0.35
    confirmed_alpha: float = 0.85
    overlay_error_px: float = 8.0
    night_brightness: float = 0.72
    stroke_scale: float = 1.0
    font_scale: float = 1.0
    overlay_alpha: float = 1.0
    show_info_layer: bool = True


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
    # Preferred horizontal field of view (deg) when choosing among rear cameras.
    preferred_hfov_deg: float = 75.0


@dataclass
class ThermalConfig:
    # Skin / battery temperature (C) steps. Below warn_c nothing is throttled.
    warn_c: float = 42.0
    drop_far_roi_c: float = 43.0
    drop_infer_hz_c: float = 45.0
    slow_factor: float = 0.7


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
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelPackageRef = field(default_factory=ModelPackageRef)
    default_mount_id: str = "left_handlebar_v1"

    def snapshot(self) -> dict[str, Any]:
        return _to_plain(self)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    return obj


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class ConfigError(ValueError):
    pass


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) under '{path}': {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        default = getattr(cls(), name)
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value, f"{path}.{name}")
        elif isinstance(default, tuple):
            if not isinstance(value, (list, tuple)):
                raise ConfigError(f"'{path}.{name}' must be a list")
            kwargs[name] = tuple(value)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"'{path}.{name}' must be true/false")
            kwargs[name] = value
        elif isinstance(default, int) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"'{path}.{name}' must be a number")
            kwargs[name] = int(value)
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"'{path}.{name}' must be a number")
            kwargs[name] = float(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _from_dict(data: dict[str, Any]) -> RparConfig:
    return _build(RparConfig, data, "rpar")


def default_config() -> RparConfig:
    return RparConfig()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return repo_root() / "configs" / "rpar.defaults.yaml"


def load_config(path: str | Path | None = None) -> RparConfig:
    cfg = default_config()
    if path is None:
        here = default_config_path()
        path = here if here.exists() else None
    if path is None:
        return cfg
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return _from_dict(_merge(cfg.snapshot(), raw))


def save_config(cfg: RparConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.snapshot(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def android_config_json(cfg: RparConfig) -> str:
    """The exact text written to `android/app/src/main/assets/rpar.defaults.json`."""
    return json.dumps(cfg.snapshot(), indent=2, ensure_ascii=False) + "\n"


def export_android_config(cfg: RparConfig | None = None, dest: str | Path | None = None) -> Path:
    cfg = cfg or load_config()
    dest = Path(dest) if dest else repo_root() / "android" / "app" / "src" / "main" / "assets" / "rpar.defaults.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(android_config_json(cfg), encoding="utf-8")
    return dest
