from __future__ import annotations

import numpy as np

from rpar import SCHEMA_VERSION
from rpar.alerts import AlertPolicy, compose_phrase
from rpar.config import AlertConfig, load_config
from rpar.enums import (
    Direction,
    GeometryType,
    InferenceBackend,
    LifecycleState,
    ObjectState,
    PerceptionStatus,
    SemanticType,
    Severity,
    VisibilityClass,
    bump_kind,
    is_bump_hazard,
)
from rpar.ml.bump_dataset import clip_split, ingest_inbox, propose_labels, write_data_yaml
from rpar.ml.bump_prompts import (
    CLASS_TO_ID,
    clip_class_hint,
    detection_to_obs,
    keep_box,
    map_det_name,
    refine_manhole_geometry,
    spec_attrs,
    xyxy_to_yolo,
)
from rpar.ml.world_bump import try_load_world_bump
from rpar.models import FrameMeta, PerceptionResult, RoadObservation, SynchronizedFrame, TrackedRoadObject
from rpar.segengine import merge_bump_perception


def _frame(bgr: np.ndarray) -> SynchronizedFrame:
    h, w = bgr.shape[:2]
    return SynchronizedFrame(
        meta=FrameMeta(
            frame_id=0,
            sensor_timestamp_ns=1,
            image_timestamp_ns=1,
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
            availability={},
        ),
        bgr=bgr,
        pose=None,
        angular_velocity=None,
        linear_accel=None,
        location=None,
        speed_mps=10.0,
    )


def _obs(semantic, bbox, **kw) -> RoadObservation:
    x0, y0, x1, y1 = bbox
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return RoadObservation(
        timestamp_ns=1,
        source_frame_id=0,
        semantic_type=semantic,
        geometry_type=kw.get("geometry", GeometryType.CONCAVE),
        state=kw.get("state", ObjectState.ABNORMAL),
        severity=kw.get("severity", Severity.HEAVY),
        mask_rle=None,
        polygon=poly,
        bbox=bbox,
        model_confidence=kw.get("conf", 0.8),
        quality_at_mask=0.9,
        visibility=VisibilityClass.CLEAR,
        calibrated_confidence=0.8,
    )


def test_prompt_map_stays_on_spec_enums():
    assert map_det_name("large pothole") == "pothole"
    assert map_det_name("sunken manhole cover") == "manhole_cover"
    assert map_det_name("speed bump") == "speed_bump"
    sem, geo, st, _sev = spec_attrs("pothole", "large pothole")
    assert sem == SemanticType.POTHOLE and geo == GeometryType.CONCAVE and st == ObjectState.ABNORMAL
    sem, geo, st, _sev = spec_attrs("manhole_cover", "sunken manhole cover")
    assert geo == GeometryType.CONCAVE and st == ObjectState.ABNORMAL
    assert clip_class_hint("大坑_day_01.mp4") == "pothole"
    assert clip_class_hint("下沉井盖_night.mov") == "manhole_cover"
    assert clip_class_hint("减速带_01.mp4") == "speed_bump"
    assert clip_class_hint("mixed_ride.mp4") is None


def test_sunken_cover_is_darker_than_rim():
    img = np.full((80, 80, 3), 140, dtype=np.uint8)
    img[28:52, 28:52] = 40
    geo, st, _sev = refine_manhole_geometry(img, (10, 10, 70, 70))
    assert geo == GeometryType.CONCAVE
    assert st == ObjectState.ABNORMAL
    flat = np.full((80, 80, 3), 120, dtype=np.uint8)
    geo2, st2, _ = refine_manhole_geometry(flat, (10, 10, 70, 70))
    assert geo2 == GeometryType.FLAT
    assert st2 == ObjectState.NORMAL


def test_keep_box_rejects_full_frame_road():
    assert keep_box("pothole", (0, 120, 959, 534), 960, 540) is False
    assert keep_box("pothole", (200, 180, 380, 300), 640, 360) is True
    assert keep_box("manhole_cover", (500, 350, 570, 390), 960, 540) is True
    assert keep_box("manhole_cover", (693, 257, 807, 291), 960, 540) is True
    assert keep_box("pothole", (1, 154, 954, 535), 960, 540) is False
    assert keep_box("pothole", (20, 400, 80, 460), 960, 540) is False
    assert keep_box("pothole", (900, 200, 960, 248), 1920, 1080) is True
    assert keep_box("speed_bump", (620, 240, 1280, 280), 1920, 1080) is True
    assert keep_box("manhole_cover", (880, 190, 960, 230), 1920, 1080) is True
    bgr = np.zeros((360, 640, 3), dtype=np.uint8)
    obs = detection_to_obs(_frame(bgr), "large pothole", (200, 180, 380, 300), 0.62, bgr)
    assert obs is not None
    assert obs.semantic_type == SemanticType.POTHOLE
    assert obs.geometry_type == GeometryType.CONCAVE
    _xc, _yc, bw, bh = xyxy_to_yolo(obs.bbox, 640, 360)
    assert 0 < bw < 1 and 0 < bh < 1


def test_merge_bump_replaces_heuristic_when_model_hits():
    h = PerceptionResult(
        timestamp_ns=1,
        source_frame_id=0,
        road_polygon=[(0, 10), (40, 10), (40, 40), (0, 40)],
        occluded_polygons=[],
        observations=[
            _obs(SemanticType.SPEED_BUMP, (8, 8, 16, 16), geometry=GeometryType.CONVEX),
            _obs(SemanticType.ROAD_JOINT, (24, 24, 32, 32), geometry=GeometryType.FLAT, state=ObjectState.NORMAL, severity=Severity.NONE),
        ],
        backend=InferenceBackend.HEURISTIC,
        latency_ms=4.0,
        input_sizes=[(96, 48)],
    )
    empty = PerceptionResult(
        timestamp_ns=1,
        source_frame_id=0,
        road_polygon=[],
        occluded_polygons=[],
        observations=[],
        backend=InferenceBackend.GPU,
        latency_ms=12.0,
        input_sizes=[(640, 640)],
    )
    stripped = merge_bump_perception(h, empty)
    assert not any(o.semantic_type == SemanticType.SPEED_BUMP for o in stripped.observations)
    assert any(o.semantic_type == SemanticType.ROAD_JOINT for o in stripped.observations)
    bump = PerceptionResult(
        timestamp_ns=1,
        source_frame_id=0,
        road_polygon=[],
        occluded_polygons=[],
        observations=[_obs(SemanticType.POTHOLE, (20, 20, 36, 36))],
        backend=InferenceBackend.GPU,
        latency_ms=12.0,
        input_sizes=[(640, 640)],
    )
    m = merge_bump_perception(h, bump)
    kinds = {o.semantic_type for o in m.observations}
    assert SemanticType.POTHOLE in kinds
    assert SemanticType.SPEED_BUMP not in kinds
    assert SemanticType.ROAD_JOINT in kinds
    assert m.backend == InferenceBackend.GPU
    assert stripped.backend == InferenceBackend.GPU
    off_road = PerceptionResult(
        timestamp_ns=1,
        source_frame_id=0,
        road_polygon=[(0, 30), (20, 30), (20, 40), (0, 40)],
        occluded_polygons=[[(50, 50), (80, 50), (80, 80), (50, 80)]],
        observations=[],
        backend=InferenceBackend.GPU,
        latency_ms=12.0,
        input_sizes=[(640, 640)],
    )
    far = PerceptionResult(
        timestamp_ns=1,
        source_frame_id=0,
        road_polygon=[],
        occluded_polygons=[],
        observations=[
            _obs(SemanticType.POTHOLE, (4, 4, 12, 12)),
            _obs(SemanticType.SPEED_BUMP, (55, 55, 70, 70), geometry=GeometryType.CONVEX),
        ],
        backend=InferenceBackend.GPU,
        latency_ms=12.0,
        input_sizes=[(640, 640)],
    )
    kept_far = merge_bump_perception(off_road, far)
    assert any(o.semantic_type == SemanticType.POTHOLE and o.bbox[0] == 4 for o in kept_far.observations)
    assert not any(o.semantic_type == SemanticType.SPEED_BUMP for o in kept_far.observations)


def test_is_open_vocab_ignores_parent_folder_named_world():
    from pathlib import Path
    from rpar.ml.world_bump import _is_open_vocab

    assert _is_open_vocab(Path("models/yolo-world/yolov8m-worldv2.pt")) is True
    assert _is_open_vocab(Path("models/yolo-world/rpar-bump-vocab.pt")) is False
    assert _is_open_vocab(Path("models/bump-world-0.1.0/best.pt")) is False


def test_inbox_ingest_and_fake_proposals(tmp_path):
    import cv2

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    clip = inbox / "pothole_day_01.mp4"
    w, h = 320, 180
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    assert vw.isOpened()
    for _ in range(12):
        vw.write(np.full((h, w, 3), 50, dtype=np.uint8))
    vw.release()
    dest = tmp_path / "ds"
    ingested = ingest_inbox(inbox, dest, sample_fps=2.0, max_frames_per_clip=20)
    assert ingested["n_clips"] == 1
    assert ingested["n_frames"] >= 2
    assert ingested["split"]["train"]

    def fake_detect(bgr):
        hh, ww = bgr.shape[:2]
        return [{"name": "pothole", "bbox": (ww * 0.3, hh * 0.45, ww * 0.7, hh * 0.8), "conf": 0.7}]

    proposed = propose_labels(dest, detect_fn=fake_detect)
    assert proposed["ok"] is True
    assert proposed["n_positive_frames"] >= 1
    body = write_data_yaml(dest).read_text(encoding="utf-8")
    assert "pothole" in body and "speed_bump" in body and "manhole_cover" in body


def test_clip_split_keeps_whole_videos_together():
    train, val, _test = clip_split(["a", "b", "c", "d", "e", "f"], seed=7)
    assert not (set(train) & set(val))


def test_bump_phrases_and_default_alert_threshold():
    assert bump_kind(SemanticType.POTHOLE) == "大坑"
    assert is_bump_hazard(SemanticType.MANHOLE_COVER, GeometryType.CONCAVE, ObjectState.ABNORMAL)
    assert not is_bump_hazard(SemanticType.MANHOLE_COVER, GeometryType.FLAT, ObjectState.NORMAL)
    assert "大坑" in compose_phrase(Direction.CENTER_FRONT, SemanticType.POTHOLE)
    assert (
        compose_phrase(
            Direction.LEFT_FRONT,
            SemanticType.MANHOLE_COVER,
            geometry=GeometryType.CONCAVE,
            state=ObjectState.ABNORMAL,
        )
        == "左前方下沉井盖"
    )
    obj = TrackedRoadObject(
        schema_version=SCHEMA_VERSION,
        track_id=3,
        timestamp_ns=10**10,
        lifecycle_state=LifecycleState.CONFIRMED,
        semantic_type=SemanticType.POTHOLE,
        geometry_type=GeometryType.CONCAVE,
        object_state=ObjectState.ABNORMAL,
        severity=Severity.HEAVY,
        direction=Direction.CENTER_FRONT,
        distance_m=14.0,
        distance_confidence=0.8,
        distance_valid=True,
        ttc_s=1.2,
        model_confidence=0.35,
        visibility_confidence=0.8,
        temporal_confidence=0.7,
        geometry_consistency=0.7,
        effective_confidence=0.30,
        path_relevance=0.85,
        risk_score=0.4,
        alert_score=0.0,
        polygon=[(10, 10), (40, 10), (40, 40), (10, 40)],
        bbox=(10, 10, 40, 40),
        mask_rle=None,
        source_frame_id=1,
        mount_profile_id="left_handlebar_v1",
        model_version="yolo-world",
        visual_style="solid",
    )
    d = AlertPolicy(AlertConfig()).evaluate(obj, PerceptionStatus.NORMAL, 10**10, True)
    assert d.fired is True
    assert "大坑" in d.phrase


def test_missing_bump_weights_do_not_load_ultralytics(monkeypatch):
    monkeypatch.setattr("rpar.ml.world_bump.resolve_bump_weights", lambda path=None: None)
    assert try_load_world_bump() is None
    cfg = load_config()
    assert cfg.alert.bump_score_threshold <= 0.10
    assert CLASS_TO_ID["pothole"] == 0
