from __future__ import annotations

from rpar.alerts import AlertPolicy, compose_phrase
from rpar.config import AlertConfig
from rpar.enums import ALERT_FORBIDDEN_PHRASES, Direction, GeometryType, LifecycleState, ObjectState, PerceptionStatus, SemanticType, Severity
from rpar.models import TrackedRoadObject
from rpar import SCHEMA_VERSION


def _obj(**kw) -> TrackedRoadObject:
    base = dict(
        schema_version=SCHEMA_VERSION,
        track_id=7,
        timestamp_ns=1_000_000_000,
        lifecycle_state=LifecycleState.CONFIRMED,
        semantic_type=SemanticType.POTHOLE,
        geometry_type=GeometryType.CONCAVE,
        object_state=ObjectState.ABNORMAL,
        severity=Severity.HEAVY,
        direction=Direction.CENTER_FRONT,
        distance_m=12.0,
        distance_confidence=0.8,
        distance_valid=True,
        ttc_s=1.1,
        model_confidence=0.9,
        visibility_confidence=0.85,
        temporal_confidence=0.9,
        geometry_consistency=0.85,
        effective_confidence=0.7,
        path_relevance=0.9,
        risk_score=0.7,
        alert_score=0.0,
        polygon=[(10, 10), (40, 10), (40, 40), (10, 40)],
        bbox=(10, 10, 40, 40),
        mask_rle=None,
        source_frame_id=3,
        mount_profile_id="left_handlebar_v1",
        model_version="heuristic-cv-0.1.0",
        visual_style="solid",
    )
    base.update(kw)
    return TrackedRoadObject(**base)


def test_phrase_has_no_steering_advice():
    phrase = compose_phrase(Direction.LEFT_FRONT, SemanticType.POTHOLE)
    assert phrase.startswith("左前方")
    assert "大坑" in phrase
    for bad in ALERT_FORBIDDEN_PHRASES:
        assert bad not in phrase


def test_sunken_manhole_phrase():
    phrase = compose_phrase(
        Direction.CENTER_FRONT,
        SemanticType.MANHOLE_COVER,
        geometry=GeometryType.CONCAVE,
        state=ObjectState.ABNORMAL,
    )
    assert phrase == "正前方下沉井盖"


def test_normal_manhole_does_not_alert():
    pol = AlertPolicy(AlertConfig())
    obj = _obj(semantic_type=SemanticType.MANHOLE_COVER, object_state=ObjectState.NORMAL, geometry_type=GeometryType.FLAT, severity=Severity.NONE)
    d = pol.evaluate(obj, PerceptionStatus.NORMAL, 10**10, True)
    assert d.fired is False
    assert "normal_or_flat" in d.reasons


def test_normal_crack_texture_does_not_alert():
    pol = AlertPolicy(AlertConfig(score_threshold=0.01, min_effective=0.0, min_visibility=0.0, min_severity=0))
    obj = _obj(
        semantic_type=SemanticType.ROUGH_BROKEN,
        object_state=ObjectState.NORMAL,
        geometry_type=GeometryType.ROUGH,
        severity=Severity.NONE,
    )
    d = pol.evaluate(obj, PerceptionStatus.NORMAL, 10**10, True)
    assert d.fired is False
    assert "normal_or_flat" in d.reasons


def test_off_corridor_display_but_no_alert():
    pol = AlertPolicy(AlertConfig())
    obj = _obj(path_relevance=0.1)
    d = pol.evaluate(obj, PerceptionStatus.NORMAL, 10**10, True)
    assert d.fired is False
    assert "off_corridor" in d.reasons


def test_low_quality_pauses_alerts():
    pol = AlertPolicy(AlertConfig())
    d = pol.evaluate(_obj(), PerceptionStatus.SEVERE_BLUR, 10**10, True)
    assert d.fired is False
    assert "quality_pause" in d.reasons
    bump = pol.evaluate(_obj(), PerceptionStatus.OCCLUDED, 10**10, True)
    assert bump.fired is True
    other = pol.evaluate(
        _obj(track_id=8, semantic_type=SemanticType.ROUGH_BROKEN, geometry_type=GeometryType.ROUGH),
        PerceptionStatus.OCCLUDED,
        10**10,
        True,
    )
    assert other.fired is False
    assert "quality_pause" in other.reasons


def test_one_alert_per_track_and_cooldown():
    pol = AlertPolicy(AlertConfig(score_threshold=0.05, min_effective=0.0, min_visibility=0.0))
    obj = _obj()
    a1 = pol.evaluate(obj, PerceptionStatus.NORMAL, 10**10, True)
    assert a1.fired
    a2 = pol.evaluate(obj, PerceptionStatus.NORMAL, 10**10 + 1000, True)
    assert a2.fired is False
    assert "already_alerted" in a2.reasons
    other = _obj(track_id=99)
    a3 = pol.evaluate(other, PerceptionStatus.NORMAL, 10**10 + 2000, True)
    assert a3.fired is False
    assert "global_cooldown" in a3.reasons


def test_candidate_never_alerts():
    pol = AlertPolicy(AlertConfig(score_threshold=0.01))
    d = pol.evaluate(_obj(lifecycle_state=LifecycleState.CANDIDATE), PerceptionStatus.NORMAL, 10**10, True)
    assert d.fired is False
