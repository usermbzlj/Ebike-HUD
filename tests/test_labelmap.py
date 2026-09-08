from rpar.enums import GeometryType, ObjectState, SemanticType, Severity
from rpar.ml.labelmap import map_geometry_alias, map_public_label, map_record


def test_rdd_pothole_is_depression_not_crack():
    m = map_public_label("rdd2022", "D40")
    assert m.semantic == SemanticType.POTHOLE
    assert m.geometry == GeometryType.CONCAVE
    assert m.state == ObjectState.ABNORMAL
    assert m.alert_eligible is True


def test_rdd_cracks_are_not_alerts():
    for raw in ("D00", "D10", "D20", "crack"):
        m = map_public_label("rdd2022", raw)
        assert m.semantic == SemanticType.ROUGH_BROKEN
        assert m.state == ObjectState.NORMAL
        assert m.alert_eligible is False


def test_rome_manhole_is_not_pothole():
    m = map_public_label("rome2026", "maintenance hole")
    assert m.semantic == SemanticType.MANHOLE_COVER
    assert m.geometry == GeometryType.FLAT
    assert m.state == ObjectState.NORMAL
    assert m.alert_eligible is False


def test_patch_and_speed_bump():
    patch = map_public_label("svrdd", "longitudinal patch")
    assert patch.semantic == SemanticType.REPAIR_PATCH
    assert patch.alert_eligible is False
    bump = map_public_label("bdroad", "Speed Breaker")
    assert bump.semantic == SemanticType.SPEED_BUMP
    assert bump.geometry == GeometryType.CONVEX


def test_unknown_class_never_becomes_pothole():
    m = map_public_label("misc", "random_blob")
    assert m.semantic == SemanticType.UNKNOWN_ANOMALY
    assert m.alert_eligible is False
    rec = map_record("misc", "random_blob")
    assert rec["semantic_type"] == "unknown_anomaly"


def test_geometry_and_severity_aliases():
    assert map_geometry_alias("depression") == GeometryType.CONCAVE
    assert map_geometry_alias("protrusion") == GeometryType.CONVEX
    m = map_public_label("cnrdd", "subsidence", severity="severe")
    assert m.severity == Severity.HEAVY
    assert m.semantic != SemanticType.POTHOLE
