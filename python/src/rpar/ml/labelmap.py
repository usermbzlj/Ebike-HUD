"""Map public road-damage class names onto RPAR attributes.

Public datasets answer different questions (engineering distress vs rider impact).
Never concat their class tables into one YOLO list. Geometry is primary; semantic
is an attribute; cracks and ordinary manholes are not voice-alert targets.
"""

from __future__ import annotations

from dataclasses import dataclass

from rpar.enums import GeometryType, ObjectState, SemanticType, Severity


@dataclass(frozen=True)
class MappedLabel:
    semantic: SemanticType
    geometry: GeometryType
    state: ObjectState
    severity: Severity
    alert_eligible: bool
    source: str
    raw: str
    note: str = ""


def _norm(raw: str) -> str:
    return " ".join(str(raw).strip().lower().replace("_", " ").replace("-", " ").split())


def map_geometry_alias(raw: str) -> GeometryType:
    key = _norm(raw)
    table = {
        "concave": GeometryType.CONCAVE,
        "depression": GeometryType.CONCAVE,
        "sunken": GeometryType.CONCAVE,
        "convex": GeometryType.CONVEX,
        "protrusion": GeometryType.CONVEX,
        "raised": GeometryType.CONVEX,
        "rough": GeometryType.ROUGH,
        "step": GeometryType.STEP,
        "step transition": GeometryType.STEP,
        "flat": GeometryType.FLAT,
        "unknown": GeometryType.UNKNOWN,
        "uncertain": GeometryType.UNKNOWN,
    }
    return table.get(key, GeometryType.UNKNOWN)


def map_severity_alias(raw: str | int | None) -> Severity:
    if raw is None:
        return Severity.UNKNOWN
    if isinstance(raw, int):
        if raw in {0, 1, 2, 3}:
            return Severity(raw)
        return Severity.UNKNOWN
    key = _norm(str(raw))
    table = {
        "none": Severity.NONE,
        "0": Severity.NONE,
        "mild": Severity.LIGHT,
        "light": Severity.LIGHT,
        "1": Severity.LIGHT,
        "moderate": Severity.MEDIUM,
        "medium": Severity.MEDIUM,
        "2": Severity.MEDIUM,
        "severe": Severity.HEAVY,
        "heavy": Severity.HEAVY,
        "3": Severity.HEAVY,
    }
    return table.get(key, Severity.UNKNOWN)


def map_public_label(source: str, raw: str, severity: str | int | None = None) -> MappedLabel:
    """Convert one public class name. Unknown names stay unknown; they never become pothole."""
    src = _norm(source)
    key = _norm(raw)
    sev = map_severity_alias(severity)

    def out(
        semantic: SemanticType,
        geometry: GeometryType,
        state: ObjectState,
        default_sev: Severity,
        alert: bool,
        note: str,
    ) -> MappedLabel:
        return MappedLabel(
            semantic=semantic,
            geometry=geometry,
            state=state,
            severity=sev if sev != Severity.UNKNOWN else default_sev,
            alert_eligible=alert and state == ObjectState.ABNORMAL,
            source=src or "unknown",
            raw=str(raw),
            note=note,
        )

    # RDD2022 / China_MotorBike: D00/D10/D20 cracks, D40 pothole.
    if key in {"d00", "d10", "d20", "longitudinal crack", "transverse crack", "alligator crack", "crack", "cracks"}:
        return out(
            SemanticType.ROUGH_BROKEN,
            GeometryType.ROUGH,
            ObjectState.NORMAL,
            Severity.NONE,
            False,
            "crack is texture context, not a voice-alert target",
        )
    if key in {"d40", "pothole", "potholes"}:
        return out(
            SemanticType.POTHOLE,
            GeometryType.CONCAVE,
            ObjectState.ABNORMAL,
            Severity.MEDIUM,
            True,
            "vertical depression; still needs RideSet confirmation",
        )
    if key in {"maintenance hole", "maintenancehole", "manhole", "manhole cover", "manholecover"}:
        return out(
            SemanticType.MANHOLE_COVER,
            GeometryType.FLAT,
            ObjectState.NORMAL,
            Severity.NONE,
            False,
            "ordinary cover is hard-negative; height_state comes from self-labeling",
        )
    if key in {"open manhole", "open manhole cover", "sunken manhole", "sunken manhole cover", "settled manhole", "sunken cover"}:
        return out(
            SemanticType.MANHOLE_COVER,
            GeometryType.CONCAVE,
            ObjectState.ABNORMAL,
            Severity.HEAVY,
            True,
            "open or sunken cover is abnormal geometry, not a synonym of pothole",
        )
    if key in {"longitudinal patch", "transverse patch", "patch", "repair patch", "repair"}:
        return out(
            SemanticType.REPAIR_PATCH,
            GeometryType.FLAT,
            ObjectState.NORMAL,
            Severity.NONE,
            False,
            "flat patch must not alert",
        )
    if key in {"speed bump", "speed breaker", "speedbreaker", "hump"}:
        return out(
            SemanticType.SPEED_BUMP,
            GeometryType.CONVEX,
            ObjectState.ABNORMAL,
            Severity.MEDIUM,
            True,
            "protrusion class; public sets are small, RideSet required",
        )
    if key in {"subsidence"}:
        return out(
            SemanticType.UNKNOWN_ANOMALY,
            GeometryType.CONCAVE,
            ObjectState.ABNORMAL,
            Severity.MEDIUM,
            True,
            "CNRDD distress ≠ pothole; keep geometry, leave semantic unknown",
        )
    if key in {"rutting", "looseness"}:
        return out(
            SemanticType.ROUGH_BROKEN,
            GeometryType.ROUGH,
            ObjectState.UNKNOWN,
            Severity.LIGHT,
            False,
            "surface distress; do not treat as a pothole alert",
        )
    if key in {"drain", "gutter"}:
        return out(
            SemanticType.UNKNOWN_ANOMALY,
            GeometryType.FLAT,
            ObjectState.NORMAL,
            Severity.NONE,
            False,
            "drain/gutter is not in V0.1 core semantics",
        )
    if key in {"normal", "normal road", "plain road"}:
        return out(
            SemanticType.REPAIR_PATCH,
            GeometryType.FLAT,
            ObjectState.NORMAL,
            Severity.NONE,
            False,
            "image-level normal is a negative, not an instance",
        )
    if key in {"major damage", "minor damage"}:
        return out(
            SemanticType.UNKNOWN_ANOMALY,
            GeometryType.UNKNOWN,
            ObjectState.UNKNOWN,
            Severity.LIGHT if "minor" in key else Severity.MEDIUM,
            False,
            "BDRoad-Sense image-level class; do not invent a pothole instance",
        )
    return out(
        SemanticType.UNKNOWN_ANOMALY,
        GeometryType.UNKNOWN,
        ObjectState.UNKNOWN,
        Severity.UNKNOWN,
        False,
        "unmapped public class stays unknown; never coerce to pothole",
    )


def map_record(source: str, raw: str, severity: str | int | None = None) -> dict[str, object]:
    m = map_public_label(source, raw, severity)
    return {
        "source": m.source,
        "raw": m.raw,
        "semantic_type": m.semantic.value,
        "geometry_type": m.geometry.value,
        "state": m.state.value,
        "severity": int(m.severity),
        "alert_eligible": m.alert_eligible,
        "note": m.note,
    }
