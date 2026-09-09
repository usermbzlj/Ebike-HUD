"""Conservative audio policy: visual may be busy; voice must be gated (ALT-001..010)."""

from __future__ import annotations

from dataclasses import dataclass, field

from rpar.config import AlertConfig
from rpar.enums import (
    ALERT_FORBIDDEN_PHRASES,
    DIRECTION_TTS,
    INFO_LAYER_SEMANTICS,
    LOW_RISK_WHEN_NORMAL,
    SEMANTIC_TTS,
    Direction,
    GeometryType,
    LifecycleState,
    ObjectState,
    PerceptionStatus,
    SemanticType,
    Severity,
    bump_kind,
    is_bump_hazard,
)
from rpar.models import AlertDecision, TrackedRoadObject


def visual_score(
    calibrated: float,
    visibility: float,
    temporal: float,
    geometry_consistency: float,
) -> float:
    return float(max(0.0, calibrated) * max(0.0, visibility) * max(0.0, temporal) * max(0.0, geometry_consistency))


def urgency_from_ttc(ttc_s: float | None) -> float:
    if ttc_s is None:
        return 0.35
    if ttc_s <= 0.8:
        return 1.0
    if ttc_s <= 1.6:
        return 0.85
    if ttc_s <= 2.8:
        return 0.6
    return 0.25


def severity_score(sev: Severity, geometry: GeometryType, state: ObjectState) -> float:
    if state == ObjectState.NORMAL or geometry == GeometryType.FLAT:
        return 0.02
    table = {
        Severity.NONE: 0.05,
        Severity.LIGHT: 0.28,
        Severity.MEDIUM: 0.7,
        Severity.HEAVY: 1.0,
        Severity.UNKNOWN: 0.22,
    }
    return table.get(sev, 0.22)


def compose_phrase(
    direction: Direction,
    semantic: SemanticType,
    generic: bool = False,
    *,
    geometry: GeometryType | None = None,
    state: ObjectState | None = None,
    severity: Severity | None = None,
) -> str:
    heading = DIRECTION_TTS.get(direction, "前方")
    if generic:
        kind = "路面异常"
    elif is_bump_hazard(semantic, geometry, state):
        kind = bump_kind(semantic, geometry, state, severity)
    else:
        kind = SEMANTIC_TTS.get(semantic, "路面异常")
    return f"{heading}{kind}"


def _validate_phrase_tables() -> None:
    """ALT-010: no steering / braking advice can ever be spoken. Checked once at import."""
    for text in [*DIRECTION_TTS.values(), *SEMANTIC_TTS.values(), "路面异常", "大坑", "减速带", "下沉井盖"]:
        for bad in ALERT_FORBIDDEN_PHRASES:
            if bad in text:
                raise RuntimeError(f"forbidden advisory text in TTS table: {bad}")


_validate_phrase_tables()


def severity_weight(sev: Severity, weights: tuple[float, ...]) -> float:
    """Risk weight for a Severity from config `alert.severity_weights` (none, light, medium, heavy, unknown)."""
    idx = int(sev)
    if idx < 0:
        idx = 4
    if idx >= len(weights):
        return float(weights[-1]) if weights else 0.22
    return float(weights[idx])


@dataclass
class AlertPolicy:
    cfg: AlertConfig
    enabled: bool = True
    last_global_ns: int = 0
    last_class_ns: dict[str, int] = field(default_factory=dict)
    alerted_tracks: dict[int, int] = field(default_factory=dict)
    last_severity: dict[int, int] = field(default_factory=dict)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def evaluate(
        self,
        obj: TrackedRoadObject,
        status: PerceptionStatus,
        now_ns: int,
        geometry_valid: bool,
    ) -> AlertDecision:
        reasons: list[str] = []
        snapshot = {
            "effective_confidence": obj.effective_confidence,
            "visibility_confidence": obj.visibility_confidence,
            "temporal_confidence": obj.temporal_confidence,
            "geometry_consistency": obj.geometry_consistency,
            "path_relevance": obj.path_relevance,
            "risk_score": obj.risk_score,
            "alert_score": obj.alert_score,
            "distance_m": obj.distance_m,
            "distance_valid": obj.distance_valid,
            "ttc_s": obj.ttc_s,
            "lifecycle_state": obj.lifecycle_state.value,
            "severity": int(obj.severity),
            "object_state": obj.object_state.value,
            "geometry_type": obj.geometry_type.value,
            "semantic_type": obj.semantic_type.value,
            "threshold": self.cfg.score_threshold,
            "bump_threshold": self.cfg.bump_score_threshold,
            "status": status.value,
        }

        def reject(reason: str, score: float = 0.0) -> AlertDecision:
            reasons.append(reason)
            return AlertDecision(
                timestamp_ns=now_ns,
                track_id=obj.track_id,
                fired=False,
                phrase="",
                direction=obj.direction,
                semantic_type=obj.semantic_type,
                alert_score=score,
                threshold=self.cfg.score_threshold,
                reasons=list(reasons),
                snapshot=snapshot,
            )

        if not self.enabled:
            return reject("alerts_disabled")
        if obj.lifecycle_state not in {LifecycleState.CONFIRMED, LifecycleState.ALERTED}:
            return reject("not_confirmed")
        if self.cfg.pause_on_degraded and status in {
            PerceptionStatus.SEVERE_BLUR,
            PerceptionStatus.LENS_CONTAMINATION,
            PerceptionStatus.PERCEPTION_LIMITED,
        }:
            return reject("quality_pause")
        if (
            self.cfg.pause_on_degraded
            and status == PerceptionStatus.OCCLUDED
            and not is_bump_hazard(obj.semantic_type, obj.geometry_type, obj.object_state)
        ):
            return reject("quality_pause")
        if not geometry_valid:
            return reject("geometry_invalid")
        if obj.semantic_type in INFO_LAYER_SEMANTICS:
            return reject("info_layer")
        if obj.semantic_type in LOW_RISK_WHEN_NORMAL and (
            obj.object_state == ObjectState.NORMAL or obj.geometry_type == GeometryType.FLAT
        ):
            return reject("normal_or_flat")
        if int(obj.severity) >= 0 and int(obj.severity) < self.cfg.min_severity and obj.severity != Severity.UNKNOWN:
            return reject("severity_too_low")
        if obj.visibility_confidence < self.cfg.min_visibility:
            return reject("visibility_gate")
        if obj.effective_confidence < self.cfg.min_effective:
            return reject("effective_gate")
        if obj.path_relevance < self.cfg.min_path_relevance:
            return reject("off_corridor")
        if obj.direction == Direction.UNKNOWN:
            return reject("direction_unknown")

        vs = visual_score(
            obj.model_confidence,
            obj.visibility_confidence,
            obj.temporal_confidence,
            obj.geometry_consistency,
        )
        sev = severity_score(obj.severity, obj.geometry_type, obj.object_state)
        urg = urgency_from_ttc(obj.ttc_s)
        bump = is_bump_hazard(obj.semantic_type, obj.geometry_type, obj.object_state)
        if bump:
            # Bumps are the primary hazard class: never let a missing TTC starve them,
            # but they still have to clear every gate above and the (lower) bump threshold.
            urg = max(urg, 0.65)
        suppression = 1.0
        if obj.semantic_type == SemanticType.UNKNOWN_ANOMALY:
            suppression *= 0.55
        score = vs * sev * obj.path_relevance * urg * suppression
        threshold = self.cfg.bump_score_threshold if bump else self.cfg.score_threshold
        snapshot["visual_score"] = vs
        snapshot["severity_score"] = sev
        snapshot["urgency"] = urg
        snapshot["alert_score"] = score
        snapshot["threshold"] = threshold
        snapshot["bump_hazard"] = bump

        if score < threshold:
            return reject("below_threshold", score)

        prev = self.alerted_tracks.get(obj.track_id)
        if prev is not None:
            prev_sev = self.last_severity.get(obj.track_id, 0)
            jumped = int(obj.severity) - prev_sev >= self.cfg.realert_severity_jump and int(obj.severity) >= 2
            if not jumped:
                return reject("already_alerted", score)

        cooldown = int(self.cfg.global_cooldown_s * 1e9)
        if self.last_global_ns and now_ns - self.last_global_ns < cooldown:
            return reject("global_cooldown", score)
        cls_key = obj.semantic_type.value
        last_c = self.last_class_ns.get(cls_key, 0)
        if last_c and now_ns - last_c < int(self.cfg.class_cooldown_s * 1e9):
            return reject("class_cooldown", score)

        generic = obj.severity == Severity.UNKNOWN
        phrase = compose_phrase(
            obj.direction,
            obj.semantic_type,
            generic=generic,
            geometry=obj.geometry_type,
            state=obj.object_state,
            severity=obj.severity,
        )
        self.last_global_ns = now_ns
        self.last_class_ns[cls_key] = now_ns
        self.alerted_tracks[obj.track_id] = now_ns
        self.last_severity[obj.track_id] = int(obj.severity) if int(obj.severity) >= 0 else 0
        reasons.append("fired")
        return AlertDecision(
            timestamp_ns=now_ns,
            track_id=obj.track_id,
            fired=True,
            phrase=phrase,
            direction=obj.direction,
            semantic_type=obj.semantic_type,
            alert_score=score,
            threshold=threshold,
            reasons=reasons,
            snapshot=snapshot,
        )
