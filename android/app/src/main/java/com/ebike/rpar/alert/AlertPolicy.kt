package com.ebike.rpar.alert

import com.ebike.rpar.config.AlertConfig
import com.ebike.rpar.model.AlertDecision
import com.ebike.rpar.model.Direction
import com.ebike.rpar.model.EnumCopy
import com.ebike.rpar.model.GeometryType
import com.ebike.rpar.model.LifecycleState
import com.ebike.rpar.model.ObjectState
import com.ebike.rpar.model.PerceptionStatus
import com.ebike.rpar.model.SemanticType
import com.ebike.rpar.model.Severity
import com.ebike.rpar.model.TrackedRoadObject
import org.json.JSONObject

fun visualScore(calibrated: Double, visibility: Double, temporal: Double, geometry: Double): Double =
    maxOf(0.0, calibrated) * maxOf(0.0, visibility) * maxOf(0.0, temporal) * maxOf(0.0, geometry)

fun urgencyFromTtc(ttcS: Double?): Double = when {
    ttcS == null -> 0.35
    ttcS <= 0.8 -> 1.0
    ttcS <= 1.6 -> 0.85
    ttcS <= 2.8 -> 0.6
    else -> 0.25
}

fun severityScore(sev: Severity, geometry: GeometryType, state: ObjectState): Double {
    if (state == ObjectState.NORMAL || geometry == GeometryType.FLAT) return 0.02
    return when (sev) {
        Severity.NONE -> 0.05
        Severity.LIGHT -> 0.28
        Severity.MEDIUM -> 0.7
        Severity.HEAVY -> 1.0
        Severity.UNKNOWN -> 0.22
    }
}

fun composePhrase(
    direction: Direction,
    semantic: SemanticType,
    generic: Boolean = false,
    geometry: GeometryType = GeometryType.UNKNOWN,
    state: ObjectState = ObjectState.UNKNOWN,
): String {
    val heading = EnumCopy.DIRECTION_TTS[direction] ?: "前方"
    val kind = when {
        generic -> "路面异常"
        EnumCopy.isBumpHazard(semantic, geometry, state) -> EnumCopy.bumpKind(semantic, geometry, state)
        else -> EnumCopy.SEMANTIC_TTS[semantic] ?: "路面异常"
    }
    val phrase = heading + kind
    EnumCopy.ALERT_FORBIDDEN.forEach { bad ->
        require(!phrase.contains(bad)) { "forbidden advisory text: $bad" }
    }
    return phrase
}

class AlertPolicy(val cfg: AlertConfig, var enabled: Boolean = true) {
    var lastGlobalNs: Long = 0
    private val lastClassNs = HashMap<String, Long>()
    private val alertedTracks = HashMap<Int, Long>()
    private val lastSeverity = HashMap<Int, Int>()

    fun evaluate(
        obj: TrackedRoadObject,
        status: PerceptionStatus,
        nowNs: Long,
        geometryValid: Boolean,
    ): AlertDecision {
        val reasons = ArrayList<String>()
        val snapshot = JSONObject()
        snapshot.put("effective_confidence", obj.effectiveConfidence)
        snapshot.put("visibility_confidence", obj.visibilityConfidence)
        snapshot.put("temporal_confidence", obj.temporalConfidence)
        snapshot.put("geometry_consistency", obj.geometryConsistency)
        snapshot.put("path_relevance", obj.pathRelevance)
        snapshot.put("risk_score", obj.riskScore)
        snapshot.put("distance_m", obj.distanceM)
        snapshot.put("distance_valid", obj.distanceValid)
        snapshot.put("ttc_s", obj.ttcS)
        snapshot.put("lifecycle_state", obj.lifecycleState.wire)
        snapshot.put("severity", obj.severity.code)
        snapshot.put("object_state", obj.objectState.wire)
        snapshot.put("geometry_type", obj.geometryType.wire)
        snapshot.put("semantic_type", obj.semanticType.wire)
        snapshot.put("threshold", cfg.scoreThreshold)
        snapshot.put("bump_threshold", cfg.bumpScoreThreshold)
        snapshot.put("status", status.wire)

        fun reject(reason: String, score: Double = 0.0): AlertDecision {
            reasons += reason
            return AlertDecision(nowNs, obj.trackId, false, "", obj.direction, obj.semanticType, score, cfg.scoreThreshold, reasons.toList(), snapshot)
        }

        if (!enabled) return reject("alerts_disabled")
        if (obj.lifecycleState != LifecycleState.CONFIRMED && obj.lifecycleState != LifecycleState.ALERTED) return reject("not_confirmed")
        if (cfg.pauseOnDegraded && status in setOf(
                PerceptionStatus.SEVERE_BLUR,
                PerceptionStatus.LENS_CONTAMINATION,
                PerceptionStatus.PERCEPTION_LIMITED,
            )
        ) return reject("quality_pause")
        if (cfg.pauseOnDegraded && status == PerceptionStatus.OCCLUDED &&
            !EnumCopy.isBumpHazard(obj.semanticType, obj.geometryType, obj.objectState)
        ) return reject("quality_pause")
        if (!geometryValid) return reject("geometry_invalid")
        if (obj.semanticType in EnumCopy.INFO_LAYER) return reject("info_layer")
        if (obj.semanticType in EnumCopy.LOW_RISK_WHEN_NORMAL &&
            (obj.objectState == ObjectState.NORMAL || obj.geometryType == GeometryType.FLAT)
        ) return reject("normal_or_flat")
        if (obj.severity.code >= 0 && obj.severity.code < cfg.minSeverity && obj.severity != Severity.UNKNOWN) {
            return reject("severity_too_low")
        }
        if (obj.visibilityConfidence < cfg.minVisibility &&
            !EnumCopy.isBumpHazard(obj.semanticType, obj.geometryType, obj.objectState)
        ) return reject("visibility_gate")
        if (obj.effectiveConfidence < cfg.minEffective) {
            val bumpEarly = EnumCopy.isBumpHazard(obj.semanticType, obj.geometryType, obj.objectState) &&
                obj.modelConfidence >= 0.08
            if (!bumpEarly) return reject("effective_gate")
        }
        if (obj.pathRelevance < cfg.minPathRelevance) return reject("off_corridor")
        if (obj.direction == Direction.UNKNOWN) return reject("direction_unknown")

        val vs = visualScore(obj.modelConfidence, obj.visibilityConfidence, obj.temporalConfidence, obj.geometryConsistency)
        val sev = severityScore(obj.severity, obj.geometryType, obj.objectState)
        var urg = urgencyFromTtc(obj.ttcS)
        val bump = EnumCopy.isBumpHazard(obj.semanticType, obj.geometryType, obj.objectState)
        if (bump) urg = maxOf(urg, 0.65)
        var suppression = 1.0
        if (obj.semanticType == SemanticType.UNKNOWN_ANOMALY) suppression *= 0.55
        var score = vs * sev * obj.pathRelevance * urg * suppression
        if (bump) score = maxOf(score, obj.modelConfidence * obj.pathRelevance)
        val threshold = if (bump) cfg.bumpScoreThreshold else cfg.scoreThreshold
        snapshot.put("visual_score", vs)
        snapshot.put("severity_score", sev)
        snapshot.put("urgency", urg)
        snapshot.put("alert_score", score)
        snapshot.put("threshold", threshold)
        snapshot.put("bump_hazard", bump)
        if (score < threshold) return reject("below_threshold", score)

        val prev = alertedTracks[obj.trackId]
        if (prev != null) {
            val prevSev = lastSeverity[obj.trackId] ?: 0
            val jumped = obj.severity.code - prevSev >= cfg.realertSeverityJump && obj.severity.code >= 2
            if (!jumped) return reject("already_alerted", score)
        }
        val cooldown = (cfg.globalCooldownS * 1e9).toLong()
        if (lastGlobalNs != 0L && nowNs - lastGlobalNs < cooldown) return reject("global_cooldown", score)
        val cls = obj.semanticType.wire
        val lastC = lastClassNs[cls] ?: 0L
        if (lastC != 0L && nowNs - lastC < (cfg.classCooldownS * 1e9).toLong()) return reject("class_cooldown", score)

        val phrase = composePhrase(
            obj.direction,
            obj.semanticType,
            generic = obj.severity == Severity.UNKNOWN,
            geometry = obj.geometryType,
            state = obj.objectState,
        )
        lastGlobalNs = nowNs
        lastClassNs[cls] = nowNs
        alertedTracks[obj.trackId] = nowNs
        lastSeverity[obj.trackId] = if (obj.severity.code >= 0) obj.severity.code else 0
        reasons += "fired"
        return AlertDecision(nowNs, obj.trackId, true, phrase, obj.direction, obj.semanticType, score, threshold, reasons, snapshot)
    }
}
