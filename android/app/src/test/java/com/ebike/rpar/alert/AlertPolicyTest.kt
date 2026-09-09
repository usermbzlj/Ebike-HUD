package com.ebike.rpar.alert

import com.ebike.rpar.config.AlertConfig
import com.ebike.rpar.model.Direction
import com.ebike.rpar.model.GeometryType
import com.ebike.rpar.model.LifecycleState
import com.ebike.rpar.model.ObjectState
import com.ebike.rpar.model.PerceptionStatus
import com.ebike.rpar.model.SemanticType
import com.ebike.rpar.model.Severity
import com.ebike.rpar.model.TrackedRoadObject
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AlertPolicyTest {
    @Test
    fun phraseHasNoSteerAdvice() {
        val p = composePhrase(Direction.LEFT_FRONT, SemanticType.POTHOLE)
        assertTrue(p.startsWith("左前方"))
        assertTrue(p.contains("大坑"))
        assertFalse(p.contains("避让"))
        assertFalse(p.contains("转向"))
        assertFalse(p.contains("制动"))
    }

    @Test
    fun visualScoreIsMultiplicative() {
        val s = visualScore(0.8, 0.5, 1.0, 1.0)
        assertTrue(s < 0.8)
        assertTrue(s > 0.3)
    }

    @Test
    fun normalCoverDoesNotFire() {
        val policy = AlertPolicy(AlertConfig(), enabled = true)
        val obj = TrackedRoadObject(
            schemaVersion = "1.0",
            trackId = 1,
            timestampNs = 1_000L,
            lifecycleState = LifecycleState.CONFIRMED,
            semanticType = SemanticType.MANHOLE_COVER,
            geometryType = GeometryType.FLAT,
            objectState = ObjectState.NORMAL,
            severity = Severity.NONE,
            direction = Direction.CENTER_FRONT,
            distanceM = 12.0,
            distanceConfidence = 0.8,
            distanceValid = true,
            ttcS = 1.2,
            modelConfidence = 0.9,
            visibilityConfidence = 0.9,
            temporalConfidence = 0.9,
            geometryConsistency = 0.9,
            effectiveConfidence = 0.8,
            pathRelevance = 0.9,
            riskScore = 0.1,
            alertScore = 0.0,
            polygon = emptyList(),
            bbox = floatArrayOf(0f, 0f, 1f, 1f),
            maskRle = null,
            sourceFrameId = 1L,
            mountProfileId = "left_handlebar_v1",
            modelVersion = "heuristic-cv-0.1.0",
            visualStyle = "solid",
        )
        val d = policy.evaluate(obj, PerceptionStatus.NORMAL, 2_000L, true)
        assertFalse(d.fired)
        assertTrue(d.reasons.any { it.contains("normal") || it.contains("info") || it.contains("flat") || it.contains("severity") })
    }

    @Test
    fun puddleInfoLayerDoesNotFire() {
        val policy = AlertPolicy(AlertConfig(), enabled = true)
        val obj = TrackedRoadObject(
            schemaVersion = "1.0",
            trackId = 2,
            timestampNs = 1_000L,
            lifecycleState = LifecycleState.CONFIRMED,
            semanticType = SemanticType.PUDDLE,
            geometryType = GeometryType.FLAT,
            objectState = ObjectState.UNKNOWN,
            severity = Severity.NONE,
            direction = Direction.CENTER_FRONT,
            distanceM = 8.0,
            distanceConfidence = 0.8,
            distanceValid = true,
            ttcS = 1.2,
            modelConfidence = 0.9,
            visibilityConfidence = 0.9,
            temporalConfidence = 0.9,
            geometryConsistency = 0.9,
            effectiveConfidence = 0.8,
            pathRelevance = 0.9,
            riskScore = 0.2,
            alertScore = 0.0,
            polygon = emptyList(),
            bbox = floatArrayOf(0f, 0f, 1f, 1f),
            maskRle = null,
            sourceFrameId = 1L,
            mountProfileId = "left_handlebar_v1",
            modelVersion = "heuristic-cv-0.1.0",
            visualStyle = "solid",
        )
        val d = policy.evaluate(obj, PerceptionStatus.NORMAL, 2_000L, true)
        assertFalse(d.fired)
        assertTrue(d.reasons.any { it.contains("info") })
    }
}
