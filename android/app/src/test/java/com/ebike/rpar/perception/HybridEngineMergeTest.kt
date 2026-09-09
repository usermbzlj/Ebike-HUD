package com.ebike.rpar.perception

import com.ebike.rpar.model.GeometryType
import com.ebike.rpar.model.InferenceBackend
import com.ebike.rpar.model.ObjectState
import com.ebike.rpar.model.PerceptionResult
import com.ebike.rpar.model.RoadObservation
import com.ebike.rpar.model.SemanticType
import com.ebike.rpar.model.Severity
import com.ebike.rpar.model.VisibilityClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HybridEngineMergeTest {
    private fun obs(semantic: SemanticType, x0: Float, y0: Float, x1: Float, y1: Float): RoadObservation {
        val poly = listOf(x0 to y0, x1 to y0, x1 to y1, x0 to y1)
        return RoadObservation(
            timestampNs = 1,
            sourceFrameId = 0,
            semanticType = semantic,
            geometryType = GeometryType.CONCAVE,
            state = ObjectState.ABNORMAL,
            severity = Severity.MEDIUM,
            maskRle = null,
            polygon = poly,
            bbox = floatArrayOf(x0, y0, x1, y1),
            modelConfidence = 0.8,
            qualityAtMask = 0.9,
            visibility = VisibilityClass.CLEAR,
            calibratedConfidence = 0.8,
        )
    }

    @Test
    fun mergeKeepsHeuristicAndAddsFarSidecar() {
        val primary = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = listOf(0f to 10f, 20f to 10f, 20f to 20f, 0f to 20f),
            occludedPolygons = emptyList(),
            observations = listOf(obs(SemanticType.POTHOLE, 8f, 8f, 16f, 16f)),
            backend = InferenceBackend.HEURISTIC,
            latencyMs = 4.0,
            inputSizes = listOf(intArrayOf(96, 48)),
        )
        val sidecar = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = listOf(1f to 11f, 40f to 11f, 40f to 30f, 1f to 30f),
            occludedPolygons = listOf(listOf(2f to 2f, 6f to 2f, 6f to 6f, 2f to 6f)),
            observations = listOf(obs(SemanticType.UNKNOWN_ANOMALY, 12f, 14f, 18f, 22f)),
            backend = InferenceBackend.CPU,
            latencyMs = 9.0,
            inputSizes = listOf(intArrayOf(96, 48)),
        )
        val m = HybridEngine.merge(primary, sidecar, null)
        assertEquals(1f, m.roadPolygon[0].first)
        assertTrue(m.observations.any { it.semanticType == SemanticType.POTHOLE })
        assertTrue(m.observations.any { it.semanticType == SemanticType.UNKNOWN_ANOMALY })
        assertEquals(9.0, m.latencyMs, 1e-6)
    }

    @Test
    fun mergeDropsOffRoadAndOccluded() {
        val primary = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = listOf(0f to 0f, 10f to 0f, 10f to 10f, 0f to 10f),
            occludedPolygons = emptyList(),
            observations = listOf(
                obs(SemanticType.POTHOLE, 12f, 14f, 18f, 22f),
                obs(SemanticType.MANHOLE_COVER, 80f, 80f, 90f, 90f),
                obs(SemanticType.POTHOLE, 3f, 3f, 5f, 5f),
            ),
            backend = InferenceBackend.HEURISTIC,
            latencyMs = 4.0,
            inputSizes = listOf(intArrayOf(96, 48)),
        )
        val sidecar = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = listOf(1f to 11f, 40f to 11f, 40f to 30f, 1f to 30f),
            occludedPolygons = listOf(listOf(2f to 2f, 6f to 2f, 6f to 6f, 2f to 6f)),
            observations = emptyList(),
            backend = InferenceBackend.CPU,
            latencyMs = 9.0,
            inputSizes = listOf(intArrayOf(96, 48)),
        )
        val m = HybridEngine.merge(primary, sidecar, null)
        assertTrue(m.observations.any { it.semanticType == SemanticType.POTHOLE })
        assertTrue(m.observations.none { it.semanticType == SemanticType.MANHOLE_COVER })
        assertTrue(m.observations.none { it.bbox[0] == 3f })
    }

    @Test
    fun bumpSidecarReplacesHeuristicEvenIfEmpty() {
        val road = listOf(0f to 0f, 40f to 0f, 40f to 40f, 0f to 40f)
        val primary = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = road,
            occludedPolygons = emptyList(),
            observations = listOf(obs(SemanticType.POTHOLE, 8f, 8f, 16f, 16f)),
            backend = InferenceBackend.HEURISTIC,
            latencyMs = 4.0,
            inputSizes = listOf(intArrayOf(96, 48)),
        )
        val sidecar = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = emptyList(),
            occludedPolygons = emptyList(),
            observations = emptyList(),
            backend = InferenceBackend.CPU,
            latencyMs = 9.0,
            inputSizes = listOf(intArrayOf(320, 320)),
        )
        val m = HybridEngine.merge(primary, sidecar, null, replacesBump = true)
        assertTrue(m.observations.none { it.semanticType == SemanticType.POTHOLE })
        assertEquals(9.0, m.latencyMs, 1e-6)
    }

    @Test
    fun bumpSidecarKeepsInfoLayerAddsManhole() {
        val road = listOf(0f to 0f, 40f to 0f, 40f to 40f, 0f to 40f)
        val primary = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = road,
            occludedPolygons = emptyList(),
            observations = listOf(
                obs(SemanticType.POTHOLE, 8f, 8f, 16f, 16f),
                obs(SemanticType.PUDDLE, 20f, 20f, 28f, 28f),
            ),
            backend = InferenceBackend.HEURISTIC,
            latencyMs = 4.0,
            inputSizes = listOf(intArrayOf(96, 48)),
        )
        val sidecar = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = emptyList(),
            occludedPolygons = emptyList(),
            observations = listOf(obs(SemanticType.MANHOLE_COVER, 10f, 12f, 18f, 22f)),
            backend = InferenceBackend.CPU,
            latencyMs = 11.0,
            inputSizes = listOf(intArrayOf(320, 320)),
        )
        val m = HybridEngine.merge(primary, sidecar, null, replacesBump = true)
        assertTrue(m.observations.none { it.semanticType == SemanticType.POTHOLE })
        assertTrue(m.observations.any { it.semanticType == SemanticType.PUDDLE })
        assertTrue(m.observations.any { it.semanticType == SemanticType.MANHOLE_COVER })
        assertEquals(InferenceBackend.CPU, m.backend)
    }

    @Test
    fun bumpSidecarKeepsOffRoadPitAndDropsOccluded() {
        val road = listOf(0f to 30f, 20f to 30f, 20f to 40f, 0f to 40f)
        val occ = listOf(listOf(50f to 50f, 80f to 50f, 80f to 80f, 50f to 80f))
        val primary = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = road,
            occludedPolygons = occ,
            observations = listOf(obs(SemanticType.POTHOLE, 8f, 8f, 16f, 16f)),
            backend = InferenceBackend.HEURISTIC,
            latencyMs = 4.0,
            inputSizes = listOf(intArrayOf(96, 48)),
        )
        val sidecar = PerceptionResult(
            timestampNs = 1,
            sourceFrameId = 0,
            roadPolygon = emptyList(),
            occludedPolygons = emptyList(),
            observations = listOf(
                obs(SemanticType.POTHOLE, 4f, 4f, 12f, 12f),
                obs(SemanticType.SPEED_BUMP, 55f, 55f, 70f, 70f),
            ),
            backend = InferenceBackend.CPU,
            latencyMs = 11.0,
            inputSizes = listOf(intArrayOf(640, 640)),
        )
        val m = HybridEngine.merge(primary, sidecar, null, replacesBump = true)
        assertTrue(m.observations.any { it.semanticType == SemanticType.POTHOLE && it.bbox[0] == 4f })
        assertTrue(m.observations.none { it.semanticType == SemanticType.SPEED_BUMP })
    }
}
