package com.ebike.rpar.app

import com.ebike.rpar.alert.AlertPolicy
import com.ebike.rpar.alert.visualScore
import com.ebike.rpar.config.RparConfig
import com.ebike.rpar.geometry.GeometryEngine
import com.ebike.rpar.model.AlertDecision
import com.ebike.rpar.model.Direction
import com.ebike.rpar.model.EnumCopy
import com.ebike.rpar.model.InferenceBackend
import com.ebike.rpar.model.LifecycleState
import com.ebike.rpar.model.PerceptionStatus
import com.ebike.rpar.model.PerceptionView
import com.ebike.rpar.model.RenderPrimitive
import com.ebike.rpar.model.RunMode
import com.ebike.rpar.model.SCHEMA_VERSION
import com.ebike.rpar.model.SynchronizedFrame
import com.ebike.rpar.model.TrackedRoadObject
import com.ebike.rpar.model.UiMode
import com.ebike.rpar.perception.NoOpEngine
import com.ebike.rpar.perception.PerceptionEngine
import com.ebike.rpar.quality.GrayImage
import com.ebike.rpar.quality.QualityScheduler
import com.ebike.rpar.quality.evaluateFrame
import com.ebike.rpar.quality.perceptionStatus
import com.ebike.rpar.tracking.TrackEngine
import com.ebike.rpar.tracking.TrackInternal
import kotlinx.coroutines.runBlocking
import java.util.ArrayDeque
import kotlin.math.min

class RealtimePipeline(
    val cfg: RparConfig,
    var engine: PerceptionEngine,
    val geometry: GeometryEngine,
    var modelVersion: String = cfg.model.packageId,
) {
    val tracker = TrackEngine(cfg.tracking)
    val alerts = AlertPolicy(cfg.alert)
    var scheduler = QualityScheduler(cfg.quality)
    private var lastNs: Long? = null
    var badStreakS = 0.0
    var status: PerceptionStatus = PerceptionStatus.NORMAL
    var lastView: PerceptionView? = null
    var inferCount = 0
    var frameCount = 0
    var droppedInfer = 0
    var runMode: RunMode = RunMode.REALTIME_PERCEPTION
    var thermalC: Double? = null
    var recSeconds: Double = 0.0
    private val latencies = ArrayDeque<Double>()
    private val inferTimes = ArrayDeque<Long>()
    private var lastInferNs = 0L
    private var inferPeriodNs = (1e9 / maxOf(cfg.runtime.inferFps, 1.0)).toLong()
    private val baseInferPeriodNs = inferPeriodNs

    fun reset() {
        tracker.reset()
        scheduler = QualityScheduler(cfg.quality)
        lastNs = null
        badStreakS = 0.0
        status = PerceptionStatus.NORMAL
    }

    fun setSafe(safe: Boolean) {
        if (safe) engine = NoOpEngine()
    }

    suspend fun step(frame: SynchronizedFrame, uiMode: UiMode = UiMode.RIDING): PerceptionView {
        val t0 = frame.meta.sensorTimestampNs
        val dt = if (lastNs == null) 1.0 / 60.0 else maxOf(1e-3, (t0 - lastNs!!) / 1e9)
        lastNs = t0
        frameCount++
        if (thermalC != null && thermalC!! >= 42.0) {
            inferPeriodNs = (1e9 / maxOf(cfg.runtime.thermalMinInferFps, 1.0)).toLong()
        } else {
            inferPeriodNs = baseInferPeriodNs
        }

        val gray = when {
            frame.yuv != null -> GrayImage.fromYuv(frame.yuv, 480)
            frame.bitmap != null -> GrayImage.fromBitmap(frame.bitmap, 480)
            else -> GrayImage(64, 36, IntArray(64 * 36) { 80 })
        }
        val qmap = evaluateFrame(gray, frame.meta.width, frame.meta.height, cfg.quality)
        scheduler.push(frame, qmap)
        val sel = scheduler.select(t0)
        if (sel.q.globalQuality.usable) badStreakS = 0.0 else badStreakS += dt
        status = when (runMode) {
            RunMode.SAFE_MODE -> PerceptionStatus.SAFE_MODE
            else -> perceptionStatus(sel.q, badStreakS)
        }
        if (thermalC != null && thermalC!! >= 42.0 && status == PerceptionStatus.NORMAL) {
            status = PerceptionStatus.THERMAL_THROTTLE
        }
        val allowNew = sel.fresh && sel.q.globalQuality.usable && status !in setOf(
            PerceptionStatus.SEVERE_BLUR, PerceptionStatus.PERCEPTION_LIMITED, PerceptionStatus.LENS_CONTAMINATION,
        )
        val engineOn = runMode != RunMode.CAPTURE_ONLY && runMode != RunMode.SAFE_MODE
        val doInfer = engineOn && sel.fresh && (t0 - lastInferNs) >= inferPeriodNs
        var observations = emptyList<com.ebike.rpar.model.RoadObservation>()
        var backend = InferenceBackend.HEURISTIC
        var inferMs = 0.0
        var dual = true
        if (doInfer) {
            val result = engine.infer(sel.frame, sel.q)
            observations = result.observations
            backend = result.backend
            inferMs = result.latencyMs
            dual = result.dualScale
            lastInferNs = t0
            inferCount++
            inferTimes.addLast(t0)
            while (inferTimes.size > 40) inferTimes.removeFirst()
        } else droppedInfer++

        val tracks = tracker.update(observations, t0, allowNewHighConf = allowNew, qualityOk = allowNew, dtS = dt)
        var speed = frame.speedMps ?: frame.location?.speedMps
        val trackedObjs = ArrayList<TrackedRoadObject>()
        val fired = ArrayList<AlertDecision>()
        for (tr in tracks) {
            val (dist, dconf, dvalid) = geometry.distanceForTrack(tr, t0)
            val prev = tr.distanceHist.lastOrNull()?.second
            if (dist != null) tr.distanceHist += t0 to dist
            if (GeometryEngine.shouldMarkPassed(tr, dist, prev, cfg.geometry.nearBoundaryM)) {
                tracker.markPassed(tr.trackId, t0, "near_boundary")
                tr.state = LifecycleState.PASSED
            }
            val ttc = geometry.ttc(tr, speed, t0)
            var direction = geometry.directionFor(tr)
            val relevance = geometry.pathRelevance(tr)
            val temporal = temporal(tr, t0)
            val gcons = if (tr.roadXy == null) 0.35 else (0.4 + 0.6 * dconf).coerceIn(0.0, 1.0)
            val vis = tr.qualityAtMask.coerceIn(0.0, 1.0)
            val eff = visualScore(tr.modelConfidence, vis, temporal, gcons)
            if (!dvalid && direction == Direction.UNKNOWN) direction = Direction.CENTER_FRONT
            val sevTable = mapOf(0 to 0.05, 1 to 0.3, 2 to 0.7, 3 to 1.0, -1 to 0.22)
            val risk = eff * (sevTable[tr.severity.code] ?: 0.22) * relevance
            val showDist = dvalid && geometry.valid
            val obj = TrackedRoadObject(
                schemaVersion = SCHEMA_VERSION,
                trackId = tr.trackId,
                timestampNs = t0,
                lifecycleState = tr.state,
                semanticType = tr.semantic,
                geometryType = tr.geometry,
                objectState = tr.objectState,
                severity = tr.severity,
                direction = direction,
                distanceM = if (showDist) dist else null,
                distanceConfidence = dconf,
                distanceValid = showDist,
                ttcS = ttc,
                modelConfidence = tr.modelConfidence,
                visibilityConfidence = vis,
                temporalConfidence = temporal,
                geometryConsistency = gcons,
                effectiveConfidence = eff,
                pathRelevance = relevance,
                riskScore = risk,
                alertScore = 0.0,
                polygon = tr.polygon,
                bbox = tr.bbox,
                maskRle = null,
                sourceFrameId = tr.sourceFrameId,
                mountProfileId = geometry.mount.profileId,
                modelVersion = modelVersion,
                visualStyle = if (tr.state == LifecycleState.CANDIDATE) "dashed" else "solid",
                roadXyM = tr.roadXy?.let { it[0] to it[1] },
            )
            val decision = alerts.evaluate(obj, status, t0, geometry.valid)
            obj.alertScore = decision.alertScore
            if (decision.fired) {
                tr.state = LifecycleState.ALERTED
                tr.alerted = true
                obj.lifecycleState = LifecycleState.ALERTED
                fired += decision
            }
            trackedObjs += obj
        }
        trackedObjs.sortByDescending { it.riskScore }
        trackedObjs.forEachIndexed { i, o -> o.labelRank = i }
        val primitives = primitives(trackedObjs, uiMode, sel.q)
        val e2e = (if (doInfer) sel.ageMs else 0.0) + inferMs
        latencies.addLast(e2e)
        while (latencies.size > 120) latencies.removeFirst()
        val p95 = percentile(latencies, 95.0)
        var inferFps = 0.0
        if (inferTimes.size >= 2) {
            val span = (inferTimes.last() - inferTimes.first()) / 1e9
            inferFps = (inferTimes.size - 1) / maxOf(span, 1e-3)
        }
        val view = PerceptionView(
            timestampNs = t0,
            status = status,
            statusCopy = EnumCopy.RIDING_STATUS[status] ?: "感知受限",
            tracks = trackedObjs,
            primitives = primitives,
            alerts = fired,
            quality = sel.q,
            speedKmh = speed?.times(3.6),
            backend = backend,
            modelVersion = modelVersion,
            inferFps = inferFps,
            arFps = 1.0 / dt,
            latencyP95Ms = p95,
            queueDepth = scheduler.size,
            thermalC = thermalC,
            blur = sel.q.globalQuality.motionBlur,
            glare = sel.q.globalQuality.glare,
            recSeconds = recSeconds,
            dualScale = dual,
        )
        lastView = view
        return view
    }

    fun stepBlocking(frame: SynchronizedFrame, uiMode: UiMode) = runBlocking { step(frame, uiMode) }

    private fun temporal(tr: TrackInternal, nowNs: Long): Double {
        val age = (nowNs - tr.createdNs) / 1e9
        val stab = min(1.0, tr.hits / 6.0) * min(1.0, age / maxOf(cfg.tracking.confirmWindowS, 1e-3))
        val missPen = Math.pow(0.85, tr.misses.toDouble())
        return (stab * missPen).coerceIn(0.05, 1.0)
    }

    private fun primitives(objs: List<TrackedRoadObject>, uiMode: UiMode, qmap: com.ebike.rpar.model.FrameQualityMap): List<RenderPrimitive> {
        val prims = ArrayList<RenderPrimitive>()
        if (qmap.occupancyOccludedRatio > 0.25) {
            prims += RenderPrimitive(
                trackId = -1, polygon = emptyList(),
                colorRgba = floatArrayOf(0.45f, 0.5f, 0.58f, 0.22f),
                dashed = true, thickness = 1f, label = null, labelPriority = 99, fade = 1f, kind = "occlusion",
            )
        }
        val corridor = geometry.projectCorridorPixels()
        if (uiMode == UiMode.RESEARCH && corridor.isNotEmpty()) {
            prims += RenderPrimitive(
                -2, corridor, floatArrayOf(0.2f, 0.75f, 0.8f, 0.18f),
                dashed = true, thickness = 2f, label = null, labelPriority = 80, fade = 1f, kind = "corridor",
            )
        }
        var labeled = 0
        for (obj in objs) {
            if (obj.lifecycleState == LifecycleState.EXPIRED) continue
            val fade = when (obj.lifecycleState) {
                LifecycleState.PASSED -> 0.35f
                LifecycleState.CANDIDATE -> cfg.render.candidateAlpha
                else -> cfg.render.confirmedAlpha
            }
            val high = obj.lifecycleState in setOf(LifecycleState.CONFIRMED, LifecycleState.ALERTED) && obj.riskScore > 0.4
            var color = when (obj.geometryType.wire) {
                "concave" -> floatArrayOf(0.15f, 0.82f, 0.78f, fade)
                "rough" -> floatArrayOf(0.95f, 0.78f, 0.25f, fade)
                else -> floatArrayOf(0.55f, 0.85f, 0.95f, fade)
            }
            if (high) color = floatArrayOf(color[0], color[1], color[2], minOf(1f, fade + 0.1f))
            val dashed = obj.lifecycleState == LifecycleState.CANDIDATE || obj.lifecycleState == LifecycleState.TRACKED
            var label: String? = null
            val allow = uiMode == UiMode.RESEARCH || labeled < cfg.render.ridingMaxLabels
            if (allow && obj.lifecycleState != LifecycleState.CANDIDATE) {
                val distTxt = geometry.displayDistance(obj.distanceM, obj.distanceValid, obj.distanceConfidence)
                label = if (uiMode == UiMode.RIDING) {
                    val dirCn = EnumCopy.DIRECTION_TTS[obj.direction] ?: ""
                    "$dirCn ${distTxt ?: ""}".trim()
                } else {
                    "ID ${obj.trackId} · ${obj.semanticType.wire} · ${"%.2f".format(obj.modelConfidence)}" +
                        (if (distTxt != null) " · $distTxt" else "")
                }
                labeled++
            }
            prims += RenderPrimitive(
                obj.trackId, obj.polygon, color, dashed,
                thickness = if (high) 3.2f else 2f,
                label = label,
                labelPriority = obj.labelRank ?: 50,
                fade = fade,
                kind = "anomaly",
            )
        }
        return prims
    }

    private fun percentile(vals: ArrayDeque<Double>, p: Double): Double {
        if (vals.isEmpty()) return 0.0
        val s = vals.sorted()
        val i = ((p / 100.0) * (s.size - 1)).toInt().coerceIn(0, s.lastIndex)
        return s[i]
    }
}
