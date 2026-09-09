package com.ebike.rpar.app

import com.ebike.rpar.alert.AlertPolicy
import com.ebike.rpar.alert.visualScore
import com.ebike.rpar.config.RparConfig
import com.ebike.rpar.geometry.GeometryEngine
import com.ebike.rpar.geometry.Transforms
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
import com.ebike.rpar.model.SemanticType
import com.ebike.rpar.model.SynchronizedFrame
import com.ebike.rpar.model.TrackedRoadObject
import com.ebike.rpar.model.UiMode
import com.ebike.rpar.model.estimateImpactScore
import com.ebike.rpar.perception.DualScaleRoi
import com.ebike.rpar.perception.HeuristicEngine
import com.ebike.rpar.perception.NoOpEngine
import com.ebike.rpar.perception.PerceptionEngine
import com.ebike.rpar.quality.GrayImage
import com.ebike.rpar.quality.QualityScheduler
import com.ebike.rpar.quality.allowNewObservations
import com.ebike.rpar.quality.evaluateFrame
import com.ebike.rpar.quality.occlusionCoverRatio
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
    var lastObservations: List<com.ebike.rpar.model.RoadObservation> = emptyList()
    var lastRoadPolygon: List<Pair<Float, Float>> = emptyList()
    var lastOccPolygons: List<List<Pair<Float, Float>>> = emptyList()
    var didInfer: Boolean = false
    var inferCount = 0
    var frameCount = 0
    var droppedInfer = 0
    var runMode: RunMode = RunMode.REALTIME_PERCEPTION
    var thermalC: Double? = null
    var thermalReason: String? = null
    var skipFarRoi: Boolean = false
    var recSeconds: Double = 0.0
    var nightPalette: Boolean = false
    var researchHeatmap: Boolean = true
    var strokeScale: Float = cfg.render.strokeScale
    var fontScale: Float = cfg.render.fontScale
    var overlayAlpha: Float = cfg.render.overlayAlpha
    var showInfoLayer: Boolean = cfg.render.showInfoLayer
    private val latencies = ArrayDeque<Double>()
    private val inferTimes = ArrayDeque<Long>()
    private var lastInferNs = 0L
    private var inferPeriodNs = (1e9 / maxOf(cfg.runtime.inferFps, 1.0)).toLong()
    private val baseInferPeriodNs = inferPeriodNs
    private var lastStatus: PerceptionStatus = PerceptionStatus.NORMAL
    private var lastInputFar: IntArray = cfg.model.inputFar
    private var lastInputNear: IntArray = cfg.model.inputNear

    private fun applyThermal() {
        val prev = thermalReason
        val t = thermalC
        if (t == null || t < 42.0) {
            inferPeriodNs = baseInferPeriodNs
            skipFarRoi = false
            thermalReason = null
        } else if (t >= 45.0) {
            skipFarRoi = true
            inferPeriodNs = (1e9 / maxOf(cfg.runtime.thermalMinInferFps, 1.0)).toLong()
            thermalReason = "THERMAL_DROP_INFER_HZ"
        } else if (t >= 43.0) {
            skipFarRoi = true
            inferPeriodNs = baseInferPeriodNs
            thermalReason = "THERMAL_DROP_FAR_ROI"
        } else {
            skipFarRoi = false
            val slowed = maxOf(cfg.runtime.thermalMinInferFps, cfg.runtime.inferFps * 0.7)
            inferPeriodNs = (1e9 / maxOf(slowed, 1.0)).toLong()
            thermalReason = "THERMAL_DROP_INFER_HZ"
        }
        val he = engine as? HeuristicEngine
        if (he != null) he.skipFarRoi = skipFarRoi
        if (prev != thermalReason && thermalReason != null) {
            lastThermalEvent = thermalReason
        }
    }
    var lastThermalEvent: String? = null
    var lastStatusEvent: String? = null

    fun reset() {
        tracker.reset()
        scheduler = QualityScheduler(cfg.quality)
        lastNs = null
        badStreakS = 0.0
        status = PerceptionStatus.NORMAL
        lastObservations = emptyList()
        lastRoadPolygon = emptyList()
        lastOccPolygons = emptyList()
        didInfer = false
        thermalReason = null
        skipFarRoi = false
        inferPeriodNs = baseInferPeriodNs
        lastStatus = PerceptionStatus.NORMAL
    }

    fun setSafe(safe: Boolean) {
        if (safe) engine = NoOpEngine()
    }

    suspend fun step(frame: SynchronizedFrame, uiMode: UiMode = UiMode.RIDING): PerceptionView {
        val t0 = frame.meta.sensorTimestampNs
        val dt = if (lastNs == null) 1.0 / 60.0 else maxOf(1e-3, (t0 - lastNs!!) / 1e9)
        lastNs = t0
        frameCount++
        applyThermal()

        val gray = when {
            frame.yuv != null -> GrayImage.fromYuv(frame.yuv, 480)
            frame.bitmap != null -> GrayImage.fromBitmap(frame.bitmap, 480)
            else -> GrayImage(64, 36, IntArray(64 * 36) { 80 })
        }
        val hl = geometry.mount.let { if (it.headlightValid) it.headlightMean else null }
        val qmap = evaluateFrame(gray, frame.meta.width, frame.meta.height, cfg.quality, hl)
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
        if (lastStatus != status) {
            lastStatusEvent = when (status) {
                PerceptionStatus.PERCEPTION_LIMITED -> "PERCEPTION_LIMITED"
                PerceptionStatus.LENS_CONTAMINATION -> "LENS_CONTAMINATION"
                PerceptionStatus.OCCLUDED -> "OCCLUDED"
                PerceptionStatus.SEVERE_BLUR -> "SEVERE_BLUR"
                PerceptionStatus.THERMAL_THROTTLE -> thermalReason ?: "THERMAL_THROTTLE"
                else -> null
            }
            lastStatus = status
        }
        val occRatio = occlusionCoverRatio(lastOccPolygons, frame.meta.width, frame.meta.height)
        var allowNew = allowNewObservations(status, sel.q.globalQuality.usable, sel.fresh, occRatio)
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
            dual = result.dualScale && !skipFarRoi
            if (result.inputSizes.isNotEmpty()) {
                lastInputFar = result.inputSizes.first()
                lastInputNear = result.inputSizes.last()
            }
            lastInferNs = t0
            inferCount++
            inferTimes.addLast(t0)
            while (inferTimes.size > 40) inferTimes.removeFirst()
            lastObservations = observations
            lastRoadPolygon = result.roadPolygon
            lastOccPolygons = result.occludedPolygons
            didInfer = true
            val cover = occlusionCoverRatio(lastOccPolygons, frame.meta.width, frame.meta.height)
            if (cover >= 0.08) {
                status = PerceptionStatus.OCCLUDED
            }
            allowNew = allowNewObservations(status, sel.q.globalQuality.usable, true, cover)
        } else {
            droppedInfer++
            didInfer = false
            observations = lastObservations
        }
        val qualityOk = sel.q.globalQuality.usable &&
            status != PerceptionStatus.SEVERE_BLUR &&
            status != PerceptionStatus.LENS_CONTAMINATION &&
            status != PerceptionStatus.PERCEPTION_LIMITED
        val tracks = tracker.update(
            observations, t0, allowNewHighConf = allowNew, qualityOk = qualityOk, dtS = dt,
            cameraYawRate = frame.angularVelocity?.getOrNull(2) ?: 0.0,
            frameW = frame.meta.width.toDouble(),
        )
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
            val depthConf = if (dvalid) dconf else 0.0
            val impact = estimateImpactScore(frame.linearAccel, speed, if (showDist) dist else null)
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
                maskRle = tr.maskRle,
                sourceFrameId = tr.sourceFrameId,
                mountProfileId = geometry.mount.profileId,
                modelVersion = modelVersion,
                visualStyle = if (tr.state == LifecycleState.CANDIDATE) "dashed" else "solid",
                roadXyM = tr.roadXy?.let { it[0] to it[1] },
                depthConfidence = depthConf,
                impactScore = impact,
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
        val e2e = (if (doInfer) sel.ageMs else 0.0) + inferMs
        latencies.addLast(e2e)
        while (latencies.size > 120) latencies.removeFirst()
        val p95 = percentile(latencies, 95.0)
        val p50 = percentile(latencies, 50.0)
        val primitives = primitives(trackedObjs, uiMode, sel.q, frame.meta.width, frame.meta.height, frame.angularVelocity?.getOrNull(2) ?: 0.0, p95)
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
            latencyP50Ms = p50,
            droppedInfer = droppedInfer,
            inputFar = lastInputFar,
            inputNear = lastInputNear,
            roadPolygon = lastRoadPolygon,
            occludedPolygons = lastOccPolygons,
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

    private fun primitives(objs: List<TrackedRoadObject>, uiMode: UiMode, qmap: com.ebike.rpar.model.FrameQualityMap, frameW: Int, frameH: Int, yawRate: Double = 0.0, latencyMs: Double = 0.0): List<RenderPrimitive> {
        val prims = ArrayList<RenderPrimitive>()
        if (lastRoadPolygon.size >= 3) {
            val a = if (uiMode == UiMode.RIDING) 0.16f else 0.28f
            prims += RenderPrimitive(
                -6, lastRoadPolygon, floatArrayOf(0.12f, 0.92f, 0.38f, a),
                dashed = false, thickness = 2f, label = null, labelPriority = 85, fade = 0.32f, kind = "road",
            )
        }
        lastOccPolygons.forEachIndexed { i, poly ->
            if (poly.size < 3) return@forEachIndexed
            prims += RenderPrimitive(
                -7 - i, poly, floatArrayOf(0.55f, 0.58f, 0.68f, 0.42f),
                dashed = false, thickness = 2f,
                label = if (uiMode == UiMode.RESEARCH) "vehicle" else null,
                labelPriority = 86, fade = 0.7f, kind = "occlusion",
            )
        }
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
        if (uiMode == UiMode.RESEARCH) {
            val fw = cfg.model.inputFar[0]; val fh = cfg.model.inputFar[1]
            val nw = cfg.model.inputNear[0]; val nh = cfg.model.inputNear[1]
            prims += RenderPrimitive(
                -3,
                DualScaleRoi.farPolygon(frameW, frameH),
                floatArrayOf(0.35f, 0.9f, 0.55f, 0.12f),
                dashed = true, thickness = 1f, label = "far ${fw}x$fh", labelPriority = 90, fade = 0.4f, kind = "roi",
            )
            prims += RenderPrimitive(
                -4,
                DualScaleRoi.nearPolygon(frameW, frameH),
                floatArrayOf(0.9f, 0.7f, 0.2f, 0.10f),
                dashed = true, thickness = 1f, label = "near ${nw}x$nh", labelPriority = 91, fade = 0.4f, kind = "roi",
            )
            if (researchHeatmap) {
                for (t in qmap.tiles) {
                    if (t.visibility.wire == "clear") continue
                    val a = 0.16f
                    val col = when (t.visibility.wire) {
                        "blur" -> floatArrayOf(0.16f, 0.35f, 0.82f, a)
                        "glare" -> floatArrayOf(1f, 0.86f, 0.16f, a)
                        "underexposed" -> floatArrayOf(0.16f, 0.31f, 0.7f, a)
                        "overexposed" -> floatArrayOf(0.94f, 0.94f, 0.94f, a)
                        else -> floatArrayOf(0.47f, 0.47f, 0.47f, a)
                    }
                    prims += RenderPrimitive(
                        -10,
                        listOf(t.x0.toFloat() to t.y0.toFloat(), t.x1.toFloat() to t.y0.toFloat(), t.x1.toFloat() to t.y1.toFloat(), t.x0.toFloat() to t.y1.toFloat()),
                        col, dashed = false, thickness = 1f, label = null, labelPriority = 95, fade = a, kind = "heatmap",
                    )
                }
            }
        }
        var labeled = 0
        val stroke = strokeScale
        val overlayA = overlayAlpha.coerceIn(0.15f, 1f)
        for (obj in objs) {
            if (obj.lifecycleState == LifecycleState.EXPIRED) continue
            val info = obj.semanticType in EnumCopy.INFO_LAYER
            if (info && !showInfoLayer) continue
            if (!info && obj.lifecycleState == LifecycleState.TRACKED) {
                if (!EnumCopy.isBumpHazard(obj.semanticType, obj.geometryType, obj.objectState)) continue
            }
            if (!info && obj.lifecycleState == LifecycleState.CANDIDATE) continue
            val fade0 = when (obj.lifecycleState) {
                LifecycleState.PASSED -> 0.35f
                LifecycleState.CANDIDATE -> cfg.render.candidateAlpha
                else -> cfg.render.confirmedAlpha
            }
            val fade = (fade0 * overlayA).coerceIn(0.05f, 1f)
            val high = !info && obj.lifecycleState in setOf(LifecycleState.CONFIRMED, LifecycleState.ALERTED) && obj.riskScore > 0.4
            var color = when {
                info && obj.semanticType == SemanticType.PUDDLE -> floatArrayOf(0.35f, 0.55f, 0.88f, fade)
                info -> floatArrayOf(0.72f, 0.66f, 0.42f, fade)
                EnumCopy.isBumpHazard(obj.semanticType, obj.geometryType, obj.objectState) ->
                    floatArrayOf(0.95f, 0.28f, 0.16f, fade)
                obj.geometryType.wire == "concave" -> floatArrayOf(0.15f, 0.82f, 0.78f, fade)
                obj.geometryType.wire == "rough" -> floatArrayOf(0.95f, 0.78f, 0.25f, fade)
                else -> floatArrayOf(0.55f, 0.85f, 0.95f, fade)
            }
            if (high) color = floatArrayOf(color[0], color[1], color[2], minOf(1f, fade + 0.1f))
            val dashed = obj.lifecycleState == LifecycleState.CANDIDATE || obj.lifecycleState == LifecycleState.TRACKED
            var label: String? = null
            val allow = uiMode == UiMode.RESEARCH || labeled < cfg.render.ridingMaxLabels
            if (allow && obj.lifecycleState in setOf(LifecycleState.CONFIRMED, LifecycleState.ALERTED) && !(info && uiMode == UiMode.RIDING)) {
                val distTxt = geometry.displayDistance(obj.distanceM, obj.distanceValid, obj.distanceConfidence)
                label = if (uiMode == UiMode.RIDING) {
                    val dirCn = EnumCopy.DIRECTION_TTS[obj.direction] ?: ""
                    if (EnumCopy.isBumpHazard(obj.semanticType, obj.geometryType, obj.objectState)) {
                        val kind = EnumCopy.bumpKind(obj.semanticType, obj.geometryType, obj.objectState)
                        if (distTxt != null) "$dirCn$kind · $distTxt" else "$dirCn$kind"
                    } else {
                        "$dirCn ${distTxt ?: ""}".trim()
                    }
                } else {
                    "ID ${obj.trackId} · ${obj.semanticType.wire} · ${"%.2f".format(obj.modelConfidence)}" +
                        (if (distTxt != null) " · $distTxt" else "")
                }
                labeled++
            }
            prims += RenderPrimitive(
                obj.trackId, Transforms.displayCompensate(obj.polygon, yawRate, latencyMs, frameW), color, dashed,
                thickness = (if (high) 3.2f else 2f) * stroke,
                label = label,
                labelPriority = obj.labelRank ?: 50,
                fade = fade,
                kind = if (info) "info" else "anomaly",
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
