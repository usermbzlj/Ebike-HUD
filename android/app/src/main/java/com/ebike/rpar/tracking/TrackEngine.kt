package com.ebike.rpar.tracking

import com.ebike.rpar.config.TrackingConfig
import com.ebike.rpar.model.GeometryType
import com.ebike.rpar.model.LifecycleState
import com.ebike.rpar.model.ObjectState
import com.ebike.rpar.model.RoadObservation
import com.ebike.rpar.model.SemanticType
import com.ebike.rpar.model.Severity
import com.ebike.rpar.model.VisibilityClass
import com.ebike.rpar.model.bboxIou
import com.ebike.rpar.model.bboxOf
import com.ebike.rpar.model.groundContact
import com.ebike.rpar.model.polygonCentroid
import kotlin.math.hypot

class TrackInternal(
    val trackId: Int,
    var hits: Int,
    var misses: Int,
    val createdNs: Long,
    var lastNs: Long,
    var confirmedNs: Long? = null,
    var alerted: Boolean = false,
    var semantic: SemanticType = SemanticType.UNKNOWN_ANOMALY,
    var geometry: GeometryType = GeometryType.UNKNOWN,
    var objectState: ObjectState = ObjectState.UNKNOWN,
    var severity: Severity = Severity.UNKNOWN,
    var polygon: List<Pair<Float, Float>> = emptyList(),
    var bbox: FloatArray = floatArrayOf(0f, 0f, 0f, 0f),
    var modelConfidence: Double = 0.0,
    var qualityAtMask: Double = 0.0,
    var visibility: VisibilityClass = VisibilityClass.UNKNOWN,
    var state: LifecycleState = LifecycleState.CANDIDATE,
    var expireReason: String = "",
    var mean: DoubleArray = DoubleArray(6),
    var cov: DoubleArray = DoubleArray(36) { i -> if (i % 7 == 0) 40.0 else 0.0 },
    var roadXy: DoubleArray? = null,
    val distanceHist: MutableList<Pair<Long, Double>> = ArrayList(),
    var fade: Double = 1.0,
    var sourceFrameId: Long = 0,
    var holdUntilNs: Long? = null,
    var maskRle: com.ebike.rpar.model.MaskRle? = null,
)

class KalmanImage(private val q: Double, private val r: Double) {
    fun predict(mean: DoubleArray, cov: DoubleArray, dt: Double): Pair<DoubleArray, DoubleArray> {
        val f = ident(6)
        f[0 * 6 + 2] = dt
        f[1 * 6 + 3] = dt
        val qn = DoubleArray(36)
        val diag = doubleArrayOf(q, q, q * 4, q * 4, q * 0.5, q * 0.5)
        val s = maxOf(dt, 1e-3)
        for (i in 0..5) qn[i * 6 + i] = diag[i] * s
        val meanP = matVec(f, mean, 6)
        val covP = add(mul(mul(f, cov, 6), transpose(f, 6), 6), qn, 6)
        return meanP to covP
    }

    fun update(mean: DoubleArray, cov: DoubleArray, z: DoubleArray): Pair<DoubleArray, DoubleArray> {
        val h = DoubleArray(4 * 6)
        h[0] = 1.0; h[1 * 6 + 1] = 1.0; h[2 * 6 + 4] = 1.0; h[3 * 6 + 5] = 1.0
        val rr = DoubleArray(16) { i -> if (i % 5 == 0) r else 0.0 }
        val hx = DoubleArray(4) { i ->
            (0 until 6).sumOf { h[i * 6 + it] * mean[it] }
        }
        val y = DoubleArray(4) { z[it] - hx[it] }
        val s = add(mulRect(h, cov, 4, 6, 6), rr, 4)
        val k = mulRect(mulRect(cov, transposeRect(h, 4, 6), 6, 6, 4), invert4(s), 6, 4, 4)
        val meanU = DoubleArray(6) { i -> mean[i] + (0 until 4).sumOf { k[i * 4 + it] * y[it] } }
        val kh = mulRect(k, h, 6, 4, 6)
        val iKh = ident(6)
        for (i in 0 until 36) iKh[i] -= kh[i]
        val covU = mul(iKh, cov, 6)
        return meanU to covU
    }

    private fun ident(n: Int) = DoubleArray(n * n) { i -> if (i % (n + 1) == 0) 1.0 else 0.0 }
    private fun transpose(m: DoubleArray, n: Int) = DoubleArray(n * n) { i -> m[(i % n) * n + i / n] }
    private fun transposeRect(m: DoubleArray, rows: Int, cols: Int) = DoubleArray(cols * rows) { i ->
        val r = i / rows; val c = i % rows
        m[c * cols + r]
    }
    private fun matVec(m: DoubleArray, v: DoubleArray, n: Int) = DoubleArray(n) { r ->
        (0 until n).sumOf { m[r * n + it] * v[it] }
    }
    private fun mul(a: DoubleArray, b: DoubleArray, n: Int) = DoubleArray(n * n) { i ->
        val r = i / n; val c = i % n
        (0 until n).sumOf { a[r * n + it] * b[it * n + c] }
    }
    private fun add(a: DoubleArray, b: DoubleArray, n: Int) = DoubleArray(n * n) { a[it] + b[it] }
    private fun mulRect(a: DoubleArray, b: DoubleArray, ar: Int, ac: Int, bc: Int): DoubleArray {
        val o = DoubleArray(ar * bc)
        for (r in 0 until ar) for (c in 0 until bc) {
            var s = 0.0
            for (k in 0 until ac) s += a[r * ac + k] * b[k * bc + c]
            o[r * bc + c] = s
        }
        return o
    }

    private fun invert4(m: DoubleArray): DoubleArray {
        val a = Array(4) { r -> DoubleArray(8) { c -> if (c < 4) m[r * 4 + c] else if (c - 4 == r) 1.0 else 0.0 } }
        for (i in 0..3) {
            var piv = i
            for (r in i..3) if (kotlin.math.abs(a[r][i]) > kotlin.math.abs(a[piv][i])) piv = r
            val tmp = a[i]; a[i] = a[piv]; a[piv] = tmp
            val d = a[i][i]
            if (kotlin.math.abs(d) < 1e-12) continue
            for (c in 0..7) a[i][c] /= d
            for (r in 0..3) if (r != i) {
                val f = a[r][i]
                for (c in 0..7) a[r][c] -= f * a[i][c]
            }
        }
        val inv = DoubleArray(16)
        for (r in 0..3) for (c in 0..3) inv[r * 4 + c] = a[r][c + 4]
        return inv
    }
}

class TrackEngine(private val cfg: TrackingConfig) {
    private var nextId = 1
    val tracks = LinkedHashMap<Int, TrackInternal>()
    private val kf = KalmanImage(cfg.processNoise, cfg.measNoise)
    private val history = ArrayList<Triple<Int, LifecycleState, String>>()

    fun reset() {
        tracks.clear(); nextId = 1; history.clear()
    }

    private fun meas(obs: RoadObservation): DoubleArray {
        val (cx, cy) = if (obs.polygon.isNotEmpty()) groundContact(obs.polygon) else polygonCentroid(obs.polygon)
        val w = maxOf(4.0, (obs.bbox[2] - obs.bbox[0]).toDouble())
        val h = maxOf(4.0, (obs.bbox[3] - obs.bbox[1]).toDouble())
        return doubleArrayOf(cx.toDouble(), cy.toDouble(), w, h)
    }

    private fun spawn(obs: RoadObservation, now: Long): TrackInternal {
        val z = meas(obs)
        val tr = TrackInternal(
            trackId = nextId++,
            hits = 1,
            misses = 0,
            createdNs = now,
            lastNs = now,
            semantic = obs.semanticType,
            geometry = obs.geometryType,
            objectState = obs.state,
            severity = obs.severity,
            polygon = obs.polygon.toList(),
            bbox = obs.bbox.copyOf(),
            modelConfidence = obs.modelConfidence,
            qualityAtMask = obs.qualityAtMask,
            visibility = obs.visibility,
            mean = doubleArrayOf(z[0], z[1], 0.0, 0.0, z[2], z[3]),
            sourceFrameId = obs.sourceFrameId,
            maskRle = obs.maskRle,
        )
        tracks[tr.trackId] = tr
        history += Triple(tr.trackId, tr.state, "spawn")
        return tr
    }

    private fun familyOf(s: SemanticType): String = when (s) {
        SemanticType.POTHOLE, SemanticType.MANHOLE_COVER, SemanticType.UNKNOWN_ANOMALY,
        SemanticType.ROUGH_BROKEN, SemanticType.REPAIR_PATCH -> "hole"
        SemanticType.SPEED_BUMP, SemanticType.ROAD_JOINT -> "band"
        SemanticType.PUDDLE, SemanticType.GRAVEL -> "info"
    }

    private fun compatible(a: SemanticType, b: SemanticType): Boolean =
        a == b || familyOf(a) == familyOf(b)

    private fun active(tr: TrackInternal): Boolean =
        tr.state == LifecycleState.CANDIDATE || tr.state == LifecycleState.TRACKED ||
            tr.state == LifecycleState.CONFIRMED || tr.state == LifecycleState.ALERTED

    fun update(
        observations: List<RoadObservation>,
        nowNs: Long,
        allowNewHighConf: Boolean = true,
        qualityOk: Boolean = true,
        dtS: Double = 0.033,
        cameraYawRate: Double = 0.0,
        frameW: Double = 1920.0,
        focalPx: Double? = null,
    ): List<TrackInternal> {
        val fPx = if (focalPx != null && focalPx > 0) focalPx else frameW * 0.55
        tracks.values.filter { active(it) }.forEach { tr ->
            val p = kf.predict(tr.mean, tr.cov, maxOf(dtS, 1e-3))
            tr.mean = p.first; tr.cov = p.second
            tr.mean[0] += cameraYawRate * dtS * fPx
        }
        val unused = tracks.filter { active(it.value) }.keys.toMutableSet()
        val usedObs = HashSet<Int>()
        val pairs = ArrayList<Triple<Double, Int, Int>>()
        for ((tid, tr) in tracks) {
            if (!active(tr)) continue
            val size = maxOf(tr.mean[4], tr.mean[5])
            val thr = (0.75 * size).coerceIn(16.0, cfg.centerMatchPx)
            observations.forEachIndexed { j, obs ->
                if (!compatible(tr.semantic, obs.semanticType)) return@forEachIndexed
                val iou = maxOf(bboxIou(tr.bbox, obs.bbox), bboxIou(predBox(tr), obs.bbox))
                val (cx, cy) = if (obs.polygon.isNotEmpty()) groundContact(obs.polygon) else polygonCentroid(obs.polygon)
                val dist = hypot(cx.toDouble() - tr.mean[0], cy.toDouble() - tr.mean[1])
                val same = obs.semanticType == tr.semantic
                val score = iou * (if (same) 1.2 else 0.8) - dist / (4.0 * cfg.centerMatchPx)
                if (iou >= cfg.iouMatch || dist < thr) pairs += Triple(score, tid, j)
            }
        }
        pairs.sortByDescending { it.first }
        val takenTr = HashSet<Int>()
        val assigned = ArrayList<Pair<Int?, Int>>()
        for ((_, tid, j) in pairs) {
            if (tid in takenTr || j in usedObs) continue
            assigned += tid to j
            takenTr += tid; usedObs += j; unused.remove(tid)
        }
        observations.indices.forEach { j -> if (j !in usedObs) assigned += null to j }

        for ((tid, j) in assigned) {
            val obs = observations[j]
            if (tid == null) {
                if (allowNewHighConf) spawn(obs, nowNs)
                continue
            }
            val tr = tracks[tid] ?: continue
            val u = kf.update(tr.mean, tr.cov, meas(obs))
            tr.mean = u.first; tr.cov = u.second
            tr.hits += 1; tr.misses = 0; tr.lastNs = nowNs
            tr.polygon = obs.polygon.toList()
            tr.bbox = if (obs.bbox[2] > obs.bbox[0]) obs.bbox.copyOf() else bboxOf(obs.polygon)
            tr.modelConfidence = 0.7 * tr.modelConfidence + 0.3 * obs.modelConfidence
            tr.qualityAtMask = obs.qualityAtMask
            tr.visibility = obs.visibility
            tr.sourceFrameId = obs.sourceFrameId
            tr.maskRle = obs.maskRle
            if (obs.semanticType != SemanticType.UNKNOWN_ANOMALY) tr.semantic = obs.semanticType
            if (obs.geometryType != GeometryType.UNKNOWN) tr.geometry = obs.geometryType
            if (obs.state != ObjectState.UNKNOWN) tr.objectState = obs.state
            if (obs.severity != Severity.UNKNOWN) tr.severity = obs.severity
            tr.fade = 1.0
            advance(tr, nowNs, observed = true, qualityOk = qualityOk)
        }
        for (tid in unused.toList()) {
            val tr = tracks[tid] ?: continue
            tr.misses += 1
            if (!qualityOk) {
                tr.holdUntilNs = nowNs + (cfg.lowQualityHoldS * 1e9).toLong()
                tr.fade = maxOf(0.25, tr.fade * 0.82)
                if (tr.state == LifecycleState.CONFIRMED || tr.state == LifecycleState.ALERTED || tr.state == LifecycleState.TRACKED) {
                    if ((nowNs - tr.lastNs) / 1e9 <= cfg.lowQualityHoldS) continue
                }
            }
            advance(tr, nowNs, observed = false, qualityOk = qualityOk)
        }
        tracks.entries.removeAll {
            it.value.state == LifecycleState.EXPIRED ||
                (it.value.state == LifecycleState.PASSED && (nowNs - it.value.lastNs) / 1e9 > cfg.passedRemoveS)
        }
        return tracks.values.toList()
    }

    private fun predBox(tr: TrackInternal): FloatArray {
        val cx = tr.mean[0]; val cy = tr.mean[1]; val w = tr.mean[4]; val h = tr.mean[5]
        return floatArrayOf((cx - w / 2).toFloat(), (cy - h).toFloat(), (cx + w / 2).toFloat(), cy.toFloat())
    }

    fun markPassed(trackId: Int, nowNs: Long, reason: String) {
        val tr = tracks[trackId] ?: return
        tr.state = LifecycleState.PASSED
        tr.expireReason = reason
        tr.lastNs = nowNs
        history += Triple(trackId, tr.state, reason)
    }

    private fun trFade(tr: TrackInternal) = tr.fade

    private fun advance(tr: TrackInternal, nowNs: Long, observed: Boolean, qualityOk: Boolean) {
        val age = (nowNs - tr.createdNs) / 1e9
        var need = cfg.minConfirmHits
        val bump = tr.semantic == SemanticType.POTHOLE || tr.semantic == SemanticType.SPEED_BUMP ||
            (tr.semantic == SemanticType.MANHOLE_COVER && tr.geometry == GeometryType.CONCAVE)
        if (tr.semantic == SemanticType.UNKNOWN_ANOMALY) need += cfg.unknownAnomalyExtraHits
        if (tr.semantic == SemanticType.ROUGH_BROKEN) need += 3
        if (bump) need = minOf(need, maxOf(2, cfg.bumpConfirmHits))
        if (tr.state == LifecycleState.CANDIDATE) {
            val trackedAge = if (bump) 0.08 else cfg.confirmWindowS * 0.4
            if (tr.hits >= 2 && age >= trackedAge) {
                tr.state = LifecycleState.TRACKED
                history += Triple(tr.trackId, tr.state, "associated")
            } else if (!observed) {
                if ((nowNs - tr.lastNs) / 1e9 > cfg.candidateMaxAgeS) {
                    tr.state = LifecycleState.EXPIRED
                    tr.expireReason = "candidate_timeout"
                    history += Triple(tr.trackId, tr.state, tr.expireReason)
                }
                return
            }
        }
        if (tr.state == LifecycleState.TRACKED && qualityOk && observed) {
            val confAge = if (bump) cfg.confirmWindowS * 0.5 else cfg.confirmWindowS
            if (tr.hits >= need && age >= confAge) {
                tr.state = LifecycleState.CONFIRMED
                tr.confirmedNs = nowNs
                history += Triple(tr.trackId, tr.state, "stable_visible")
            }
        }
        if ((tr.state == LifecycleState.TRACKED || tr.state == LifecycleState.CONFIRMED || tr.state == LifecycleState.ALERTED) && !observed) {
            if ((nowNs - tr.lastNs) / 1e9 > cfg.lowQualityHoldS + 0.25) {
                tr.state = LifecycleState.EXPIRED
                tr.expireReason = "lost"
                tr.fade = 0.0
                history += Triple(tr.trackId, tr.state, tr.expireReason)
            }
        }
    }
}
