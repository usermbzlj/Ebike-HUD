package com.ebike.rpar.camera

import kotlin.math.abs

data class CaptureSyncHit(
    val matched: Boolean,
    val resultTs: Long?,
    val codes: List<String>,
    val payload: Any?,
)

/**
 * 1:1 SENSOR_TIMESTAMP match between CaptureResult and Image (SYNC-002).
 * Detects REORDER / DUPLICATE / UNMATCHED_FRAME / ORPHAN_RESULT.
 */
class CaptureResultRing(
    private val maxDtNs: Long = 2_000_000L,
    private val capacity: Int = 24,
) {
    private data class Slot(val ts: Long, val payload: Any?, var used: Boolean = false)

    private val slots = ArrayDeque<Slot>(capacity + 1)
    private var lastImageTs: Long? = null

    fun pushResult(ts: Long, payload: Any? = null): List<String> {
        val orphans = ArrayList<String>()
        while (slots.size >= capacity) {
            val old = slots.removeFirst()
            if (!old.used) orphans += "ORPHAN_RESULT"
        }
        slots.addLast(Slot(ts, payload))
        return orphans
    }

    fun matchImage(imageTs: Long): CaptureSyncHit {
        val codes = ArrayList<String>()
        val prev = lastImageTs
        if (prev != null && imageTs < prev) codes += "REORDER"
        if (prev != null && imageTs == prev) codes += "DUPLICATE"
        lastImageTs = imageTs
        val exact = slots.firstOrNull { !it.used && it.ts == imageTs }
        val slot = exact ?: slots.filter { !it.used }.minByOrNull { abs(it.ts - imageTs) }
            ?.takeIf { abs(it.ts - imageTs) <= maxDtNs }
        if (slot == null) {
            codes += "UNMATCHED_FRAME"
            return CaptureSyncHit(false, null, codes, null)
        }
        slot.used = true
        return CaptureSyncHit(true, slot.ts, codes, slot.payload)
    }
}
