package com.ebike.rpar.recorder

import java.io.File

/** Crash leftovers: copy `*.part.mp4` to `*.mp4.recovered` (CAM-010). */
object SegmentRecovery {
    fun recoverPartFiles(dir: File): List<File> {
        if (!dir.exists()) return emptyList()
        val recovered = ArrayList<File>()
        dir.walkTopDown().forEach { part ->
            if (!part.isFile || !part.name.endsWith(".part.mp4")) return@forEach
            val dest = File(part.parentFile, part.name.removeSuffix(".part.mp4") + ".mp4.recovered")
            part.copyTo(dest, overwrite = true)
            part.delete()
            recovered += dest
        }
        return recovered
    }

    fun nextSegmentIndex(videoDir: File): Int {
        val files = videoDir.listFiles() ?: return 0
        var max = -1
        val re = Regex("""segment_(\d+)""")
        for (f in files) {
            val m = re.find(f.name) ?: continue
            max = maxOf(max, m.groupValues[1].toInt())
        }
        return max + 1
    }
}
