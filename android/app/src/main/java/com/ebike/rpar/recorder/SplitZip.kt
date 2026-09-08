package com.ebike.rpar.recorder

import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** REC-006: pack a session into size-capped zip volumes. */
object SplitZip {
    fun export(src: File, destPrefix: File, maxBytes: Long = 512L * 1024 * 1024): List<File> {
        val files = src.walkTopDown().filter { it.isFile }.sortedBy { it.invariantSeparatorsPath }.toList()
        val parts = ArrayList<File>()
        var index = 1
        var batch = ArrayList<File>()
        var size = 0L
        fun flush() {
            if (batch.isEmpty()) return
            val parent = destPrefix.parentFile ?: src.parentFile
            parent.mkdirs()
            val out = File(parent, "${destPrefix.name}.part${index.toString().padStart(2, '0')}.zip")
            ZipOutputStream(out.outputStream()).use { zos ->
                batch.forEach { f ->
                    zos.putNextEntry(ZipEntry(f.relativeTo(src).invariantSeparatorsPath))
                    f.inputStream().use { it.copyTo(zos) }
                    zos.closeEntry()
                }
            }
            parts += out
            index++
            batch = ArrayList()
            size = 0L
        }
        for (f in files) {
            val n = f.length()
            if (batch.isNotEmpty() && size + n > maxBytes) flush()
            batch += f
            size += n
        }
        flush()
        return parts
    }
}
