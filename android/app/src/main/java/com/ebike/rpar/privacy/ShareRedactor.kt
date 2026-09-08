package com.ebike.rpar.privacy

import org.json.JSONObject
import java.io.File

object ShareRedactor {
    fun export(src: File, dest: File): File {
        dest.mkdirs()
        src.walkTopDown().filter { it.isFile }.forEach { f ->
            val rel = f.relativeTo(src)
            val out = File(dest, rel.path)
            out.parentFile?.mkdirs()
            when {
                rel.invariantSeparatorsPath.contains("location/location.jsonl") -> {
                    out.writeText(f.readLines().joinToString("\n") { line ->
                        if (line.isBlank()) "" else coarsen(JSONObject(line)).toString()
                    }.trim() + if (f.length() > 0) "\n" else "")
                }
                rel.name == "manifest.json" -> {
                    val o = JSONObject(f.readText())
                    o.put("privacy_mode", "SHARE_REDACTED")
                    o.put("notes", o.optString("notes") + " | GPS coarsened; pixel redaction via desktop rpar share")
                    out.writeText(o.toString(2))
                }
                else -> f.copyTo(out, overwrite = true)
            }
        }
        File(dest, "SHARE_README.txt").writeText(
            "SHARE_REDACTED: location coarsened. For face/plate video blur run desktop: rpar share <session>\n",
        )
        return dest
    }

    private fun coarsen(o: JSONObject): JSONObject {
        if (o.has("latitude") && !o.isNull("latitude")) o.put("latitude", kotlin.math.round(o.getDouble("latitude") * 100.0) / 100.0)
        if (o.has("longitude") && !o.isNull("longitude")) o.put("longitude", kotlin.math.round(o.getDouble("longitude") * 100.0) / 100.0)
        o.put("horizontal_accuracy_m", maxOf(o.optDouble("horizontal_accuracy_m", 0.0), 150.0))
        o.put("privacy", "coarsened")
        return o
    }
}
