package com.ebike.rpar.privacy

import org.json.JSONObject
import java.util.regex.Pattern

object CrashSanitizer {
    private val PATH = Pattern.compile("""([A-Za-z]:)?[/\\][^\s"']+\.(mp4|jpg|png|jsonl|json|bin)""", Pattern.CASE_INSENSITIVE)
    private val SENSITIVE = setOf(
        "latitude", "longitude", "altitude", "file_path", "absolute_path", "path",
        "video_path", "frame_bgr", "jpeg", "bitmap",
    )

    fun sanitizeText(text: String): String {
        var out = PATH.matcher(text).replaceAll("[path]")
        out = out.replace(Regex(""""latitude"\s*:\s*-?\d+(\.\d+)?"""), """"latitude": null""")
        out = out.replace(Regex(""""longitude"\s*:\s*-?\d+(\.\d+)?"""), """"longitude": null""")
        return out
    }

    fun sanitize(obj: JSONObject): JSONObject {
        val keys = obj.keys()
        val out = JSONObject()
        while (keys.hasNext()) {
            val k = keys.next()
            val v = obj.get(k)
            out.put(k, if (k.lowercase() in SENSITIVE) JSONObject.NULL else if (v is String) sanitizeText(v) else v)
        }
        return out
    }
}
