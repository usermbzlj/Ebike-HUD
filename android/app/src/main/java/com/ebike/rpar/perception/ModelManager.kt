package com.ebike.rpar.perception

import android.content.Context
import android.util.Log
import com.ebike.rpar.config.RparConfig
import com.ebike.rpar.model.APP_VERSION
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

class ModelManager(private val context: Context, private val cfg: RparConfig) {
    data class Loaded(
        val engine: PerceptionEngine,
        val packageId: String,
        val dir: File?,
        val rollback: Boolean,
        val error: String?,
    )

    fun load(packageDir: File? = null, safeMode: Boolean = false): Loaded {
        if (safeMode) {
            return Loaded(NoOpEngine(), "safe-noop", null, false, null)
        }
        val lastGood = prefs().getString(KEY_LAST_GOOD, cfg.model.packageId)
        return try {
            val chosen = packageDir ?: resolvePackageDir(cfg.model.packageId)
            val engine = loadFromDir(chosen)
            prefs().edit().putString(KEY_LAST_GOOD, chosen?.name ?: cfg.model.packageId).apply()
            Loaded(engine, chosen?.name ?: cfg.model.packageId, chosen, false, null)
        } catch (t: Throwable) {
            Log.w(TAG, "model load failed, rolling back", t)
            try {
                val rb = resolvePackageDir(lastGood ?: cfg.model.packageId)
                val engine = loadFromDir(rb)
                Loaded(engine, lastGood ?: cfg.model.packageId, rb, true, t.message)
            } catch (t2: Throwable) {
                Log.e(TAG, "rollback failed, heuristic assets", t2)
                Loaded(HeuristicEngine(cfg), cfg.model.packageId, null, true, t2.message)
            }
        }
    }

    fun scanSideload(): List<File> {
        val root = File(context.filesDir, "models")
        if (!root.exists()) return emptyList()
        return root.listFiles()?.filter { File(it, "manifest.json").exists() } ?: emptyList()
    }

    private fun loadFromDir(dir: File?): PerceptionEngine {
        val manifest = readManifest(dir)
        if (manifest != null) {
            val compat = manifest.optString("compatible_app", ">=0.1.0")
            if (!compatible(compat, APP_VERSION)) {
                throw IllegalStateException("incompatible app $APP_VERSION vs $compat")
            }
            val expectedObj = manifest.optJSONObject("sha256")
            if (expectedObj != null && dir != null) {
                val it = expectedObj.keys()
                while (it.hasNext()) {
                    val name = it.next()
                    val f = File(dir, name)
                    if (!f.exists()) continue
                    val actual = sha256(f)
                    val want = expectedObj.optString(name)
                    if (want.isNotBlank() && !actual.equals(want, ignoreCase = true)) {
                        throw IllegalStateException("sha256 mismatch $name")
                    }
                }
            } else {
                val expected = manifest.optString("sha256", "")
                val modelFile = dir?.listFiles()?.firstOrNull { it.name.endsWith(".tflite") || it.name.endsWith(".bin") }
                if (expected.isNotBlank() && modelFile != null) {
                    val actual = sha256(modelFile)
                    if (!actual.equals(expected, ignoreCase = true)) {
                        throw IllegalStateException("sha256 mismatch")
                    }
                }
            }
            val engine = manifest.optString("engine", "heuristic")
            val modelFile = dir?.listFiles()?.firstOrNull { it.name.endsWith(".tflite") || it.name.endsWith(".bin") }
            if (engine == "heuristic" || engine == "heuristic-cv") return HeuristicEngine(cfg)
            if (engine == "oracle") return NoOpEngine()
            if (modelFile != null) {
                Log.i(TAG, "LiteRT present but not wired; heuristic fallback")
                return HeuristicEngine(cfg)
            }
        }
        return HeuristicEngine(cfg)
    }

    private fun readManifest(dir: File?): JSONObject? {
        if (dir != null && File(dir, "manifest.json").exists()) {
            return JSONObject(File(dir, "manifest.json").readText())
        }
        return try {
            val path = "models/${cfg.model.packageId}/manifest.json"
            JSONObject(context.assets.open(path).bufferedReader().use { it.readText() })
        } catch (_: Throwable) {
            null
        }
    }

    private fun resolvePackageDir(packageId: String): File? {
        val sideload = File(context.filesDir, "models/$packageId")
        if (File(sideload, "manifest.json").exists()) return sideload
        ensureAssetsCopied()
        val copied = File(context.filesDir, "models/$packageId")
        return if (File(copied, "manifest.json").exists()) copied else null
    }

    private fun ensureAssetsCopied() {
        try {
            val dest = File(context.filesDir, "models/${cfg.model.packageId}")
            dest.mkdirs()
            val names = context.assets.list("models/${cfg.model.packageId}") ?: emptyArray()
            for (name in names) {
                val out = File(dest, name)
                if (out.exists()) continue
                context.assets.open("models/${cfg.model.packageId}/$name").use { input ->
                    out.outputStream().use { input.copyTo(it) }
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "asset copy skipped", t)
        }
    }

    private fun compatible(req: String, app: String): Boolean {
        val m = Regex(""">=\s*([0-9.]+)""").find(req.trim()) ?: return true
        return compareSemver(app, m.groupValues[1]) >= 0
    }

    private fun compareSemver(a: String, b: String): Int {
        val aa = a.split('.').map { it.toIntOrNull() ?: 0 }
        val bb = b.split('.').map { it.toIntOrNull() ?: 0 }
        val n = maxOf(aa.size, bb.size)
        for (i in 0 until n) {
            val d = (aa.getOrElse(i) { 0 }) - (bb.getOrElse(i) { 0 })
            if (d != 0) return d
        }
        return 0
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { ins ->
            val buf = ByteArray(8192)
            while (true) {
                val n = ins.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    private fun prefs() = context.getSharedPreferences("rpar_model", Context.MODE_PRIVATE)

    companion object {
        private const val TAG = "ModelManager"
        private const val KEY_LAST_GOOD = "last_good_package"
    }
}
