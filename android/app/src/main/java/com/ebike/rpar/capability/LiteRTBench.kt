package com.ebike.rpar.capability

import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import kotlin.system.measureNanoTime

/** CAP-005: CPU microbench always; Interpreter probe when a .tflite pack is present. */
object LiteRTBench {
    fun run(modelFile: File? = null): JSONObject {
        val timesMs = ArrayList<Double>()
        val n = 64
        val a = FloatArray(n * n) { i -> (i % 17).toFloat() }
        val b = FloatArray(n * n) { i -> ((i * 3) % 13).toFloat() }
        val c = FloatArray(n * n)
        var first = 0.0
        repeat(12) { iter ->
            val ns = measureNanoTime {
                var k = 0
                while (k < n) {
                    var i = 0
                    while (i < n) {
                        var acc = 0f
                        var j = 0
                        while (j < n) {
                            acc += a[i * n + j] * b[j * n + k]
                            j++
                        }
                        c[i * n + k] = acc
                        i++
                    }
                    k++
                }
            }
            val ms = ns / 1e6
            if (iter == 0) first = ms else timesMs += ms
        }
        timesMs.sort()
        val p50 = timesMs[timesMs.size / 2]
        val p95 = timesMs[((timesMs.size - 1) * 0.95).toInt().coerceIn(0, timesMs.lastIndex)]
        val tflite = probeInterpreter(modelFile)
        return JSONObject()
            .put("candidates", JSONArray().put("CPU").put("GPU").put("NPU"))
            .put("note", "CPU GEMM microbench; Interpreter probe on packaged .tflite; GPU/NPU need sideloaded CompiledModel")
            .put("tflite", tflite)
            .put(
                "results",
                JSONArray()
                    .put(JSONObject().put("backend", "CPU").put("status", "ok").put("first_ms", first).put("p50_ms", p50).put("p95_ms", p95).put("stable_10min", "not_run").put("tflite", tflite.optString("status")))
                    .put(JSONObject().put("backend", "GPU").put("status", "unavailable_until_litert_package"))
                    .put(JSONObject().put("backend", "NPU").put("status", "unavailable_until_litert_package")),
            )
    }

    fun probeInterpreter(modelFile: File?): JSONObject {
        val o = JSONObject().put("runtime", "org.tensorflow.lite.Interpreter")
        if (modelFile == null || !modelFile.exists()) {
            return o.put("status", "runtime_present_no_file")
        }
        return try {
            val interp = Interpreter(modelFile, Interpreter.Options().apply { setNumThreads(1) })
            val nIn = interp.inputTensorCount
            val nOut = interp.outputTensorCount
            interp.close()
            o.put("status", "ok")
            o.put("file", modelFile.name)
            o.put("input_tensors", nIn)
            o.put("output_tensors", nOut)
            o
        } catch (t: Throwable) {
            o.put("status", "placeholder_or_incompatible")
            o.put("file", modelFile.name)
            o.put("detail", t.message ?: t.javaClass.simpleName)
            o
        }
    }
}
