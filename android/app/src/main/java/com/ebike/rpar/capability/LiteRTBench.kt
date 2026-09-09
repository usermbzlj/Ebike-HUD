package com.ebike.rpar.capability

import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import kotlin.system.measureNanoTime

data class BenchWindow(
    val firstMs: Double,
    val p50Ms: Double,
    val p95Ms: Double,
    val nIters: Int,
    val durationS: Double,
    val memoryStartMb: Double,
    val memoryEndMb: Double,
    val thermalStartC: Double? = null,
    val thermalEndC: Double? = null,
    val backend: String = "CPU",
)

/** CAP-005: CPU/Interpreter windowed microbench. 10 min is a separate requested duration. */
object LiteRTBench {
    const val STABLE_10MIN_S = 600.0

    fun percentile(sortedAsc: List<Double>, p: Double): Double {
        if (sortedAsc.isEmpty()) return 0.0
        val i = ((p / 100.0) * (sortedAsc.size - 1)).toInt().coerceIn(0, sortedAsc.lastIndex)
        return sortedAsc[i]
    }

    fun memoryMb(): Double {
        val rt = Runtime.getRuntime()
        return (rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0)
    }

    fun measureGemmWindow(
        durationMs: Long,
        n: Int = 48,
        thermalC: (() -> Double?)? = null,
    ): BenchWindow {
        val a = FloatArray(n * n) { i -> (i % 17).toFloat() }
        val b = FloatArray(n * n) { i -> ((i * 3) % 13).toFloat() }
        val c = FloatArray(n * n)
        val times = ArrayList<Double>()
        val mem0 = memoryMb()
        val t0c = thermalC?.invoke()
        val t0 = System.nanoTime()
        val end = t0 + durationMs.coerceAtLeast(20L) * 1_000_000L
        var first = 0.0
        var iter = 0
        while (System.nanoTime() < end || iter < 2) {
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
            if (iter == 0) first = ms else times += ms
            iter++
            if (iter > 50_000) break
        }
        times.sort()
        val dur = (System.nanoTime() - t0) / 1e9
        return BenchWindow(
            firstMs = first,
            p50Ms = percentile(times, 50.0),
            p95Ms = percentile(times, 95.0),
            nIters = times.size,
            durationS = dur,
            memoryStartMb = mem0,
            memoryEndMb = memoryMb(),
            thermalStartC = t0c,
            thermalEndC = thermalC?.invoke(),
            backend = "CPU",
        )
    }

    fun run(
        modelFile: File? = null,
        durationMs: Long = 800L,
        thermalC: (() -> Double?)? = null,
        onnxFile: File? = null,
    ): JSONObject {
        val win = measureGemmWindow(durationMs, thermalC = thermalC)
        val tflite = probeInterpreter(modelFile)
        val onnx = probeOnnx(onnxFile)
        val requested = STABLE_10MIN_S
        val stableStatus = if (win.durationS >= requested * 0.95) "ok" else "short_probe"
        val cpu = JSONObject()
        cpu.put("backend", "CPU")
        cpu.put("status", "ok")
        cpu.put("first_ms", win.firstMs)
        cpu.put("p50_ms", win.p50Ms)
        cpu.put("p95_ms", win.p95Ms)
        cpu.put("n_iters", win.nIters)
        cpu.put("ran_s", win.durationS)
        cpu.put("memory_start_mb", win.memoryStartMb)
        cpu.put("memory_end_mb", win.memoryEndMb)
        if (win.thermalStartC != null) cpu.put("thermal_start_c", win.thermalStartC)
        if (win.thermalEndC != null) cpu.put("thermal_end_c", win.thermalEndC)
        cpu.put("tflite", tflite.optString("status"))
        cpu.put("onnx", onnx.optString("status"))
        val stable = JSONObject()
        stable.put("status", stableStatus)
        stable.put("requested_s", requested)
        stable.put("ran_s", win.durationS)
        stable.put("first_ms", win.firstMs)
        stable.put("p50_ms", win.p50Ms)
        stable.put("p95_ms", win.p95Ms)
        stable.put("memory_start_mb", win.memoryStartMb)
        stable.put("memory_end_mb", win.memoryEndMb)
        if (win.thermalStartC != null) stable.put("thermal_start_c", win.thermalStartC)
        if (win.thermalEndC != null) stable.put("thermal_end_c", win.thermalEndC)
        stable.put("note", "CAP-005 10 min window is requested_s=600; first-run probe is short_probe until the capability screen runs the long bench")
        val gpu = JSONObject()
        gpu.put("backend", "GPU")
        gpu.put("status", "unavailable_until_litert_package")
        val npu = JSONObject()
        npu.put("backend", "NPU")
        npu.put("status", "unavailable_until_litert_package")
        val results = JSONArray()
        results.put(cpu)
        results.put(gpu)
        results.put(npu)
        val cands = JSONArray()
        cands.put("CPU")
        cands.put("GPU")
        cands.put("NPU")
        val o = JSONObject()
        o.put("candidates", cands)
        o.put(
            "note",
            "CPU GEMM window + Interpreter/ONNX probe; GPU/NPU need sideloaded CompiledModel on PKC110",
        )
        o.put("tflite", tflite)
        o.put("onnx", onnx)
        o.put("results", results)
        o.put("stable_10min", stable)
        return o
    }

    fun probeInterpreter(modelFile: File?): JSONObject {
        val o = JSONObject()
        o.put("runtime", "org.tensorflow.lite.Interpreter")
        if (modelFile == null || !modelFile.exists()) {
            o.put("status", "runtime_present_no_file")
            return o
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

    fun probeOnnx(modelFile: File?): JSONObject {
        val o = JSONObject()
        o.put("runtime", "onnxruntime-android")
        if (modelFile == null || !modelFile.exists() || modelFile.length() < 1_000_000L) {
            o.put("status", "runtime_present_no_file")
            return o
        }
        return try {
            val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
            val opts = ai.onnxruntime.OrtSession.SessionOptions()
            val session = env.createSession(modelFile.absolutePath, opts)
            val nIn = session.inputNames.size
            val nOut = session.outputNames.size
            session.close()
            o.put("status", "ok")
            o.put("file", modelFile.name)
            o.put("input_tensors", nIn)
            o.put("output_tensors", nOut)
            o
        } catch (t: Throwable) {
            o.put("status", "incompatible_or_missing_runtime")
            o.put("file", modelFile.name)
            o.put("detail", t.message ?: t.javaClass.simpleName)
            o
        }
    }
}
