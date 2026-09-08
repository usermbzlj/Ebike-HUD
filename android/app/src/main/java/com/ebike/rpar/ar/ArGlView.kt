package com.ebike.rpar.ar

import android.content.Context
import android.graphics.PixelFormat
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import com.ebike.rpar.model.RenderPrimitive
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.hypot

class ArGlView(context: Context) : GLSurfaceView(context) {
    val renderer = ArRenderer()

    init {
        setEGLContextClientVersion(2)
        setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        holder.setFormat(PixelFormat.TRANSLUCENT)
        setZOrderMediaOverlay(true)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun setPrimitives(
        prims: List<RenderPrimitive>,
        width: Int,
        height: Int,
        night: Boolean = false,
        nightBrightness: Float = 0.72f,
        overlayAlpha: Float = 1f,
        strokeScale: Float = 1f,
    ) {
        renderer.set(prims, width, height, night, nightBrightness, overlayAlpha, strokeScale)
    }
}

class ArRenderer : GLSurfaceView.Renderer {
    @Volatile private var prims: List<RenderPrimitive> = emptyList()
    @Volatile private var w = 1920
    @Volatile private var h = 1080
    @Volatile private var night = false
    @Volatile private var nightB = 0.72f
    @Volatile private var overlayA = 1f
    @Volatile private var strokeS = 1f
    private var program = 0
    private var aPos = 0
    private var uColor = 0

    fun set(p: List<RenderPrimitive>, width: Int, height: Int, nightMode: Boolean = false, brightness: Float = 0.72f, overlayAlpha: Float = 1f, strokeScale: Float = 1f) {
        prims = p
        night = nightMode
        nightB = brightness
        overlayA = overlayAlpha.coerceIn(0.15f, 1f)
        strokeS = strokeScale.coerceIn(0.5f, 2.5f)
        if (width > 0) w = width
        if (height > 0) h = height
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        val vs = """
            attribute vec2 aPos;
            void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
        """.trimIndent()
        val fs = """
            precision mediump float;
            uniform vec4 uColor;
            void main() { gl_FragColor = uColor; }
        """.trimIndent()
        program = link(vs, fs)
        aPos = GLES20.glGetAttribLocation(program, "aPos")
        uColor = GLES20.glGetUniformLocation(program, "uColor")
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClearColor(0f, 0f, 0f, 0f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        GLES20.glUseProgram(program)
        val snapshot = prims
        for (p in snapshot) {
            if (p.polygon.size < 2) continue
            val ndc = p.polygon.map { (x, y) ->
                (x / w.toFloat()) * 2f - 1f to (1f - (y / h.toFloat()) * 2f)
            }
            val mul = if (night) nightB else 1f
            val a = (p.colorRgba[3] * overlayA).coerceIn(0.05f, 1f)
            val col = floatArrayOf(p.colorRgba[0] * mul, p.colorRgba[1] * mul, p.colorRgba[2] * mul, a)
            if (p.polygon.size >= 3 && !p.dashed) {
                drawFan(ndc, floatArrayOf(col[0], col[1], col[2], col[3] * 0.28f))
            }
            drawOutline(ndc, col, p.dashed, p.thickness * strokeS)
        }
    }

    private fun drawFan(pts: List<Pair<Float, Float>>, color: FloatArray) {
        val verts = FloatArray(pts.size * 2)
        pts.forEachIndexed { i, (x, y) -> verts[i * 2] = x; verts[i * 2 + 1] = y }
        GLES20.glUniform4fv(uColor, 1, color, 0)
        val buf = buf(verts)
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, buf)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, pts.size)
    }

    private fun drawOutline(pts: List<Pair<Float, Float>>, color: FloatArray, dashed: Boolean, thickness: Float) {
        GLES20.glLineWidth(thickness.coerceIn(1f, 8f))
        GLES20.glUniform4fv(uColor, 1, color, 0)
        if (!dashed) {
            val closed = pts + pts.first()
            val verts = FloatArray(closed.size * 2)
            closed.forEachIndexed { i, (x, y) -> verts[i * 2] = x; verts[i * 2 + 1] = y }
            val buf = buf(verts)
            GLES20.glEnableVertexAttribArray(aPos)
            GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, buf)
            GLES20.glDrawArrays(GLES20.GL_LINE_STRIP, 0, closed.size)
            return
        }
        val segs = ArrayList<Float>()
        for (i in pts.indices) {
            val a = pts[i]; val b = pts[(i + 1) % pts.size]
            val len = hypot((b.first - a.first).toDouble(), (b.second - a.second).toDouble())
            val n = maxOf(1, (len / 0.04).toInt())
            for (k in 0 until n step 2) {
                val u0 = k / n.toFloat(); val u1 = ((k + 1) / n.toFloat()).coerceAtMost(1f)
                segs += a.first + (b.first - a.first) * u0
                segs += a.second + (b.second - a.second) * u0
                segs += a.first + (b.first - a.first) * u1
                segs += a.second + (b.second - a.second) * u1
            }
        }
        if (segs.isEmpty()) return
        val buf = buf(segs.toFloatArray())
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, buf)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, segs.size / 2)
    }

    private fun buf(v: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(v.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(v); position(0)
        }

    private fun link(vsSrc: String, fsSrc: String): Int {
        fun compile(type: Int, src: String): Int {
            val id = GLES20.glCreateShader(type)
            GLES20.glShaderSource(id, src)
            GLES20.glCompileShader(id)
            return id
        }
        val vs = compile(GLES20.GL_VERTEX_SHADER, vsSrc)
        val fs = compile(GLES20.GL_FRAGMENT_SHADER, fsSrc)
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, vs)
        GLES20.glAttachShader(p, fs)
        GLES20.glLinkProgram(p)
        return p
    }
}
