package com.ebike.rpar.calibration

import android.content.Context
import com.ebike.rpar.geometry.Transforms
import com.ebike.rpar.model.MountProfile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import kotlin.math.abs

enum class WizardStep {
    UPRIGHT, HANDLEBAR_CENTER, HORIZON, CENTERLINE, NEAR_REF, KNOWN_DISTANCE, HEALTH, DONE
}

class CalibrationStore(private val context: Context) {
    private val dir get() = File(context.filesDir, "calibration").also { it.mkdirs() }
    private val prefs get() = context.getSharedPreferences("rpar_cal", Context.MODE_PRIVATE)

    fun defaultProfile(width: Int = 1920, height: Int = 1080): MountProfile =
        hash(Transforms.defaultMount(width, height))

    fun list(): List<MountProfile> {
        val files = dir.listFiles()?.filter { it.extension == "json" } ?: return emptyList()
        return files.mapNotNull {
            try { MountProfile.fromJson(JSONObject(it.readText())) } catch (_: Throwable) { null }
        }
    }

    fun loadActive(width: Int, height: Int): MountProfile {
        val id = prefs.getString("active_id", null)
        return list().firstOrNull { it.profileId == id } ?: defaultProfile(width, height).copy(valid = false)
    }

    fun save(profile: MountProfile): MountProfile {
        val hashed = hash(profile)
        File(dir, "${hashed.profileId}.json").writeText(hashed.toJson().toString(2))
        prefs.edit().putString("active_id", hashed.profileId).apply()
        return hashed
    }

    fun setActive(id: String) {
        prefs.edit().putString("active_id", id).apply()
    }

    fun health(profile: MountProfile, pitchErr: Double, rollErr: Double, pitchLim: Double, rollLim: Double): Pair<Boolean, String> {
        if (abs(pitchErr) > pitchLim || abs(rollErr) > rollLim) return false to "install_health_fail"
        if (profile.cameraHeightM < 0.4 || profile.cameraHeightM > 2.4) return false to "height_out_of_range"
        if (!profile.landscape) return false to "not_landscape"
        return true to "ok"
    }

    fun applyWizard(
        base: MountProfile,
        horizonY: Float,
        centerX: Float,
        nearM: Double,
        known5: Double?,
        known10: Double?,
        known20: Double?,
        pitch: Double,
        roll: Double,
        yaw: Double,
        heightM: Double,
        name: String,
    ): MountProfile {
        val id = name.ifBlank { "mount_${System.currentTimeMillis()}" }.replace(" ", "_")
        var profile = base.copy(
            profileId = id,
            name = name.ifBlank { id },
            horizonYPx = horizonY.toDouble(),
            vehicleCenterlineXPx = centerX.toDouble(),
            nearReferenceM = nearM,
            knownDistance5mPx = known5,
            knownDistance10mPx = known10,
            knownDistance20mPx = known20,
            pitchDeg = pitch,
            rollDeg = roll,
            yawDeg = yaw,
            cameraHeightM = heightM,
            landscape = true,
            lateralOffsetM = -0.32,
            valid = true,
            headlightMean = null,
            headlightValid = false,
        )
        val markers = ArrayList<Pair<Double, Double>>()
        known5?.let { markers += 5.0 to it }
        known10?.let { markers += 10.0 to it }
        known20?.let { markers += 20.0 to it }
        if (markers.size >= 2) {
            profile = Transforms.fitMountFromDistanceMarkers(
                profile,
                Transforms.defaultIntrinsics(),
                markers,
            )
        }
        return save(profile)
    }

    private fun hash(p: MountProfile): MountProfile {
        val md = MessageDigest.getInstance("SHA-256")
        val payload = p.copy(calibrationHash = "").toJson().toString()
        val hex = md.digest(payload.toByteArray()).joinToString("") { "%02x".format(it) }.take(16)
        return p.copy(calibrationHash = hex)
    }

    fun indexJson(): JSONArray {
        val a = JSONArray()
        list().forEach { a.put(it.toJson()) }
        return a
    }
}
