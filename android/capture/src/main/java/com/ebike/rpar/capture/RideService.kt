package com.ebike.rpar.capture

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

class RideService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val channelId = "ride_capture"
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(
                NotificationChannel(channelId, getString(R.string.notif_channel), NotificationManager.IMPORTANCE_LOW),
            )
        }
        val pi = PendingIntent.getActivity(
            this, 0, Intent(this, CaptureActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val notif = NotificationCompat.Builder(this, channelId)
            .setContentTitle(getString(R.string.notif_title))
            .setContentText(getString(R.string.notif_body))
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(pi)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(2, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
        } else {
            startForeground(2, notif)
        }
        return START_STICKY
    }
}
