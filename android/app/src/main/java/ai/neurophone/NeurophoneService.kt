// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
package ai.neurophone

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import ai.neurophone.widget.NeurophoneAppWidget

/**
 * Long-running foreground service that owns the sensor → LSM → ESN → bridge
 * loop. Pushes neural state to the home-screen widget every 1 s.
 *
 * Native (Rust) inference goes through `NativeLib`. While that JNI stack is
 * being completed this service emits a synthetic but smooth salience signal
 * derived from accelerometer variance, so the widget visibly responds even
 * when running on dev hardware without the full LLM.
 */
class NeurophoneService : Service(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accel: Sensor? = null
    private val handler = Handler(Looper.getMainLooper())
    private var wakeLock: PowerManager.WakeLock? = null

    // Rolling stats for synthetic salience (variance of accel magnitude).
    private val window = ArrayDeque<Float>()
    private val windowCap = 50

    private val tickRunnable = object : Runnable {
        override fun run() {
            publishWidgetState()
            handler.postDelayed(this, 1_000)
        }
    }

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        val pm = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "neurophone:service").apply {
            setReferenceCounted(false)
            acquire(10 * 60 * 1000L)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundCompat()
        accel?.let {
            sensorManager.registerListener(
                this, it,
                SensorManager.SENSOR_DELAY_GAME // ~50 Hz
            )
        }
        try { NativeLib.init(); NativeLib.start() } catch (_: Throwable) { /* dev mode */ }
        handler.post(tickRunnable)
        return START_STICKY
    }

    override fun onDestroy() {
        handler.removeCallbacks(tickRunnable)
        sensorManager.unregisterListener(this)
        try { NativeLib.stop() } catch (_: Throwable) {}
        wakeLock?.let { if (it.isHeld) it.release() }
        // TODO(#83 rebase): widget now reads live state from the Rust core
        //  (NativeLib.getState()); we only nudge it to re-render. The old
        //  publishState(...) SharedPreferences push was dropped with the
        //  Java widget shim port.
        NeurophoneAppWidget.requestRefresh(this)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ACCELEROMETER) return
        val mag = kotlin.math.sqrt(
            (event.values[0] * event.values[0] +
             event.values[1] * event.values[1] +
             event.values[2] * event.values[2]).toDouble()
        ).toFloat()
        if (window.size == windowCap) window.removeFirst()
        window.addLast(mag)

        try {
            NativeLib.pushSensorEvent(
                "accelerometer",
                System.currentTimeMillis(),
                event.values
            )
        } catch (_: Throwable) { /* dev mode without JNI */ }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* no-op */ }

    private fun publishWidgetState() {
        // TODO(#83 rebase): the widget reads salience/description straight from
        //  the Rust core via NativeLib.getState() now, so the service no longer
        //  pushes a snapshot — it just asks the widget to re-render. computeSalience()
        //  is retained for the in-service tick/notification path.
        NeurophoneAppWidget.requestRefresh(this)
    }

    private fun computeSalience(): Float {
        if (window.size < 5) return 0f
        val mean = window.average().toFloat()
        val variance = window.sumOf { ((it - mean) * (it - mean)).toDouble() }.toFloat() / window.size
        // Normalise: 0 at rest (~9.81 m/s² constant), 1 at vigorous shake (variance ≥ 25).
        return (variance / 25f).coerceIn(0f, 1f)
    }

    private fun startForegroundCompat() {
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val notif: Notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.service_notification_title))
            .setContentText(getString(R.string.service_notification_text))
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(open)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIF_ID, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIF_ID, notif)
        }
    }

    private fun ensureChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL_ID) == null) {
            val ch = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.service_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply { description = getString(R.string.service_channel_desc) }
            nm.createNotificationChannel(ch)
        }
    }

    companion object {
        private const val CHANNEL_ID = "neurophone_runtime"
        private const val NOTIF_ID = 0x4E50 // 'NP'
    }
}
