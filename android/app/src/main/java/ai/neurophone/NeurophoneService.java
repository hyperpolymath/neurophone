// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
package ai.neurophone;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

/**
 * Thin foreground {@link Service} shim (sub-issue #111).
 *
 * <p>All business logic (sensor &rarr; LSM &rarr; ESN &rarr; bridge loop,
 * salience computation) lives behind the {@code crates/neurophone-android}
 * JNI boundary (issue #110, already merged). This Java shim only does what
 * Android's platform APIs require a JVM class for: the foreground
 * notification/wake-lock ceremony, and forwarding raw
 * {@link SensorEventListener} callbacks into {@link NativeLib#processSensor}.
 *
 * <p>Replaces the legacy Kotlin {@code NeurophoneService.kt}. gossamer itself
 * has no {@code Service}-lifecycle primitive to extend (verified: gossamer's
 * Android surface is limited to the WebView Activity + JS bridge in
 * {@code hyperpolymath/gossamer android/src/main/java/io/gossamer/}); this
 * class extends {@code android.app.Service} directly, per
 * {@code docs/migrations/RFC-ANDROID-KOTLIN-TO-RUST.adoc} "Path (A)" (hand-
 * written Java shim, immediate JNI delegation). See {@code android/README.adoc}
 * for why this supersedes the RFC's Q6 "gossamer-android-services upstream
 * companion" plan, which was never implemented upstream.
 *
 * <p>Permission set + notification channel match
 * {@code docs/OS_INTEGRATION.adoc} "Foreground service" / "Permissions".
 */
public final class NeurophoneService extends Service implements SensorEventListener {

    private static final String CHANNEL_ID = "neurophone_runtime";
    private static final int NOTIF_ID = 0x4E50; // 'NP'

    /** Sensor sampling cadence; ~50 Hz per docs/OS_INTEGRATION.adoc. */
    private static final int SENSOR_DELAY = SensorManager.SENSOR_DELAY_GAME;

    /** Reported accuracy when Android doesn't hand us a real value. */
    private static final int DEFAULT_ACCURACY = 3; // SENSOR_STATUS_ACCURACY_HIGH

    private SensorManager sensorManager;
    private PowerManager.WakeLock wakeLock;

    @Override
    public void onCreate() {
        super.onCreate();
        ensureChannel();

        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "neurophone:service");
        wakeLock.setReferenceCounted(false);
        wakeLock.acquire(10 * 60 * 1000L);

        // Guarded: dev builds / test devices without the native lib loaded
        // must not crash the service on create.
        try {
            NativeLib.init(null);
            NativeLib.start();
        } catch (Throwable t) {
            // No native lib present yet — service still starts so the
            // notification/UI shell is inspectable.
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        startForegroundCompat();
        registerSensors();
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
        try {
            NativeLib.stop();
        } catch (Throwable t) {
            // No native lib present yet.
        }
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
        }
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        int typeId = typeIdFor(event.sensor.getType());
        long timestampNs = System.currentTimeMillis() * 1_000_000L;
        try {
            NativeLib.processSensor(typeId, event.values, timestampNs, DEFAULT_ACCURACY);
        } catch (Throwable t) {
            // No native lib present yet.
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // no-op
    }

    /**
     * Maps an Android {@link Sensor} type constant to the compact id space
     * {@code sensor_map.rs} expects: accelerometer=1, magnetometer=2,
     * gyroscope=4, light=5, proximity=8, everything else=0 (unmapped, the
     * Rust side rejects it).
     */
    private static int typeIdFor(int sensorType) {
        switch (sensorType) {
            case Sensor.TYPE_ACCELEROMETER:
                return 1;
            case Sensor.TYPE_MAGNETIC_FIELD:
                return 2;
            case Sensor.TYPE_GYROSCOPE:
                return 4;
            case Sensor.TYPE_LIGHT:
                return 5;
            case Sensor.TYPE_PROXIMITY:
                return 8;
            default:
                return 0;
        }
    }

    private void registerSensors() {
        if (sensorManager == null) {
            return;
        }
        for (int type : new int[] {
                Sensor.TYPE_ACCELEROMETER,
                Sensor.TYPE_GYROSCOPE,
                Sensor.TYPE_MAGNETIC_FIELD,
                Sensor.TYPE_LIGHT,
                Sensor.TYPE_PROXIMITY,
        }) {
            registerIfPresent(type);
        }
    }

    private void registerIfPresent(int sensorType) {
        Sensor sensor = sensorManager.getDefaultSensor(sensorType);
        if (sensor != null) {
            sensorManager.registerListener(this, sensor, SENSOR_DELAY);
        }
    }

    private void startForegroundCompat() {
        PendingIntent open = PendingIntent.getActivity(
                this, 0,
                new Intent(this, NeurophoneActivity.class),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        Notification notif = new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle(getString(R.string.service_notification_title))
                .setContentText(getString(R.string.service_notification_text))
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentIntent(open)
                .setOngoing(true)
                .build();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIF_ID, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        } else {
            startForeground(NOTIF_ID, notif);
        }
    }

    private void ensureChannel() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm.getNotificationChannel(CHANNEL_ID) == null) {
            NotificationChannel ch = new NotificationChannel(
                    CHANNEL_ID,
                    getString(R.string.service_channel_name),
                    NotificationManager.IMPORTANCE_LOW);
            ch.setDescription(getString(R.string.service_channel_desc));
            nm.createNotificationChannel(ch);
        }
    }
}
