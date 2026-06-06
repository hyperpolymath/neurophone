// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
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
 * Thin foreground {@link Service} shim for the Gossamer migration.
 *
 * <p>All business logic (sensor &rarr; LSM &rarr; ESN &rarr; bridge loop,
 * salience computation, widget publishing) now lives in Rust behind the
 * {@code crates/neurophone-android} JNI boundary. This Java shim only:
 * <ul>
 *   <li>holds the foreground notification + wake lock so Android keeps us
 *       alive,</li>
 *   <li>calls {@link NativeLib#start()} on create and {@link NativeLib#stop()}
 *       on destroy,</li>
 *   <li>forwards raw sensor events into Rust via
 *       {@link NativeLib#processSensor(int, float[], long, int)}.</li>
 * </ul>
 *
 * <p>Hand-written Java is permitted only under {@code android/} (see
 * {@code .hypatia-baseline.json}); every method below is deliberately minimal.
 *
 * <p>TODO(#83): this shim is the migration seam. Once sub-PR #4
 * (NativeLib&rarr;Rust JNI) and sub-PR #3 (Gossamer scaffolding) land, the
 * remaining Android-owned concerns (foreground notification, wake lock,
 * sensor registration) should move behind Gossamer/Rust and this file should
 * shrink further or be removed entirely with the rest of {@code android/}.
 */
public final class NeurophoneService extends Service implements SensorEventListener {

    private static final String CHANNEL_ID = "neurophone_runtime";
    private static final int NOTIF_ID = 0x4E50; // 'NP'

    /** Sensor sampling cadence; ~50 Hz, matching the prior Kotlin service. */
    private static final int SENSOR_DELAY = SensorManager.SENSOR_DELAY_GAME;

    /** Default reported accuracy when a real value is unavailable. */
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

        // TODO(#83 rebase): NativeLib is a Kotlin `object`; once sub-PR #4 lands
        // the Rust JNI library `neurophone_android` must export init/start/stop/
        // processSensor. Guard so dev hardware without the native lib still runs.
        try {
            NativeLib.INSTANCE.init(null);
            NativeLib.INSTANCE.start();
        } catch (Throwable t) {
            // dev mode: no native library present yet
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
            NativeLib.INSTANCE.stop();
        } catch (Throwable t) {
            // dev mode: no native library present yet
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
        // Replicate the prior Kotlin convenience mapping: callers identified
        // sensors by name, here we collapse to the same numeric id space and
        // forward straight to Rust. timestamp ms -> ns.
        long timestampNs = System.currentTimeMillis() * 1_000_000L;
        try {
            NativeLib.INSTANCE.processSensor(typeId, event.values, timestampNs, DEFAULT_ACCURACY);
        } catch (Throwable t) {
            // dev mode without JNI
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // no-op
    }

    /**
     * Maps an Android {@link Sensor} type constant to the compact id space the
     * Rust core expects. Mirrors the sensor-name &rarr; id table that lived in
     * {@code NativeLib.pushSensorEvent} on the Kotlin side:
     * accelerometer=1, magnetometer=2, gyroscope=4, light=5, proximity=8,
     * everything else=0.
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
        // TODO(#83): widen beyond accelerometer once the Rust core consumes the
        // full sensor set; the type-id mapping above already covers them.
        registerIfPresent(Sensor.TYPE_ACCELEROMETER);
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
                new Intent(this, MainActivity.class),
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
