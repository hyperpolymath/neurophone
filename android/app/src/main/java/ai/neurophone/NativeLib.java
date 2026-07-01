// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
package ai.neurophone;

/**
 * JNI surface declarations for {@code ai.neurophone.NativeLib}.
 *
 * <p>This class carries <em>zero</em> logic: every method is a {@code native}
 * declaration that resolves directly to a {@code #[unsafe(no_mangle)]}
 * {@code extern "system" fn Java_ai_neurophone_NativeLib_*} export in
 * {@code crates/neurophone-android/src/lib.rs}. That crate is the real JNI
 * bridge implementation (issue #110, already merged on {@code main}); this
 * file only has to match its 11-method signature exactly so
 * {@code System.loadLibrary} resolution succeeds. See
 * {@code docs/migrations/JNI-SURFACE-AUDIT.adoc} for the authoritative
 * signature table.
 *
 * <p>Part of the Android Kotlin&rarr;Rust/Gossamer migration (epic #83,
 * sub-issue #109). Exempt from the estate Java/Kotlin ban via the
 * {@code android/**}{@code /src/**}{@code /*.java} carve-out in
 * {@code hyperpolymath/standards} {@code governance-reusable.yml} (RFC
 * {@code docs/migrations/RFC-ANDROID-KOTLIN-TO-RUST.adoc} Q1, standards#341).
 */
public final class NativeLib {

    static {
        // Cargo cdylib output for the `neurophone-android` package:
        // libneurophone_android.so (hyphens -> underscores).
        System.loadLibrary("neurophone_android");
    }

    private NativeLib() {
        // Static-methods-only holder; never instantiated.
    }

    /** {@code init(configJson: String?): Boolean} */
    public static native boolean init(String configJson);

    /** {@code start(): Boolean} */
    public static native boolean start();

    /** {@code stop()} */
    public static native void stop();

    /** {@code isRunning(): Boolean} */
    public static native boolean isRunning();

    /**
     * {@code processSensor(sensorType: Int, values: FloatArray, timestamp: Long,
     * accuracy: Int): Boolean}
     */
    public static native boolean processSensor(int sensorType, float[] values, long timestamp, int accuracy);

    /** {@code query(message: String, preferLocal: Boolean): String} */
    public static native String query(String message, boolean preferLocal);

    /** {@code queryLocal(message: String): String} — forces the local model. */
    public static native String queryLocal(String message);

    /** {@code queryClaude(message: String): String} — forces the cloud model. */
    public static native String queryClaude(String message);

    /** {@code getNeuralContext(): String} */
    public static native String getNeuralContext();

    /** {@code getState(): String} — the {@code SystemState} as JSON. */
    public static native String getState();

    /** {@code reset()} */
    public static native void reset();
}
