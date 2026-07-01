// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
package ai.neurophone;

import android.util.Log;
import android.webkit.JavascriptInterface;

/**
 * JavaScript-interface object exposed to the webview UI, registered via
 * {@code WebView.addJavascriptInterface(new NeurophoneBridge(), "NeurophoneBridge")}
 * in {@link NeurophoneActivity#onCreate}.
 *
 * <p>Adapted from gossamer's {@code io.gossamer.GossamerBridge}
 * ({@code hyperpolymath/gossamer android/src/main/java/io/gossamer/GossamerBridge.java})
 * &mdash; same registration shape (one {@code @JavascriptInterface}-annotated
 * object added to the WebView) &mdash; but wired directly to neurophone's own
 * {@link NativeLib} JNI ABI instead of gossamer's generic Zig-core IPC
 * dispatcher ({@code window.__gossamer_invoke} / {@code GossamerBridge
 * .postMessage} in {@code webview_android.zig}).
 *
 * <p><strong>Design divergence from gossamer, on purpose:</strong> gossamer's
 * own bridge is asynchronous (JS {@code postMessage} &rarr; native queue
 * &rarr; {@code evaluateJavascript} resolves a {@code Promise}) because its
 * IPC has to cross a generic Zig command-dispatch layer. Every
 * {@link NativeLib} call is already a fast, synchronous, in-process JNI call
 * with no such queue, so this bridge uses Android's ordinary
 * {@code addJavascriptInterface} synchronous return-value support instead of
 * re-implementing gossamer's async postMessage/callback dance. See
 * {@code android/README.adoc} "Design notes".
 *
 * <p>Every method below is a direct, defensive delegation: {@link NativeLib}
 * throws a Java {@code RuntimeException} on any core-side error (per its own
 * {@code ThrowRuntimeExAndDefault} policy); letting that escape across the
 * WebView JS-interface boundary is unreliable, so it is caught here and
 * turned into a stable failure value ({@code false} / {@code null}) the
 * AffineScript UI layer can check for.
 */
public final class NeurophoneBridge {

    private static final String TAG = "NeurophoneBridge";

    @JavascriptInterface
    public boolean init(String configJson) {
        try {
            return NativeLib.init(configJson);
        } catch (Throwable t) {
            Log.w(TAG, "init failed", t);
            return false;
        }
    }

    @JavascriptInterface
    public boolean start() {
        try {
            return NativeLib.start();
        } catch (Throwable t) {
            Log.w(TAG, "start failed", t);
            return false;
        }
    }

    @JavascriptInterface
    public void stop() {
        try {
            NativeLib.stop();
        } catch (Throwable t) {
            Log.w(TAG, "stop failed", t);
        }
    }

    @JavascriptInterface
    public boolean isRunning() {
        try {
            return NativeLib.isRunning();
        } catch (Throwable t) {
            Log.w(TAG, "isRunning failed", t);
            return false;
        }
    }

    /**
     * Parity only: the foreground {@link NeurophoneService} registers Android
     * {@code SensorEventListener}s and calls {@link NativeLib#processSensor}
     * directly; the webview UI does not drive sensor acquisition. Exposed here
     * so the UI can inject a synthetic reading (manual test / demo mode).
     *
     * <p>{@code addJavascriptInterface} cannot marshal a JS array to a Java
     * {@code float[]} parameter, so {@code valuesJson} is a JSON-encoded array
     * of numbers (e.g. {@code "[0.1,0.2,9.8]"}), parsed defensively here.
     */
    @JavascriptInterface
    public boolean processSensor(int sensorType, String valuesJson, long timestamp, int accuracy) {
        try {
            float[] values = parseFloatArray(valuesJson);
            return NativeLib.processSensor(sensorType, values, timestamp, accuracy);
        } catch (Throwable t) {
            Log.w(TAG, "processSensor failed", t);
            return false;
        }
    }

    @JavascriptInterface
    public String query(String message, boolean preferLocal) {
        try {
            return NativeLib.query(message, preferLocal);
        } catch (Throwable t) {
            Log.w(TAG, "query failed", t);
            return null;
        }
    }

    @JavascriptInterface
    public String queryLocal(String message) {
        try {
            return NativeLib.queryLocal(message);
        } catch (Throwable t) {
            Log.w(TAG, "queryLocal failed", t);
            return null;
        }
    }

    @JavascriptInterface
    public String queryClaude(String message) {
        try {
            return NativeLib.queryClaude(message);
        } catch (Throwable t) {
            Log.w(TAG, "queryClaude failed", t);
            return null;
        }
    }

    @JavascriptInterface
    public String getNeuralContext() {
        try {
            return NativeLib.getNeuralContext();
        } catch (Throwable t) {
            Log.w(TAG, "getNeuralContext failed", t);
            return null;
        }
    }

    @JavascriptInterface
    public String getState() {
        try {
            return NativeLib.getState();
        } catch (Throwable t) {
            Log.w(TAG, "getState failed", t);
            return null;
        }
    }

    @JavascriptInterface
    public void reset() {
        try {
            NativeLib.reset();
        } catch (Throwable t) {
            Log.w(TAG, "reset failed", t);
        }
    }

    /**
     * Minimal, dependency-free parser for a flat JSON array of numbers (no
     * org.json / Gson pulled in for one call site). Malformed input yields an
     * empty array rather than throwing, matching the defensive posture of the
     * rest of this bridge.
     */
    private static float[] parseFloatArray(String json) {
        if (json == null) {
            return new float[0];
        }
        String trimmed = json.trim();
        if (trimmed.length() < 2 || trimmed.charAt(0) != '['
                || trimmed.charAt(trimmed.length() - 1) != ']') {
            return new float[0];
        }
        String inner = trimmed.substring(1, trimmed.length() - 1).trim();
        if (inner.isEmpty()) {
            return new float[0];
        }
        String[] parts = inner.split(",");
        float[] out = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                out[i] = Float.parseFloat(parts[i].trim());
            } catch (NumberFormatException e) {
                out[i] = 0f;
            }
        }
        return out;
    }
}
