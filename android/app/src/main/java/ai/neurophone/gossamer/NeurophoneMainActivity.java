// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2025 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>

package ai.neurophone.gossamer;

import io.gossamer.GossamerActivity;

/**
 * NeurophoneMainActivity — Gossamer entry-point Activity for the NeuroPhone
 * Android app.
 *
 * <p>This is the scaffold introduced by sub-PR #3 of the Kotlin->Rust/Gossamer
 * migration epic (#83, RFC #97, tracking sub-issue #109). It extends
 * {@link io.gossamer.GossamerActivity} (from the {@code gossamer} Android
 * library, package {@code io.gossamer}), which hosts a full-screen WebView,
 * loads {@code libgossamer.so} via {@code System.loadLibrary("gossamer")} in a
 * static initialiser, and registers the {@code GossamerBridge} JavaScript
 * interface for native IPC.
 *
 * <p>Gossamer on Android is webview-only today, so this scaffold only overrides
 * {@link #getInitialHtml()} to render placeholder content. No native NeuroPhone
 * code is wired in yet.
 *
 * <p>This shim deliberately does NOT replace the legacy Kotlin
 * {@code ai.neurophone.MainActivity}, {@code NeurophoneService},
 * {@code BootReceiver}, {@code NativeLib}, or the widgets — those are ported in
 * later sub-PRs. The {@code android/} subtree is exempt from the banned-language
 * CI gate (see {@code .hypatia-baseline.json}, tracking #97), so this
 * hand-written Java shim is permitted.
 *
 * <p>TODO(#83 sub-PR #4): replace the placeholder HTML with the real bundled web
 *    UI by overriding {@link #getInitialUrl()} (or pointing
 *    {@code gossamer.conf.json}'s {@code build.frontendDist} at it).
 * <p>TODO(#83 sub-PR #5): port {@code ai.neurophone.NativeLib}'s 11 JNI methods
 *    (init/start/stop/processSensor/queryLocal/queryClaude/query/
 *    getNeuralContext/getState/reset/isRunning) onto the Gossamer IPC bridge,
 *    backed by the Rust core in {@code crates/neurophone-android} /
 *    {@code crates/neurophone-core}.
 * <p>TODO(#83 sub-PR #6): port the sensor pipeline (accelerometer/gyroscope/
 *    magnetometer/light/proximity) feeding the LSM->ESN loop.
 * <p>TODO(#83 sub-PR #7): port {@code NeurophoneService} (foreground service),
 *    {@code BootReceiver}, and the home-screen widgets.
 */
public class NeurophoneMainActivity extends GossamerActivity {

    /**
     * Placeholder content for the Gossamer WebView.
     *
     * <p>Returning non-null HTML makes {@link GossamerActivity} call
     * {@code webView.loadData(...)} instead of loading a URL. Replaced by the
     * real UI in sub-PR #4.
     */
    @Override
    protected String getInitialHtml() {
        // TODO(#83 sub-PR #4): remove this placeholder once the real frontend is bundled.
        return "<!DOCTYPE html>"
            + "<html lang=\"en\"><head><meta charset=\"utf-8\">"
            + "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            + "<title>NeuroPhone</title></head>"
            + "<body style=\"font-family:sans-serif;margin:2rem;\">"
            + "<h1>NeuroPhone</h1>"
            + "<p>Gossamer scaffold (sub-PR #3). UI and native bridge land in sub-PRs #4-#7.</p>"
            + "</body></html>";
    }
}
