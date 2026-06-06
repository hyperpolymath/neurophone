// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>

package ai.neurophone;

import io.gossamer.GossamerActivity;

/**
 * NeurophoneMainActivity — Gossamer webview host for NeuroPhone on Android.
 *
 * <p>Scaffolding only (epic #83, RFC PR #97, sub-issue #109, sub-PR #3). This
 * shim extends {@link io.gossamer.GossamerActivity} (from the gossamer Android
 * source tree, package {@code io.gossamer}) and overrides
 * {@link #getInitialHtml()} to display placeholder content in the Gossamer
 * WebView. It is the eventual replacement for the legacy Kotlin
 * {@code ai.neurophone.MainActivity} + its native-view UI.
 *
 * <p>Gossamer on Android is webview-only today: {@code GossamerActivity}
 * creates a full-screen {@link android.webkit.WebView}, loads
 * {@code libgossamer.so} (Zig FFI) in a static block, registers the
 * {@code GossamerBridge} JavaScript interface, and calls {@code nativeInit()}.
 * Subclasses provide content by overriding {@code getInitialHtml()} or
 * {@code getInitialUrl()}.
 *
 * <p><b>Out of scope for this sub-PR</b> (deliberately NOT ported here):
 * <ul>
 *   <li>{@code NativeLib} JNI bindings — sub-PR #4. The 11 native methods
 *       (init/start/stop/processSensor/queryLocal/queryClaude/query/
 *       getNeuralContext/getState/reset/isRunning) on {@code ai.neurophone.NativeLib}
 *       will be invoked over the Gossamer IPC bridge, not from this Activity.</li>
 *   <li>{@code NeurophoneService} foreground sensor loop — sub-PR #5.</li>
 *   <li>{@code BootReceiver} — sub-PR #6.</li>
 *   <li>Home-screen widgets — sub-PR #7.</li>
 *   <li>Real frontend bundle ({@code frontendDist}) + IPC command wiring — sub-PR #8.</li>
 * </ul>
 */
public class NeurophoneMainActivity extends GossamerActivity {

    /**
     * Placeholder HTML rendered in the Gossamer WebView.
     *
     * <p>TODO(#83 sub-PR #8): replace this inline placeholder with the real
     * NeuroPhone frontend served from {@code gossamer.conf.json}'s
     * {@code build.frontendDist} (or {@code getInitialUrl()} in dev), and wire
     * the UI to the Rust core via the Gossamer IPC bridge
     * ({@code window.GossamerBridge.postMessage}).
     */
    @Override
    protected String getInitialHtml() {
        return "<!DOCTYPE html>"
            + "<html lang=\"en\"><head><meta charset=\"utf-8\">"
            + "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            + "<title>NeuroPhone</title>"
            + "<style>"
            + "body{font-family:system-ui,sans-serif;margin:0;display:flex;"
            + "flex-direction:column;align-items:center;justify-content:center;"
            + "min-height:100vh;background:#0b1020;color:#e6ecff;text-align:center;padding:1.5rem}"
            + "h1{font-size:1.6rem;margin:0 0 .5rem}"
            + "p{opacity:.8;max-width:32ch;line-height:1.5}"
            + "code{background:#1b2444;padding:.1em .35em;border-radius:.25em}"
            + "</style></head><body>"
            + "<h1>NeuroPhone &middot; Gossamer</h1>"
            + "<p>Webview scaffold is live. The real frontend and IPC wiring land in"
            + " later sub-PRs of the Android Kotlin&rarr;Rust/Gossamer migration"
            + " (<code>#83</code>).</p>"
            + "</body></html>";
    }

    // TODO(#83 sub-PR #8): override getInitialUrl() to return
    // gossamer.conf.json build.devUrl during development, falling back to the
    // bundled frontendDist in release builds.
}
