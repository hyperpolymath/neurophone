// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
package ai.neurophone;

import android.app.Activity;
import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

/**
 * Launcher Activity: hosts a full-screen {@link WebView} loading the
 * AffineScript UI from {@code assets/gossamer-ui/index.html}, bridged to
 * {@link NativeLib} via {@link NeurophoneBridge}.
 *
 * <p>Replaces the legacy Kotlin {@code MainActivity.kt}. Part of the Android
 * Kotlin&rarr;Rust/Gossamer migration (epic #83, sub-issue #109).
 *
 * <p><strong>Provenance / design note:</strong> this class deliberately
 * mirrors the WebView-hosting shape of gossamer's own
 * {@code io.gossamer.GossamerActivity} (full-screen WebView, hardened
 * {@link WebSettings}, a single JS-interface object, careful
 * {@link #onDestroy}) &mdash; that pattern is real and reusable. It does
 * <em>not</em> extend {@code GossamerActivity} or load {@code libgossamer.so}:
 * doing so would require vendoring gossamer's Zig FFI layer and Idris2 ABI
 * ({@code src/interface/abi/Types.idr}) as a neurophone build dependency,
 * which is unbuildable in this environment (no Zig/Idris2 toolchain wired
 * into neurophone's build, no NDK to link it for Android) and out of scope
 * for this migration. Android's {@code android.webkit.WebView} is used
 * directly instead &mdash; which is, in fact, exactly what gossamer's own
 * {@code webview_android.zig} calls into via JNI, so this cuts out a
 * native-code layer we cannot build here rather than skipping anything
 * gossamer would otherwise provide. See {@code android/README.adoc}.
 */
public class NeurophoneActivity extends Activity {

    private static final String INDEX_URL = "file:///android_asset/gossamer-ui/index.html";
    private static final String JS_BRIDGE_NAME = "NeurophoneBridge";

    private WebView webView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        webView = new WebView(this);
        setContentView(webView);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        // Security: no filesystem/content access beyond the packaged assets
        // that loadUrl("file:///android_asset/...") itself resolves.
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);

        // Prevent the WebView from handing links to an external browser.
        webView.setWebViewClient(new WebViewClient());

        webView.addJavascriptInterface(new NeurophoneBridge(), JS_BRIDGE_NAME);

        webView.loadUrl(INDEX_URL);
    }

    @Override
    protected void onDestroy() {
        if (webView != null) {
            webView.removeJavascriptInterface(JS_BRIDGE_NAME);
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }
}
