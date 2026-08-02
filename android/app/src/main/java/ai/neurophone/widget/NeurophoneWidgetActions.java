// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
package ai.neurophone.widget;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import ai.neurophone.NativeLib;

/**
 * Lightweight {@link BroadcastReceiver} shim (sub-issue #113, second third of
 * the widget triple) dispatched by non-widget callers (service, boot
 * receiver, share-intent handler) to drive the widget without holding a
 * reference to it.
 *
 * <p>Replaces the legacy Kotlin {@code NeurophoneWidgetActions.kt}. The
 * pre-migration {@code PUBLISH_STATE} path (intent extras stashed into
 * SharedPreferences) is gone: the Rust core is the single source of truth,
 * so callers only need to nudge the widget into re-reading it.
 */
public final class NeurophoneWidgetActions extends BroadcastReceiver {

    public static final String ACTION_FORCE_REFRESH = "ai.neurophone.widget.FORCE_REFRESH";
    public static final String ACTION_QUERY = "ai.neurophone.widget.PUBLISH_QUERY";

    /** Query text carried by {@link #ACTION_QUERY}. */
    public static final String EXTRA_QUERY = "query";
    /** Prefer the on-device model over the cloud fallback. Defaults to true. */
    public static final String EXTRA_PREFER_LOCAL = "prefer_local";

    @Override
    public void onReceive(Context context, Intent intent) {
        final String action = intent.getAction();
        if (ACTION_FORCE_REFRESH.equals(action)) {
            NeurophoneAppWidget.requestRefresh(context);
        } else if (ACTION_QUERY.equals(action)) {
            final String query = intent.getStringExtra(EXTRA_QUERY);
            final boolean preferLocal = intent.getBooleanExtra(EXTRA_PREFER_LOCAL, true);
            if (query != null && !query.isEmpty()) {
                runQuery(query, preferLocal);
            }
            // State may have moved as a result of the query; redraw from core.
            NeurophoneAppWidget.requestRefresh(context);
        }
    }

    /**
     * Fire a query into the Rust core. Result handling (surfacing the answer)
     * is owned elsewhere; here we only ensure the core advances so the next
     * render reflects it.
     */
    private static void runQuery(String query, boolean preferLocal) {
        try {
            NativeLib.query(query, preferLocal);
        } catch (Throwable t) {
            // A failed query must not crash the broadcasting caller.
        }
    }
}
