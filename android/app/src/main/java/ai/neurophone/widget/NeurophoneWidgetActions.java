// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
package ai.neurophone.widget;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import ai.neurophone.NativeLib;

/**
 * Lightweight broadcast receiver dispatched by non-widget callers (the
 * foreground service, the boot receiver, the share-intent handler) to drive
 * the widget without holding a reference to it.
 *
 * <p>Thin hand-written Java {@link BroadcastReceiver} shim, part of the Android
 * Kotlin->Rust/Gossamer migration (epic #83). It replaces the former Kotlin
 * {@code NeurophoneWidgetActions.kt}.
 *
 * <p>The pre-migration {@code PUBLISH_STATE} path carried neural state in the
 * intent extras and stashed it in SharedPreferences. That is gone: the Rust
 * core is now the single source of truth, so callers only need to nudge the
 * widget into re-reading it via {@link NativeLib#getState()}.
 *
 * <p>Two actions:
 * <ul>
 *   <li>{@code ACTION_FORCE_REFRESH} -> re-render every mounted widget</li>
 *   <li>{@code ACTION_QUERY} -> run a one-shot query against the core via
 *       {@link NativeLib#query(String, boolean)}, then refresh</li>
 * </ul>
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
     * Fire a query into the Rust core. Result handling (surfacing the answer in
     * a notification / activity) is owned elsewhere; here we only ensure the
     * core advances so the next render reflects it.
     */
    private static void runQuery(String query, boolean preferLocal) {
        // TODO(#83 rebase): NativeLib is the Kotlin `object` from the
        //  pre-migration tree (hence `.INSTANCE`); sub-PR #3/#4/#5 may
        //  republish it as a Java facade. Until the JNI bridge lands the call
        //  resolves against the stub, so guard against UnsatisfiedLinkError.
        try {
            NativeLib.INSTANCE.query(query, preferLocal);
        } catch (Throwable t) {
            // No-op: a failed query must not crash the broadcasting caller.
        }
    }
}
