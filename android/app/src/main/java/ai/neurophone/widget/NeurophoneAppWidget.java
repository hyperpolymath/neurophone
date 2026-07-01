// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
package ai.neurophone.widget;

import android.app.PendingIntent;
import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.widget.RemoteViews;

import org.json.JSONObject;

import ai.neurophone.NativeLib;
import ai.neurophone.NeurophoneActivity;
import ai.neurophone.NeurophoneService;
import ai.neurophone.R;

/**
 * Home-screen widget (sub-issue #113, one third of the widget triple).
 *
 * <p>Thin {@link AppWidgetProvider} shim. Owns no neural state of its own:
 * every render reads the live system state straight out of
 * {@link NativeLib#getState()} (a JSON string) and {@link NativeLib#isRunning()}.
 * The "Ask" action opens {@link NeurophoneActivity}; the actual query is typed
 * there, not in the widget.
 *
 * <p>Replaces the legacy Kotlin {@code NeurophoneAppWidget.kt}. The
 * SharedPreferences-backed {@code publishState(...)} path from the legacy
 * service is gone: the Rust core is the single source of truth, and the
 * widget re-reads it on every {@code ACTION_REFRESH}.
 *
 * <p><strong>{@code NeurophoneWidgetConfigureActivity} is intentionally not
 * ported</strong> &mdash; repeating the same drop decision already recorded
 * once in this repo's history ({@code git show b0de78c}, since deleted along
 * with the rest of the legacy Kotlin tree) and in
 * {@code docs/migrations/RFC-ANDROID-KOTLIN-TO-RUST.adoc} Q3. Issue #113
 * still frames this as an open owner question, so it is repeated here rather
 * than assumed silently: the widget has no configure step and works with a
 * sensible default at install (matches Q3's "DROP" resolution).
 */
public final class NeurophoneAppWidget extends AppWidgetProvider {

    public static final String ACTION_REFRESH = "ai.neurophone.widget.ACTION_REFRESH";
    public static final String ACTION_TOGGLE = "ai.neurophone.widget.ACTION_TOGGLE";
    public static final String ACTION_QUERY = "ai.neurophone.widget.ACTION_QUERY";

    @Override
    public void onUpdate(Context context, AppWidgetManager manager, int[] ids) {
        for (int id : ids) {
            render(context, manager, id);
        }
    }

    @Override
    public void onReceive(Context context, Intent intent) {
        super.onReceive(context, intent);
        final String action = intent.getAction();
        if (ACTION_REFRESH.equals(action) || ACTION_TOGGLE.equals(action)) {
            if (ACTION_TOGGLE.equals(action)) {
                toggleService(context);
            }
            final AppWidgetManager mgr = AppWidgetManager.getInstance(context);
            final int[] ids = mgr.getAppWidgetIds(
                    new ComponentName(context, NeurophoneAppWidget.class));
            for (int id : ids) {
                render(context, mgr, id);
            }
        }
    }

    /**
     * Start/stop the foreground service. Running/stopped truth is re-read
     * from the core on the next render via {@link NativeLib#isRunning()} —
     * never cached locally.
     */
    private void toggleService(Context context) {
        final Intent svc = new Intent(context, NeurophoneService.class);
        if (nativeIsRunning()) {
            context.stopService(svc);
        } else {
            context.startForegroundService(svc);
        }
    }

    /**
     * Read fresh state from the Rust core and push it into the
     * {@link RemoteViews}. No SharedPreferences, no app-side cache.
     */
    private void render(Context context, AppWidgetManager mgr, int id) {
        final RemoteViews views =
                new RemoteViews(context.getPackageName(), R.layout.widget_neurophone);

        final boolean running = nativeIsRunning();
        float salience = 0f;
        String description = null;

        // NativeLib.getState() is a JSON snapshot of SystemState
        // (crates/neurophone-core). Parsed defensively: the widget must never
        // crash the launcher on a malformed/empty payload (e.g. before the
        // service has ever run).
        try {
            final String stateJson = NativeLib.getState();
            if (stateJson != null && !stateJson.isEmpty()) {
                final JSONObject state = new JSONObject(stateJson);
                salience = (float) state.optDouble("salience", 0d);
                description = state.optString("description", null);
            }
        } catch (Throwable t) {
            // Fall back to the running/stopped string below.
            description = null;
        }

        if (salience < 0f) {
            salience = 0f;
        } else if (salience > 1f) {
            salience = 1f;
        }

        views.setTextViewText(
                R.id.widget_state,
                description != null && !description.isEmpty()
                        ? description
                        : context.getString(running
                                ? R.string.widget_state_running
                                : R.string.widget_state_stopped));

        final int saliencePct = (int) (salience * 100f);
        views.setProgressBar(R.id.widget_salience, 100, saliencePct, false);
        views.setTextViewText(R.id.widget_salience_value, saliencePct + "%");

        views.setOnClickPendingIntent(R.id.widget_toggle, actionPI(context, ACTION_TOGGLE, id, 1));
        views.setOnClickPendingIntent(R.id.widget_refresh, actionPI(context, ACTION_REFRESH, id, 2));
        views.setOnClickPendingIntent(R.id.widget_query, queryPI(context, id));

        mgr.updateAppWidget(id, views);
    }

    private PendingIntent actionPI(Context context, String action, int widgetId, int requestCode) {
        final Intent intent = new Intent(context, NeurophoneAppWidget.class);
        intent.setAction(action);
        intent.putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId);
        final int flags = PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE;
        return PendingIntent.getBroadcast(context, requestCode * 100 + widgetId, intent, flags);
    }

    private PendingIntent queryPI(Context context, int widgetId) {
        final Intent intent = new Intent(context, NeurophoneActivity.class);
        intent.setAction(ACTION_QUERY);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        final int flags = PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE;
        return PendingIntent.getActivity(context, 1000 + widgetId, intent, flags);
    }

    /** Isolated so the one cross-language interop point is easy to audit. */
    private static boolean nativeIsRunning() {
        try {
            return NativeLib.isRunning();
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * Broadcast a refresh to every mounted widget instance. Used by
     * {@link NeurophoneWidgetActions} and other non-widget callers (service,
     * boot receiver) to nudge the widget into re-reading core state.
     */
    public static void requestRefresh(Context context) {
        final AppWidgetManager mgr = AppWidgetManager.getInstance(context);
        final int[] ids = mgr.getAppWidgetIds(new ComponentName(context, NeurophoneAppWidget.class));
        if (ids.length == 0) {
            return;
        }
        final Intent refresh = new Intent(context, NeurophoneAppWidget.class);
        refresh.setAction(ACTION_REFRESH);
        refresh.putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, ids);
        context.sendBroadcast(refresh);
    }
}
