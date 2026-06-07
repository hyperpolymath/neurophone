// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
package ai.neurophone.widget;

import android.app.PendingIntent;
import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.widget.RemoteViews;

import org.json.JSONObject;

import ai.neurophone.MainActivity;
import ai.neurophone.NativeLib;
import ai.neurophone.NeurophoneService;
import ai.neurophone.R;

/**
 * Home-screen widget for NeuroPhone.
 *
 * <p>Thin hand-written Java {@link AppWidgetProvider} shim. It owns no neural
 * state of its own: every render reads the live system state straight out of
 * the Rust core via {@link NativeLib#getState()} (a JSON string) and the "Ask"
 * action funnels a query through {@link NativeLib#query(String, boolean)}.
 *
 * <p>Layout: title bar + state line + salience meter + power/refresh/ask
 * buttons. Three intent actions:
 * <ul>
 *   <li>{@code ACTION_TOGGLE}  -> start/stop the foreground service</li>
 *   <li>{@code ACTION_REFRESH} -> re-read state from the core and redraw</li>
 *   <li>{@code ACTION_QUERY}   -> open MainActivity in query mode</li>
 * </ul>
 *
 * <p>This is part of the Android Kotlin->Rust/Gossamer migration (epic #83).
 * It replaces the former Kotlin {@code NeurophoneAppWidget.kt}; the prior
 * SharedPreferences-backed {@code publishState(...)} path is gone because the
 * Rust core is now the single source of truth. The configure activity was
 * dropped by owner decision, so the widget is fully usable with no setup step.
 */
public final class NeurophoneAppWidget extends AppWidgetProvider {

    public static final String ACTION_REFRESH = "ai.neurophone.widget.ACTION_REFRESH";
    public static final String ACTION_TOGGLE  = "ai.neurophone.widget.ACTION_TOGGLE";
    public static final String ACTION_QUERY   = "ai.neurophone.widget.ACTION_QUERY";

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
        if (ACTION_REFRESH.equals(action)
                || ACTION_TOGGLE.equals(action)
                || ACTION_QUERY.equals(action)) {
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
     * Start/stop the foreground service. The running/stopped truth is read back
     * from the Rust core on the next render via {@link NativeLib#isRunning()},
     * so we do not cache it locally.
     */
    private void toggleService(Context context) {
        final Intent svc = new Intent(context, NeurophoneService.class);
        // TODO(#83 rebase): isRunning() ships with the JNI bridge in sub-PR
        //  #3/#4/#5; until merged the call resolves against the stub NativeLib.
        if (nativeIsRunning()) {
            context.stopService(svc);
        } else {
            context.startForegroundService(svc);
        }
    }

    /**
     * Read fresh neural state from the Rust core and push it into the
     * RemoteViews. No SharedPreferences, no app-side state.
     */
    private void render(Context context, AppWidgetManager mgr, int id) {
        final RemoteViews views =
                new RemoteViews(context.getPackageName(), R.layout.widget_neurophone);

        final boolean running = nativeIsRunning();
        float salience = 0f;
        String description = null;

        // NativeLib.getState() returns a JSON snapshot of the core. Parse
        // defensively: the widget must never crash the launcher on a malformed
        // or empty payload (e.g. before the service has started).
        // TODO(#83 rebase): the concrete JSON schema is finalised alongside the
        //  JNI bridge in sub-PR #3/#4/#5. Keys below are the agreed contract;
        //  confirm on rebase and tighten if the schema changes.
        try {
            final String stateJson = NativeLib.INSTANCE.getState();
            if (stateJson != null && !stateJson.isEmpty()) {
                final JSONObject state = new JSONObject(stateJson);
                salience = (float) state.optDouble("salience", 0d);
                description = state.optString("description", null);
            }
        } catch (Throwable t) {
            // Swallow: fall back to running/stopped string below.
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

        views.setOnClickPendingIntent(
                R.id.widget_toggle, actionPI(context, ACTION_TOGGLE, id, 1));
        views.setOnClickPendingIntent(
                R.id.widget_refresh, actionPI(context, ACTION_REFRESH, id, 2));
        views.setOnClickPendingIntent(
                R.id.widget_query, queryPI(context, id));

        mgr.updateAppWidget(id, views);
    }

    private PendingIntent actionPI(Context context, String action, int widgetId, int requestCode) {
        final Intent intent = new Intent(context, NeurophoneAppWidget.class);
        intent.setAction(action);
        intent.putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId);
        final int flags = PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE;
        return PendingIntent.getBroadcast(
                context, requestCode * 100 + widgetId, intent, flags);
    }

    private PendingIntent queryPI(Context context, int widgetId) {
        final Intent intent = new Intent(context, MainActivity.class);
        intent.setAction(ACTION_QUERY);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        final int flags = PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE;
        return PendingIntent.getActivity(context, 1000 + widgetId, intent, flags);
    }

    /**
     * Ask the Rust core whether the loop is running. Isolated so the one
     * cross-language interop point is easy to retarget on the #83 rebase.
     */
    private static boolean nativeIsRunning() {
        // TODO(#83 rebase): NativeLib is currently the Kotlin `object` from the
        //  pre-migration tree, hence the `.INSTANCE` interop. Sub-PR #3/#4/#5
        //  may republish it as a Java class or static facade; drop `.INSTANCE`
        //  then. Guard against the stub throwing UnsatisfiedLinkError.
        try {
            return NativeLib.INSTANCE.isRunning();
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * Broadcast a refresh to every mounted instance of this widget. Used by
     * {@link NeurophoneWidgetActions} and other non-widget callers (service,
     * boot receiver) to nudge the widget into re-reading core state.
     */
    public static void requestRefresh(Context context) {
        final AppWidgetManager mgr = AppWidgetManager.getInstance(context);
        final int[] ids = mgr.getAppWidgetIds(
                new ComponentName(context, NeurophoneAppWidget.class));
        if (ids.length == 0) {
            return;
        }
        final Intent refresh = new Intent(context, NeurophoneAppWidget.class);
        refresh.setAction(ACTION_REFRESH);
        refresh.putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, ids);
        context.sendBroadcast(refresh);
    }
}
