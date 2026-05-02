// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
package ai.neurophone.widget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.widget.RemoteViews
import ai.neurophone.MainActivity
import ai.neurophone.NeurophoneService
import ai.neurophone.R

/**
 * Home-screen widget for NeuroPhone.
 *
 * Layout: title bar + state line + salience meter + "Ask" button.
 * Three intent actions:
 *   - ACTION_TOGGLE  → start/stop foreground service
 *   - ACTION_REFRESH → re-read state from prefs and redraw
 *   - ACTION_QUERY   → open MainActivity in query mode
 */
class NeurophoneAppWidget : AppWidgetProvider() {

    override fun onUpdate(
        context: Context,
        manager: AppWidgetManager,
        ids: IntArray
    ) {
        ids.forEach { id -> render(context, manager, id) }
    }

    override fun onReceive(context: Context, intent: Intent) {
        super.onReceive(context, intent)
        when (intent.action) {
            ACTION_REFRESH, ACTION_TOGGLE, ACTION_QUERY -> {
                val mgr = AppWidgetManager.getInstance(context)
                val ids = mgr.getAppWidgetIds(ComponentName(context, NeurophoneAppWidget::class.java))
                if (intent.action == ACTION_TOGGLE) toggleService(context)
                ids.forEach { render(context, mgr, it) }
            }
        }
    }

    private fun toggleService(context: Context) {
        val prefs = prefs(context)
        val running = prefs.getBoolean(KEY_RUNNING, false)
        val intent = Intent(context, NeurophoneService::class.java)
        if (running) {
            context.stopService(intent)
            prefs.edit().putBoolean(KEY_RUNNING, false).apply()
        } else {
            context.startForegroundService(intent)
            prefs.edit().putBoolean(KEY_RUNNING, true).apply()
        }
    }

    private fun render(context: Context, mgr: AppWidgetManager, id: Int) {
        val views = RemoteViews(context.packageName, R.layout.widget_neurophone)
        val prefs = prefs(context)

        val running = prefs.getBoolean(KEY_RUNNING, false)
        val salience = prefs.getFloat(KEY_SALIENCE, 0f).coerceIn(0f, 1f)
        val description = prefs.getString(KEY_DESCRIPTION, null)

        views.setTextViewText(
            R.id.widget_state,
            description ?: context.getString(
                if (running) R.string.widget_state_running else R.string.widget_state_stopped
            )
        )
        val saliencePct = (salience * 100f).toInt()
        views.setProgressBar(R.id.widget_salience, 100, saliencePct, false)
        views.setTextViewText(R.id.widget_salience_value, "$saliencePct%")

        // Toggle (start/stop service)
        views.setOnClickPendingIntent(
            R.id.widget_toggle,
            actionPI(context, ACTION_TOGGLE, id, requestCode = 1)
        )
        // Refresh (re-render)
        views.setOnClickPendingIntent(
            R.id.widget_refresh,
            actionPI(context, ACTION_REFRESH, id, requestCode = 2)
        )
        // Ask -> open MainActivity in query mode
        views.setOnClickPendingIntent(
            R.id.widget_query,
            queryPI(context, id)
        )

        mgr.updateAppWidget(id, views)
    }

    private fun actionPI(context: Context, action: String, widgetId: Int, requestCode: Int): PendingIntent {
        val intent = Intent(context, NeurophoneAppWidget::class.java).apply {
            this.action = action
            putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId)
        }
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        return PendingIntent.getBroadcast(context, requestCode * 100 + widgetId, intent, flags)
    }

    private fun queryPI(context: Context, widgetId: Int): PendingIntent {
        val intent = Intent(context, MainActivity::class.java).apply {
            action = ACTION_QUERY
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        return PendingIntent.getActivity(context, 1000 + widgetId, intent, flags)
    }

    companion object {
        const val ACTION_REFRESH = "ai.neurophone.widget.ACTION_REFRESH"
        const val ACTION_TOGGLE  = "ai.neurophone.widget.ACTION_TOGGLE"
        const val ACTION_QUERY   = "ai.neurophone.widget.ACTION_QUERY"

        private const val PREFS = "neurophone_widget"
        const val KEY_RUNNING = "running"
        const val KEY_SALIENCE = "salience"
        const val KEY_DESCRIPTION = "description"

        fun prefs(context: Context): SharedPreferences =
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

        /** Push fresh neural state from the service into the widget. */
        fun publishState(
            context: Context,
            running: Boolean,
            salience: Float,
            description: String?
        ) {
            prefs(context).edit()
                .putBoolean(KEY_RUNNING, running)
                .putFloat(KEY_SALIENCE, salience)
                .putString(KEY_DESCRIPTION, description)
                .apply()
            val mgr = AppWidgetManager.getInstance(context)
            val ids = mgr.getAppWidgetIds(ComponentName(context, NeurophoneAppWidget::class.java))
            if (ids.isNotEmpty()) {
                val intent = Intent(context, NeurophoneAppWidget::class.java).apply {
                    action = ACTION_REFRESH
                    putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, ids)
                }
                context.sendBroadcast(intent)
            }
        }
    }
}
