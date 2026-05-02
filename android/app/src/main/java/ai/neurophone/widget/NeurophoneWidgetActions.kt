// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
package ai.neurophone.widget

import android.appwidget.AppWidgetManager
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent

/**
 * Lightweight broadcast receiver dispatched by various non-widget callers
 * (the foreground service, the boot receiver, share-intent handler) to
 * push state into the widget without owning a reference to it.
 */
class NeurophoneWidgetActions : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            ACTION_PUBLISH_STATE -> {
                val running = intent.getBooleanExtra(EXTRA_RUNNING, false)
                val salience = intent.getFloatExtra(EXTRA_SALIENCE, 0f)
                val description = intent.getStringExtra(EXTRA_DESCRIPTION)
                NeurophoneAppWidget.publishState(context, running, salience, description)
            }
            ACTION_FORCE_REFRESH -> {
                val mgr = AppWidgetManager.getInstance(context)
                val ids = mgr.getAppWidgetIds(ComponentName(context, NeurophoneAppWidget::class.java))
                if (ids.isNotEmpty()) {
                    val refresh = Intent(context, NeurophoneAppWidget::class.java).apply {
                        action = NeurophoneAppWidget.ACTION_REFRESH
                        putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, ids)
                    }
                    context.sendBroadcast(refresh)
                }
            }
        }
    }

    companion object {
        const val ACTION_PUBLISH_STATE = "ai.neurophone.widget.PUBLISH_STATE"
        const val ACTION_FORCE_REFRESH = "ai.neurophone.widget.FORCE_REFRESH"
        const val EXTRA_RUNNING = "running"
        const val EXTRA_SALIENCE = "salience"
        const val EXTRA_DESCRIPTION = "description"
    }
}
