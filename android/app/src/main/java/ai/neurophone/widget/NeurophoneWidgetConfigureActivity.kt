// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
package ai.neurophone.widget

import android.app.Activity
import android.appwidget.AppWidgetManager
import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Drop-in widget configuration activity.
 *
 * Built programmatically (no extra layout file) to keep the widget self-contained.
 * Asks one question: local-only mode? — persisted to SharedPreferences.
 */
class NeurophoneWidgetConfigureActivity : Activity() {

    private var widgetId = AppWidgetManager.INVALID_APPWIDGET_ID

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // If user backs out before finishing, the widget is removed.
        setResult(RESULT_CANCELED)

        widgetId = intent?.extras?.getInt(
            AppWidgetManager.EXTRA_APPWIDGET_ID,
            AppWidgetManager.INVALID_APPWIDGET_ID
        ) ?: AppWidgetManager.INVALID_APPWIDGET_ID

        if (widgetId == AppWidgetManager.INVALID_APPWIDGET_ID) {
            finish(); return
        }

        val prefs = NeurophoneAppWidget.prefs(this)
        val titleText = TextView(this).apply {
            text = getString(ai.neurophone.R.string.widget_configure_title)
            textSize = 18f
            setPadding(24, 24, 24, 8)
        }
        val localOnly = CheckBox(this).apply {
            text = getString(ai.neurophone.R.string.widget_configure_local_only)
            isChecked = prefs.getBoolean(KEY_LOCAL_ONLY, true)
        }
        val ok = Button(this).apply {
            text = getString(ai.neurophone.R.string.widget_configure_done)
            setOnClickListener {
                prefs.edit().putBoolean(KEY_LOCAL_ONLY, localOnly.isChecked).apply()
                val result = Intent().putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId)
                setResult(RESULT_OK, result)
                finish()
            }
        }

        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 16, 16, 16)
            addView(titleText)
            addView(localOnly)
            addView(ok)
        })
    }

    companion object {
        const val KEY_LOCAL_ONLY = "local_only"
    }
}
