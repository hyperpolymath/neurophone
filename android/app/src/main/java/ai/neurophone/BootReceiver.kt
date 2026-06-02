// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
package ai.neurophone

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import ai.neurophone.widget.NeurophoneAppWidget

/**
 * Restart the foreground service after device reboot, but only if the user
 * had it running before (we honour the persisted widget state).
 */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action ?: return
        if (action != Intent.ACTION_BOOT_COMPLETED && action != Intent.ACTION_LOCKED_BOOT_COMPLETED) {
            return
        }
        val wasRunning = NeurophoneAppWidget.prefs(context).getBoolean(NeurophoneAppWidget.KEY_RUNNING, false)
        if (wasRunning) {
            context.startForegroundService(Intent(context, NeurophoneService::class.java))
        }
    }
}
