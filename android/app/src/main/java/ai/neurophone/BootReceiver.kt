// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
package ai.neurophone

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
/**
 * Restart the foreground service after device reboot, but only if the user
 * had it running before.
 *
 * TODO(#83 rebase): the "was running" flag used to live in the widget's
 *  SharedPreferences (NeurophoneAppWidget.prefs / KEY_RUNNING), which was
 *  removed when the widget was ported to a stateless Java shim that reads the
 *  Rust core directly. Persisting the desired-run flag is now a core concern;
 *  sub-PR #3/#4/#5 is expected to expose it (e.g. NativeLib.shouldAutostart()).
 *  Until then, conservatively do nothing on boot rather than force-start.
 */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action ?: return
        if (action != Intent.ACTION_BOOT_COMPLETED && action != Intent.ACTION_LOCKED_BOOT_COMPLETED) {
            return
        }
        // TODO(#83 rebase): wire to core-persisted autostart flag once available.
    }
}
