// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
package ai.neurophone;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

/**
 * Thin {@link BroadcastReceiver} shim (sub-issue #112) that restarts the
 * foreground {@link NeurophoneService} after boot.
 *
 * <p>Replaces the legacy Kotlin {@code BootReceiver.kt}. gossamer has no
 * {@code BroadcastReceiver} primitive at all (verified: zero foundation in
 * {@code hyperpolymath/gossamer} for this Android surface), so this is a
 * from-scratch minimal shim &mdash; not an adaptation of anything gossamer
 * provides.
 *
 * <p>Carries no policy: it does not persist or consult a "was running before
 * reboot" flag (the legacy Kotlin service delegated that decision to
 * SharedPreferences written by the now-removed widget code). Every boot,
 * unconditionally, it starts the service; the service's own {@code onCreate}
 * decides whether {@link NativeLib#init}/{@link NativeLib#start} succeed.
 * TODO(#83): if a persisted autostart preference is wanted, it should be
 * read via a new JNI accessor into {@code crates/neurophone-android}, not
 * Android SharedPreferences read here.
 */
public final class BootReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
        if (context == null || intent == null) {
            return;
        }
        final String action = intent.getAction();
        if (!Intent.ACTION_BOOT_COMPLETED.equals(action)
                && !Intent.ACTION_LOCKED_BOOT_COMPLETED.equals(action)) {
            return;
        }
        final Intent serviceIntent = new Intent(context, NeurophoneService.class);
        context.startForegroundService(serviceIntent);
    }
}
