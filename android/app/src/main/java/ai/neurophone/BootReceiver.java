// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
package ai.neurophone;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

/**
 * Thin {@link BroadcastReceiver} shim that restarts the neurophone foreground
 * {@link NeurophoneService} after the device finishes booting.
 *
 * <p>This is a deliberately minimal, hand-written Java shim that replaces the
 * previous Kotlin {@code BootReceiver}. It is part of the Android Kotlin to
 * Rust/Gossamer migration (epic #83). The receiver contains <em>no</em>
 * business logic: any decision about whether the service should actually run
 * (e.g. honouring persisted "was running" state) belongs in the Rust JNI layer
 * ({@code crates/neurophone-android}) and is reached through the service start
 * path, not here.
 *
 * <p>Hand-written Java is permitted only under {@code android/} via the
 * {@code .hypatia-baseline.json} exemption for the in-flight Gossamer
 * migration.
 *
 * <p>TODO(#83): once the Rust JNI boot-policy entrypoint lands, delegate the
 * "should we restart?" decision to {@code crates/neurophone-android} rather
 * than unconditionally starting the service.
 * <p>TODO(#83 rebase): depends on sub-PRs #4 (NativeLib to Rust) and #5
 * (Service shim); re-point the {@code NeurophoneService} reference if those
 * sub-PRs rename or relocate the service entrypoint.
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
        // Thin shim: start the foreground service from sub-PR #5. All runtime
        // policy and inference lives behind the Rust JNI in NeurophoneService.
        final Intent serviceIntent = new Intent(context, NeurophoneService.class);
        context.startForegroundService(serviceIntent);
    }
}
