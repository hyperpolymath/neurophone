#!/usr/bin/env bash
# SPDX-License-Identifier: PMPL-1.0-or-later
# SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
#
# install-on-phone.sh — One-shot ADB installer for NeuroPhone.
#
# Run from a workstation with `adb` and the phone in USB-debug mode.
# Builds the APK if needed, sideloads it, pushes the LLM model, grants
# runtime permissions, and starts the foreground service.
#
# Usage:
#   ./scripts/install-on-phone.sh                 # default: arm64
#   ./scripts/install-on-phone.sh --serial XYZ    # target a specific device
#   ./scripts/install-on-phone.sh --no-model      # skip model push
#   ./scripts/install-on-phone.sh --uninstall     # remove the app
set -euo pipefail

ABI="arm64-v8a"
SERIAL=""
PUSH_MODEL=1
UNINSTALL=0
APK_PATH=""

MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_FILE="llama-3.2-1b-q4_k_m.gguf"
DEVICE_MODEL_PATH="/data/local/tmp/${MODEL_FILE}"
PKG="ai.neurophone"

while [ $# -gt 0 ]; do
    case "$1" in
        --serial) SERIAL="$2"; shift 2 ;;
        --abi) ABI="$2"; shift 2 ;;
        --no-model) PUSH_MODEL=0; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        --apk) APK_PATH="$2"; shift 2 ;;
        -h|--help)
            sed -n '5,18p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

ADB="adb"
[ -n "$SERIAL" ] && ADB="adb -s $SERIAL"

require() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1"; exit 1; }; }
require adb

# Ensure exactly one device.
DEVICES=$($ADB devices | awk '/\tdevice$/ {print $1}')
COUNT=$(echo "$DEVICES" | grep -c . || true)
if [ "$COUNT" -lt 1 ]; then
    echo "No device. Plug in phone with USB debugging enabled."; exit 1
fi
if [ "$COUNT" -gt 1 ] && [ -z "$SERIAL" ]; then
    echo "Multiple devices: pass --serial."
    echo "$DEVICES"; exit 1
fi

if [ "$UNINSTALL" -eq 1 ]; then
    $ADB shell pm uninstall "$PKG" || true
    $ADB shell rm -f "$DEVICE_MODEL_PATH" || true
    echo "Uninstalled."
    exit 0
fi

# Locate or build the APK.
if [ -z "$APK_PATH" ]; then
    APK_PATH="android/app/build/outputs/apk/release/app-release.apk"
fi
if [ ! -f "$APK_PATH" ]; then
    echo "APK not found at $APK_PATH — building..."
    pushd "$(git rev-parse --show-toplevel)" >/dev/null
    ./scripts/build-android.sh
    (cd android && ./gradlew :app:assembleRelease --no-daemon)
    popd >/dev/null
fi

# Push model if requested and not already on device.
if [ "$PUSH_MODEL" -eq 1 ]; then
    if ! $ADB shell test -f "$DEVICE_MODEL_PATH" 2>/dev/null; then
        if [ ! -f "models/$MODEL_FILE" ]; then
            echo "Downloading $MODEL_FILE (~700 MB)..."
            mkdir -p models
            curl -fL --retry 3 -o "models/$MODEL_FILE" "$MODEL_URL"
        fi
        echo "Pushing model to device..."
        $ADB push "models/$MODEL_FILE" "$DEVICE_MODEL_PATH"
    else
        echo "Model already present on device — skipping push."
    fi
fi

echo "Installing APK..."
$ADB install -r -g "$APK_PATH"

# Grant runtime permissions explicitly (-g should cover them; belt-and-braces).
for P in BODY_SENSORS HIGH_SAMPLING_RATE_SENSORS POST_NOTIFICATIONS; do
    $ADB shell pm grant "$PKG" "android.permission.$P" 2>/dev/null || true
done

# Start the foreground service.
$ADB shell am start-foreground-service -n "$PKG/.NeurophoneService"

# Hint at adding the widget.
cat <<EOF

Done.

Next steps on the phone:
  1. Long-press the home screen → "Widgets" → drag NeuroPhone widget.
  2. Tap the power icon to start/stop the service.
  3. Tap "Ask NeuroPhone" to query.

Troubleshooting:
  $ADB logcat | grep -i neurophone
  $ADB shell dumpsys notification | grep -A3 neurophone
EOF
