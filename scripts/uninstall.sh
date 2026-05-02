#!/usr/bin/env bash
# SPDX-License-Identifier: PMPL-1.0-or-later
# SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
#
# uninstall.sh — Remove NeuroPhone (CLI + APK + data) from the device.
#
# In Termux:    bash uninstall.sh
# Over ADB:     bash uninstall.sh --adb [--serial XYZ]
set -euo pipefail

MODE="local"
SERIAL=""
PKG="ai.neurophone"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/neurophone}"
BIN="${BIN_DIR:-$HOME/.local/bin}/neurophone"

while [ $# -gt 0 ]; do
    case "$1" in
        --adb) MODE="adb"; shift ;;
        --serial) SERIAL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ "$MODE" = "adb" ]; then
    ADB="adb"; [ -n "$SERIAL" ] && ADB="adb -s $SERIAL"
    $ADB shell pm uninstall "$PKG" || true
    $ADB shell rm -f /data/local/tmp/llama-3.2-1b-q4_k_m.gguf || true
    echo "ADB uninstall done."
    exit 0
fi

# Local (Termux) mode
rm -f "$BIN"
rm -rf "$INSTALL_DIR"
rm -rf "$HOME/.config/neurophone"
rm -rf "$HOME/.local/share/neurophone"
echo "Local uninstall done."
