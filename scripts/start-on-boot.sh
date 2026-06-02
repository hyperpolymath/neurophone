#!/data/data/com.termux/files/usr/bin/bash
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
#
# start-on-boot.sh — Termux:Boot hook to keep the CLI side of NeuroPhone
# alive across reboots.
#
# Install:
#   1. Install the Termux:Boot add-on (F-Droid or Termux-Boot APK).
#   2. mkdir -p ~/.termux/boot
#   3. cp scripts/start-on-boot.sh ~/.termux/boot/neurophone
#   4. chmod +x ~/.termux/boot/neurophone
#
# The Android side of NeuroPhone (APK + foreground service) handles its
# own boot via BOOT_COMPLETED — this script is only for the Termux CLI.
set -euo pipefail

LOG="$HOME/.local/share/neurophone/boot.log"
mkdir -p "$(dirname "$LOG")"

# Acquire wakelock so Android doesn't kill us before we hand off.
termux-wake-lock || true

if command -v neurophone >/dev/null 2>&1; then
    echo "$(date -Iseconds) starting neurophone" >> "$LOG"
    nohup neurophone >> "$LOG" 2>&1 &
fi
