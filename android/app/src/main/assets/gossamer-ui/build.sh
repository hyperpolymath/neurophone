#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
#
# Build wiring: compile the AffineScript pure-logic module (src/logic.affine)
# to a Deno-ESM module (dist/logic.deno.js) via the real `affinescript`
# compiler:
#   affinescript check   src/logic.affine
#   affinescript compile --deno-esm -o dist/logic.deno.js src/logic.affine
#
# Unlike an earlier draft of this UI (neurophone commit a3487cc, since
# deleted, which assumed the toolchain wasn't vendored anywhere and shipped
# a hand-written "generated-output stub"), this repo's authoring environment
# had a real `affinescript` (hyperpolymath/affinescript) on PATH, so
# dist/logic.deno.js committed alongside this script IS genuine compiler
# output, not a stub — see android/README.adoc "AffineScript verification"
# for the exact transcript. This script still degrades gracefully if
# `affinescript` is absent (e.g. a CI runner that hasn't vendored it): it
# leaves the already-committed dist/logic.deno.js in place rather than
# failing the build.
#
# dist/ui.mjs is NOT produced by this script — it is a hand-written DOM +
# NeurophoneBridge harness (see its own file header for why: no typed DOM
# binding exists in this compiler yet, and extern fn lowering can't express
# `window.NeurophoneBridge.method(...)`-shaped calls). Only dist/logic.deno.js
# is compiler output.
#
# Usage:
#   bash build.sh              # compile src/logic.affine -> dist/logic.deno.js, then run the harness
#   bash build.sh --check-only # type check only (no compile, no harness run)
#   bash build.sh --fmt        # format src/logic.affine in place — NOTE: as
#                              #   tested in this session, the installed
#                              #   `affinescript fmt` exits 125
#                              #   ("Code formatting not yet implemented");
#                              #   this is an upstream compiler gap, not a
#                              #   bug here. Left wired for when it lands.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src/logic.affine"
DIST_DIR="$HERE/dist"
OUT="$DIST_DIR/logic.deno.js"
HARNESS="$DIST_DIR/logic.harness.mjs"

mode="build"
case "${1:-}" in
  --check-only) mode="check" ;;
  --fmt)        mode="fmt" ;;
  "")           mode="build" ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

if ! command -v affinescript >/dev/null 2>&1; then
  echo "[build.sh] 'affinescript' compiler not found on PATH."
  echo "[build.sh] Leaving the already-committed $OUT in place (not regenerating)."
  [ -f "$OUT" ] || { echo "[build.sh] ERROR: $OUT missing and no compiler to produce it." >&2; exit 1; }
  exit 0
fi

case "$mode" in
  check)
    affinescript check "$SRC"
    ;;
  fmt)
    affinescript fmt "$SRC"
    ;;
  build)
    affinescript check "$SRC"
    mkdir -p "$DIST_DIR"
    affinescript compile --deno-esm -o "$OUT" "$SRC"
    echo "[build.sh] Wrote $OUT"
    if command -v deno >/dev/null 2>&1; then
      deno run --allow-read "$HARNESS"
    else
      echo "[build.sh] 'deno' not found — skipping $HARNESS regression run."
    fi
    ;;
esac
