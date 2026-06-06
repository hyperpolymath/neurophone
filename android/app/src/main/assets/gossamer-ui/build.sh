#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
#
# Build wiring: compile the AffineScript Gossamer UI (src/*.affine) to a
# Deno-ESM module (dist/ui.mjs) that the Android webview loads via index.html.
#
# Pipeline (per the AffineScript toolchain — OCaml/Dune `affinescript` CLI):
#   affinescript check   src/*.affine      # affine/ownership + type check
#   affinescript fmt     src/*.affine      # formatting (with --fmt)
#   affinescript compile --target deno-esm -o dist/ui.mjs src/ui.affine
#
# TODO(#83 rebase): the `affinescript` compiler is NOT yet vendored in this repo
#   or CI. Until it is, this script DOES NOT regenerate dist/ui.mjs; the
#   committed hand-written stub (dist/ui.mjs) stands in. When the toolchain
#   lands: drop the stub, flip USE_STUB=0, and wire `deno task build:ui` into CI.
#
# Usage:
#   bash build.sh              # build dist/ui.mjs from src/*.affine
#   bash build.sh --check-only # type/ownership check only
#   bash build.sh --fmt        # format src/*.affine in place
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE/src"
DIST_DIR="$HERE/dist"
OUT="$DIST_DIR/ui.mjs"
ENTRY="$SRC_DIR/ui.affine"

# TODO(#83): set to 0 once the affinescript compiler is available in CI/dev.
USE_STUB=1

mode="build"
case "${1:-}" in
  --check-only) mode="check" ;;
  --fmt)        mode="fmt" ;;
  "")           mode="build" ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

if ! command -v affinescript >/dev/null 2>&1; then
  echo "[build.sh] 'affinescript' compiler not found on PATH."
  echo "[build.sh] TODO(#83): vendor/pin the AffineScript toolchain (OCaml/Dune)."
  if [ "$USE_STUB" = "1" ]; then
    echo "[build.sh] Using committed stub: $OUT (no regeneration)."
    [ -f "$OUT" ] || { echo "[build.sh] ERROR: stub $OUT missing." >&2; exit 1; }
    exit 0
  fi
  exit 1
fi

case "$mode" in
  check)
    affinescript check "$SRC_DIR"/*.affine
    ;;
  fmt)
    affinescript fmt "$SRC_DIR"/*.affine
    ;;
  build)
    affinescript check "$SRC_DIR"/*.affine
    mkdir -p "$DIST_DIR"
    affinescript compile --target deno-esm -o "$OUT" "$ENTRY"
    echo "[build.sh] Wrote $OUT"
    ;;
esac
