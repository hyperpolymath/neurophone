#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# must-check — portable, runnable enforcer for MUST.contractile invariants.
#
# MUST.contractile [enforcement] promises "quality.yml runs must-check on every
# PR". The estate's declarative form is a Nickel k9 validator
# (contractiles/k9/must-check.k9.ncl) executed by the k9 `contractile` runner;
# that runner (and its _base.ncl / template-hunt.k9.ncl bases) is not yet wired
# into this repo, so this script is the portable enforcer that runs today with
# nothing but bash + coreutils. It checks the mechanically-verifiable MUST
# invariants; it never mutates.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 2

fail=0
err() { echo "::error::MUST violation: $*"; fail=1; }

# --- License / structure ---
[ -f LICENSE ] || err "LICENSE file missing"
grep -q 'Mozilla Public License\|MPL-2.0' LICENSE || err "LICENSE is not MPL-2.0"
[ -f 0-AI-MANIFEST.a2ml ] || err "0-AI-MANIFEST.a2ml missing"
[ -d .machine_readable ] || err ".machine_readable/ directory missing"

# --- No SCM state files loose in repo root ---
for f in STATE.a2ml META.a2ml ECOSYSTEM.a2ml AGENTIC.a2ml NEUROSYM.a2ml PLAYBOOK.a2ml; do
  [ -f "$f" ] && err "SCM file $f in repo root (must live under .machine_readable/)"
done

# --- All GitHub Actions SHA-pinned (no @vN / @branch) ---
if grep -rnE "uses:[[:space:]]*[^#]*@v[0-9]" .github/workflows/*.yml 2>/dev/null | grep -vE "^[[:space:]]*#" ; then
  err "unpinned GitHub Action(s) — pin to a full 40-char commit SHA"
fi

# --- No proof escape hatches in proof/source trees ---
if grep -rnE "\b(believe_me|assert_total|unsafeCoerce|Obj\.magic)\b|^[[:space:]]*(sorry|Admitted)\b" \
     --include=*.idr --include=*.idr2 --include=*.lean --include=*.v \
     --include=*.hs --include=*.ml proofs crates 2>/dev/null ; then
  err "proof escape hatch present (believe_me/assert_total/sorry/Admitted/unsafeCoerce/Obj.magic)"
fi

# --- No new banned-language source (TypeScript / Python / Go) ---
banned=$(find . -path ./target -prune -o \
         \( -name "*.ts" -o -name "*.py" -o -name "*.go" \) -print 2>/dev/null \
         | grep -vE "\.d\.ts$|/target/|/node_modules/")
if [ -n "$banned" ]; then
  echo "$banned"
  err "banned-language source present (TypeScript/Python/Go)"
fi

# --- No hardcoded absolute home/mnt paths in source ---
if grep -rnE "(/home/[a-z]|/mnt/|/var/mnt/)" \
     --include=*.rs --include=*.sh --include=*.toml crates .github 2>/dev/null ; then
  err "hardcoded absolute path (/home|/mnt|/var/mnt) in source"
fi

# --- SPDX header on every Rust source file ---
missing_spdx=""
while IFS= read -r f; do
  head -3 "$f" | grep -q "SPDX-License-Identifier" || missing_spdx="$missing_spdx $f"
done < <(find crates -name "*.rs" -not -path "*/target/*" 2>/dev/null)
[ -n "$missing_spdx" ] && err "Rust source missing SPDX header:$missing_spdx"

if [ "$fail" = "0" ]; then
  echo "must-check: all mechanically-verifiable MUST invariants satisfied."
fi
exit "$fail"
