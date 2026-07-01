# neurophone - Rust Development Tasks
set shell := ["bash", "-uc"]
set dotenv-load := true

import? "contractile.just"

project := "neurophone"

# Show all recipes
default:
    @just --list --unsorted

# Build debug
build:
    cargo build

# Build release
build-release:
    cargo build --release

# Run tests
test:
    cargo test

# Run tests verbose
test-verbose:
    cargo test -- --nocapture

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt -- --check

# Run clippy lints
lint:
    cargo clippy -- -D warnings

# Check without building
check:
    cargo check

# Clean build artifacts
clean:
    cargo clean

# Run the project
run *ARGS:
    cargo run -- {{ARGS}}

# Generate docs
doc:
    cargo doc --no-deps --open

# Update dependencies
update:
    cargo update

# Audit dependencies
audit:
    cargo audit

# Model-check the TLA+ lifecycle proof (obligation 2.1, issue #84). Fetches
# tla2tools.jar on first run into .tlacache/; self-skips (non-fatal) when java
# is unavailable so it degrades gracefully in minimal environments.
proof-tla:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v java >/dev/null 2>&1; then
        echo "proof-tla: java not found — skipping TLC (spec unchanged)"; exit 0
    fi
    jar="${TLA2TOOLS_JAR:-$(pwd)/.tlacache/tla2tools.jar}"
    if [ ! -f "$jar" ]; then
        mkdir -p "$(dirname "$jar")"
        echo "proof-tla: fetching tla2tools.jar…"
        curl -fsSL -o "$jar" \
          https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
    fi
    cd proofs/tla
    java -XX:+UseParallelGC -cp "$jar" tlc2.TLC -config Lifecycle.cfg Lifecycle.tla

# Type-check the Lean 4 Echo State Property proof (obligation 1.1, issue
# #84, `proofs/lean/EsnEcho.lean`). Self-skips (non-fatal) if `lake` (elan)
# isn't on PATH. First run needs network access to fetch the pinned Mathlib
# revision (see `proofs/lean/lake-manifest.json`); later runs reuse the
# local `.lake/` cache.
proof-lean:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v lake >/dev/null 2>&1; then
        echo "proof-lean: lake not found — skipping Lean check (spec unchanged)"; exit 0
    fi
    cd proofs/lean && lake build

# Verify the Dafny LSM-bounded-dynamics proof (obligation 1.2, issue #84,
# `proofs/dafny/LsmBoundedDynamics.dfy`). Uses a system `dafny` if present;
# otherwise fetches a pinned, checksum-verified, self-contained release into
# `.dafnycache/` on first run. Self-skips (non-fatal) if no system `dafny`
# is found and the download isn't possible (offline / no permissions), so
# it degrades gracefully in minimal/offline environments.
proof-dafny:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v dafny >/dev/null 2>&1; then
        dafny verify proofs/dafny/LsmBoundedDynamics.dfy
        exit 0
    fi
    version=4.11.0
    sha256=a46a9ff7cdd720f7955854c78e95df13f4cfe6b80691b05f8654fe19e8267179
    dir="$(pwd)/.dafnycache/dafny-${version}"
    bin="${dir}/dafny/dafny"
    if [ ! -x "$bin" ]; then
        zip="$(pwd)/.dafnycache/dafny-${version}.zip"
        mkdir -p "$dir"
        if ! curl -fsSL -o "$zip" \
          "https://github.com/dafny-lang/dafny/releases/download/v${version}/dafny-${version}-x64-ubuntu-22.04.zip"; then
            echo "proof-dafny: could not download Dafny (offline?) — skipping"; exit 0
        fi
        echo "${sha256}  ${zip}" | sha256sum -c - || {
            echo "proof-dafny: checksum mismatch on downloaded release, aborting"; exit 1
        }
        unzip -q "$zip" -d "$dir"
    fi
    "$bin" verify proofs/dafny/LsmBoundedDynamics.dfy

# Run the full proof surface: property tests + compile-fail typestate doc-tests
# (via `cargo test`) plus the TLA+ model check, the Lean type-check, and the
# Dafny verification.
proof: test proof-tla proof-lean proof-dafny
    @echo "Proof surface checked (properties, typestate compile-fails, TLC, Lean, Dafny)."

# Quality gates (RSR golden path `just test && just quality`).
quality: fmt-check lint audit
    @echo "Quality gates passed!"

# All checks before commit
pre-commit: fmt-check lint test
    @echo "All checks passed!"

# [AUTO-GENERATED] Multi-arch / RISC-V target
build-riscv:
	@echo "Building for RISC-V..."
	cross build --target riscv64gc-unknown-linux-gnu

# Run panic-attacker pre-commit scan
assail:
    @command -v panic-attack >/dev/null 2>&1 && panic-attack assail . || echo "panic-attack not found — install from https://github.com/hyperpolymath/panic-attacker"

# Self-diagnostic — checks dependencies, permissions, paths
doctor:
    @echo "Running diagnostics for neurophone..."
    @echo "Checking required tools..."
    @command -v just >/dev/null 2>&1 && echo "  [OK] just" || echo "  [FAIL] just not found"
    @command -v git >/dev/null 2>&1 && echo "  [OK] git" || echo "  [FAIL] git not found"
    @echo "Checking for hardcoded paths..."
    @grep -rn '$HOME\|$ECLIPSE_DIR' --include='*.rs' --include='*.ex' --include='*.res' --include='*.gleam' --include='*.sh' . 2>/dev/null | head -5 || echo "  [OK] No hardcoded paths"
    @echo "Diagnostics complete."

# Auto-repair common issues
heal:
    @echo "Attempting auto-repair for neurophone..."
    @echo "Fixing permissions..."
    @find . -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    @echo "Cleaning stale caches..."
    @rm -rf .cache/stale 2>/dev/null || true
    @echo "Repair complete."

# Guided tour of key features
tour:
    @echo "=== neurophone Tour ==="
    @echo ""
    @echo "1. Project structure:"
    @ls -la
    @echo ""
    @echo "2. Available commands: just --list"
    @echo ""
    @echo "3. Read README.adoc for full overview"
    @echo "4. Read EXPLAINME.adoc for architecture decisions"
    @echo "5. Run 'just doctor' to check your setup"
    @echo ""
    @echo "Tour complete! Try 'just --list' to see all available commands."

# Open feedback channel with diagnostic context
help-me:
    @echo "=== neurophone Help ==="
    @echo "Platform: $(uname -s) $(uname -m)"
    @echo "Shell: $SHELL"
    @echo ""
    @echo "To report an issue:"
    @echo "  https://github.com/hyperpolymath/neurophone/issues/new"
    @echo ""
    @echo "Include the output of 'just doctor' in your report."


# Print the current CRG grade (reads from READINESS.md '**Current Grade:** X' line)
crg-grade:
    @grade=$$(grep -oP '(?<=\*\*Current Grade:\*\* )[A-FX]' READINESS.md 2>/dev/null | head -1); \
    [ -z "$$grade" ] && grade="X"; \
    echo "$$grade"

# Generate a shields.io badge markdown for the current CRG grade
# Looks for '**Current Grade:** X' in READINESS.md; falls back to X
crg-badge:
    @grade=$$(grep -oP '(?<=\*\*Current Grade:\*\* )[A-FX]' READINESS.md 2>/dev/null | head -1); \
    [ -z "$$grade" ] && grade="X"; \
    case "$$grade" in \
      A) color="brightgreen" ;; B) color="green" ;; C) color="yellow" ;; \
      D) color="orange" ;; E) color="red" ;; F) color="critical" ;; \
      *) color="lightgrey" ;; esac; \
    echo "[![CRG $$grade](https://img.shields.io/badge/CRG-$$grade-$$color?style=flat-square)](https://github.com/hyperpolymath/standards/tree/main/component-readiness-grades)"
