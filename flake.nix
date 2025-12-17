# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
# flake.nix - Nix Flake (fallback to guix.scm)
# Run: nix develop
{
  description = "neurophone - Neural processing and sensor bridge for Android";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rustfmt" "clippy" ];
          targets = [ "aarch64-linux-android" "armv7-linux-androideabi" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust toolchain
            rustToolchain
            cargo-audit
            cargo-outdated
            cargo-tarpaulin

            # Build essentials
            pkg-config
            openssl

            # Android NDK (for cross-compilation)
            # androidenv.androidPkgs_9_0.ndk-bundle

            # Development tools
            just
            git
          ];

          shellHook = ''
            echo "neurophone development environment"
            echo "Rust: $(rustc --version)"
            echo ""
            echo "Commands:"
            echo "  cargo build          - Build the project"
            echo "  cargo test           - Run tests"
            echo "  cargo clippy         - Run lints"
            echo "  cargo audit          - Security audit"
            echo "  just                 - See available tasks"
          '';

          RUST_BACKTRACE = 1;
          RUST_LOG = "debug";
        };

        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "neurophone";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;

          meta = with pkgs.lib; {
            description = "Neural processing and sensor bridge";
            homepage = "https://github.com/hyperpolymath/neurophone";
            license = licenses.agpl3Plus;
            maintainers = [ ];
          };
        };
      }
    );
}
