#!/bin/bash
# Setup script for NeuroPhone development environment

set -e

echo "=== NeuroPhone Development Setup ==="

# Check Rust
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "Rust version: $(rustc --version)"

# Install required Rust targets for Android
echo "Installing Android Rust targets..."
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add x86_64-linux-android

# Check for required tools
echo ""
echo "Checking required tools..."

TOOLS_OK=true

if ! command -v cargo &> /dev/null; then
    echo "  [X] cargo - NOT FOUND"
    TOOLS_OK=false
else
    echo "  [OK] cargo - $(cargo --version)"
fi

if [ -n "$ANDROID_NDK_HOME" ] && [ -d "$ANDROID_NDK_HOME" ]; then
    echo "  [OK] Android NDK - $ANDROID_NDK_HOME"
else
    echo "  [!] Android NDK - NOT SET (needed for Android builds)"
fi

if [ -n "$ANDROID_HOME" ] && [ -d "$ANDROID_HOME" ]; then
    echo "  [OK] Android SDK - $ANDROID_HOME"
else
    echo "  [!] Android SDK - NOT SET"
fi

# Download Llama model (optional)
echo ""
echo "For local LLM support, download a Llama 3.2 model:"
echo "  Recommended for mobile: llama-3.2-1b-instruct-q4_k_m.gguf"
echo ""
echo "  Download from Hugging Face:"
echo "  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
echo ""
echo "  Place in: /data/local/tmp/ on the device"
echo "  Or update LlmConfig.model_path in your app"

# Build verification
echo ""
echo "Verifying Rust build..."
cd "$(dirname "$0")/.."

if cargo check --workspace 2>/dev/null; then
    echo "[OK] Rust workspace compiles successfully"
else
    echo "[!] Rust workspace has compilation errors"
    echo "    Run 'cargo check --workspace' for details"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Set ANDROID_NDK_HOME if building for Android"
echo "  2. Run './scripts/build-android.sh' to build native libs"
echo "  3. Open 'android/' in Android Studio"
echo "  4. Configure Claude API key in app or environment"
echo ""
