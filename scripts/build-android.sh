#!/bin/bash
# Build script for NeuroPhone Android native libraries
# Cross-compiles Rust code to Android targets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANDROID_APP="$PROJECT_ROOT/android/app"
JNI_LIBS="$ANDROID_APP/src/main/jniLibs"

# Android targets
TARGETS=(
    "aarch64-linux-android"   # arm64-v8a (Oppo Reno 13)
    "armv7-linux-androideabi" # armeabi-v7a
    "x86_64-linux-android"    # x86_64 (emulator)
)

# ABI mapping
declare -A ABI_MAP=(
    ["aarch64-linux-android"]="arm64-v8a"
    ["armv7-linux-androideabi"]="armeabi-v7a"
    ["x86_64-linux-android"]="x86_64"
)

echo "=== NeuroPhone Android Build ==="
echo "Project root: $PROJECT_ROOT"

# Check for Android NDK
if [ -z "$ANDROID_NDK_HOME" ]; then
    # Try common locations
    if [ -d "$HOME/Android/Sdk/ndk" ]; then
        ANDROID_NDK_HOME=$(ls -d "$HOME/Android/Sdk/ndk"/* 2>/dev/null | tail -n1)
    elif [ -d "/opt/android-ndk" ]; then
        ANDROID_NDK_HOME="/opt/android-ndk"
    fi
fi

if [ -z "$ANDROID_NDK_HOME" ]; then
    echo "Warning: ANDROID_NDK_HOME not set. Please set it to your NDK location."
    echo "Example: export ANDROID_NDK_HOME=\$HOME/Android/Sdk/ndk/26.1.10909125"
    echo ""
    echo "For now, building for host platform only..."

    # Build for host
    echo "Building neurophone-core for host..."
    cd "$PROJECT_ROOT"
    cargo build --release -p neurophone-core

    echo ""
    echo "Host build complete. For Android, please install NDK and set ANDROID_NDK_HOME."
    exit 0
fi

echo "Using NDK: $ANDROID_NDK_HOME"

# Install Rust targets if needed
for target in "${TARGETS[@]}"; do
    if ! rustup target list --installed | grep -q "$target"; then
        echo "Installing Rust target: $target"
        rustup target add "$target"
    fi
done

# Configure cargo for Android
CARGO_CONFIG="$PROJECT_ROOT/.cargo/config.toml"
mkdir -p "$(dirname "$CARGO_CONFIG")"

cat > "$CARGO_CONFIG" << EOF
[target.aarch64-linux-android]
linker = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang"

[target.armv7-linux-androideabi]
linker = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi24-clang"

[target.x86_64-linux-android]
linker = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android24-clang"
EOF

echo "Cargo config created at $CARGO_CONFIG"

# Create jniLibs directories
for target in "${TARGETS[@]}"; do
    abi="${ABI_MAP[$target]}"
    mkdir -p "$JNI_LIBS/$abi"
done

# Build for each target
cd "$PROJECT_ROOT"
for target in "${TARGETS[@]}"; do
    abi="${ABI_MAP[$target]}"
    echo ""
    echo "=== Building for $target ($abi) ==="

    cargo build --release --target "$target" -p neurophone-android

    # Copy library
    src="$PROJECT_ROOT/target/$target/release/libneurophone_android.so"
    dst="$JNI_LIBS/$abi/libneurophone_android.so"

    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "Copied: $dst"

        # Strip debug symbols for smaller size
        "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" "$dst" 2>/dev/null || true
    else
        echo "Warning: Library not found at $src"
    fi
done

echo ""
echo "=== Build Complete ==="
echo "Native libraries copied to: $JNI_LIBS"
ls -la "$JNI_LIBS"/*/*.so 2>/dev/null || echo "No .so files found"

echo ""
echo "Next steps:"
echo "1. Open android/ in Android Studio"
echo "2. Build and run on device"
echo ""
echo "For Oppo Reno 13, primary target is arm64-v8a"
