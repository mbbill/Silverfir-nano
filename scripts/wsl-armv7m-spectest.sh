#!/usr/bin/env bash
#
# Run the WebAssembly spectest with the arm32 backend emitting Thumb-2
# under QEMU user-mode emulation, inside WSL2 on Windows. Windows
# counterpart to `armv7m-spectest.sh` (which uses Colima on macOS):
# same target triple, same qemu-arm-static invocation, but no VM
# orchestration — WSL is already a Linux environment.
#
# Usage (from a WSL shell OR from PowerShell via `wsl bash ...`):
#   ./scripts/wsl-armv7m-spectest.sh [-- extra spectest args...]
#
# Examples:
#   ./scripts/wsl-armv7m-spectest.sh
#   ./scripts/wsl-armv7m-spectest.sh -- if
#   ./scripts/wsl-armv7m-spectest.sh -- --log-level debug i32

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=armv7-unknown-linux-musleabihf
SPECTEST_BIN="$REPO_ROOT/target/$TARGET/debug/sf-nano-spectest"
TESTSUITE_DIR="${TESTSUITE_DIR:-$REPO_ROOT/target/webassembly-testsuite}"

# rustc 1.95 ICEs on sf-nano-core (check_mod_deathness panic). 1.92
# builds cleanly; pin until the upstream bug is resolved.
RUSTC_CHANNEL=1.92.0

# Parse arguments: everything after "--" is forwarded to spectest
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [-- extra spectest args...]"; exit 1 ;;
    esac
done

# --- Prerequisites ---

for cmd in qemu-arm-static cargo rustup; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found inside this WSL distro." >&2
        echo "  qemu-arm-static: sudo apt-get install -y qemu-user-static" >&2
        echo "  cargo/rustup:    https://rustup.rs" >&2
        exit 1
    fi
done

if ! rustup toolchain list | grep -q "^$RUSTC_CHANNEL-"; then
    echo "[wsl-armv7m-spectest] Installing Rust $RUSTC_CHANNEL..."
    rustup install "$RUSTC_CHANNEL"
fi

if ! rustup +"$RUSTC_CHANNEL" target list --installed | grep -q "^$TARGET$"; then
    echo "[wsl-armv7m-spectest] Adding $TARGET to $RUSTC_CHANNEL..."
    rustup +"$RUSTC_CHANNEL" target add "$TARGET"
fi

# --- Build ---

echo "[wsl-armv7m-spectest] Cross-compiling spectest (Thumb-2 JIT) for ARMv7 with rustc $RUSTC_CHANNEL..."
cargo +"$RUSTC_CHANNEL" build --target "$TARGET" -p sf-nano-spectest --no-default-features --features jit,thumb2-test

if [[ ! -f "$SPECTEST_BIN" ]]; then
    echo "ERROR: Build succeeded but binary not found at $SPECTEST_BIN" >&2
    exit 1
fi

# --- Testsuite ---

if [[ ! -d "$TESTSUITE_DIR" ]]; then
    echo "ERROR: Testsuite not found at $TESTSUITE_DIR" >&2
    echo "Download it or set TESTSUITE_DIR." >&2
    exit 1
fi

# --- Run ---

echo "[wsl-armv7m-spectest] Running spectest under qemu-arm-static (Thumb-2)..."
echo
env TESTSUITE_DIR="$TESTSUITE_DIR" \
    qemu-arm-static -cpu cortex-a15 "$SPECTEST_BIN" --backend native ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
