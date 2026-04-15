#!/usr/bin/env bash
#
# Run the WebAssembly spectest with the arm32 backend emitting Thumb-2
# under QEMU user-mode emulation inside Colima. Companion to
# `armv7a-spectest.sh` — same target triple and QEMU setup, but the
# binary is built with `--features thumb2-test` so `sf_arm32_isa_thumb`
# is set and the JIT emits Thumb-2 instructions via `enc_t2.rs`.
#
# Usage:
#   ./scripts/armv7m-spectest.sh [-- extra spectest args...]
#
# Examples:
#   ./scripts/armv7m-spectest.sh
#   ./scripts/armv7m-spectest.sh -- if
#   ./scripts/armv7m-spectest.sh -- --log-level debug i32

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=armv7-unknown-linux-musleabihf
SPECTEST_BIN="$REPO_ROOT/target/$TARGET/debug/sf-nano-spectest"
TESTSUITE_DIR="$REPO_ROOT/target/webassembly-testsuite"

# Parse arguments: everything after "--" is forwarded to spectest
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [-- extra spectest args...]"; exit 1 ;;
    esac
done

cleanup() {
    if [[ "${COLIMA_STARTED:-}" = "1" ]]; then
        echo "[armv7m-spectest] Stopping Colima (we started it)..."
        colima stop 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- Prerequisites ---

for cmd in colima qemu-system-arm cargo; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first." >&2
        exit 1
    fi
done

# --- Start Colima if needed ---

COLIMA_STARTED=0
if ! colima status &>/dev/null; then
    echo "[armv7m-spectest] Starting Colima..."
    colima start
    COLIMA_STARTED=1
fi

# Ensure qemu-user-static is installed inside the VM
if ! colima ssh -- which qemu-arm-static &>/dev/null; then
    echo "[armv7m-spectest] Installing qemu-user-static in Colima VM..."
    colima ssh -- sudo apt-get update -qq
    colima ssh -- sudo apt-get install -y -qq qemu-user-static
fi

# --- Build ---

echo "[armv7m-spectest] Cross-compiling spectest (Thumb-2 JIT) for ARMv7..."
cargo build --target "$TARGET" -p sf-nano-spectest --no-default-features --features jit,thumb2-test

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

echo "[armv7m-spectest] Running spectest under QEMU (Thumb-2)..."
echo
colima ssh -- env TESTSUITE_DIR="$TESTSUITE_DIR" \
    qemu-arm-static -cpu cortex-a15 "$SPECTEST_BIN" --backend native ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
