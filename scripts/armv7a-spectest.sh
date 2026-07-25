#!/usr/bin/env bash
#
# Deprecated direct entry point. Prefer `python3 scripts/check.py`; this script is kept as a low-level helper.
#
# Run the ARMv7 spectest under QEMU user-mode emulation inside Colima.
#
# Usage:
#   ./scripts/armv7-spectest.sh [-- extra spectest args...]
#
# Examples:
#   ./scripts/armv7-spectest.sh
#   ./scripts/armv7-spectest.sh -- if
#   ./scripts/armv7-spectest.sh -- --log-level debug i32

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=armv7-unknown-linux-musleabihf
SPECTEST_BIN="$REPO_ROOT/target/$TARGET/debug/sf-nano-spectest"
TESTSUITE_DIR="${TESTSUITE_DIR:-$REPO_ROOT/target/webassembly-testsuite}"

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
        echo "[armv7-spectest] Stopping Colima (we started it)..."
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
    echo "[armv7-spectest] Starting Colima..."
    colima start
    COLIMA_STARTED=1
fi

# Ensure qemu-user-static is installed inside the VM
if ! colima ssh -- which qemu-arm-static &>/dev/null; then
    echo "[armv7-spectest] Installing qemu-user-static in Colima VM..."
    colima ssh -- sudo apt-get update -qq
    colima ssh -- sudo apt-get install -y -qq qemu-user-static
fi

# --- Build ---

echo "[armv7-spectest] Cross-compiling spectest for ARMv7..."
cargo build --target "$TARGET" -p sf-nano-spectest --no-default-features --features jit

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

echo "[armv7-spectest] Running spectest under QEMU..."
echo
colima ssh -- env TESTSUITE_DIR="$TESTSUITE_DIR" \
    qemu-arm-static -cpu cortex-a15 "$SPECTEST_BIN" --backend native ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
