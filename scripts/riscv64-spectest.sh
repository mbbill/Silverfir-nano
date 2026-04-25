#!/usr/bin/env bash
#
# Deprecated direct entry point. Prefer `python3 scripts/check.py fast` or
# `python3 scripts/check.py full`; this script is kept as a low-level helper.
#
# Run the RV64GC spectest under QEMU user-mode emulation. On macOS this uses
# qemu-riscv64-static inside Colima; on Linux/WSL it uses qemu-riscv64-static
# directly from PATH.
#
# Usage:
#   ./scripts/riscv64-spectest.sh [-- extra spectest args...]
#
# Examples:
#   ./scripts/riscv64-spectest.sh
#   ./scripts/riscv64-spectest.sh -- f64
#   ./scripts/riscv64-spectest.sh -- select call call_indirect if
#   ./scripts/riscv64-spectest.sh -- --log-level debug f64

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=riscv64gc-unknown-linux-musl
SPECTEST_BIN="$REPO_ROOT/target/$TARGET/debug/sf-nano-spectest"
TESTSUITE_DIR="${TESTSUITE_DIR:-$REPO_ROOT/target/webassembly-testsuite}"
RV64_RUSTFLAGS="-C linker=rust-lld -C target-feature=+crt-static -C link-self-contained=yes -C panic=abort"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [-- extra spectest args...]"; exit 1 ;;
    esac
done

append_rustflags() {
    if [[ -n "${RUSTFLAGS:-}" ]]; then
        export RUSTFLAGS="$RUSTFLAGS $RV64_RUSTFLAGS"
    else
        export RUSTFLAGS="$RV64_RUSTFLAGS"
    fi
}

cleanup() {
    if [[ "${COLIMA_STARTED:-}" = "1" ]]; then
        echo "[riscv64-spectest] Stopping Colima (we started it)..."
        colima stop 2>/dev/null || true
    fi
}
trap cleanup EXIT

for cmd in cargo rustup; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first." >&2
        exit 1
    fi
done

if ! rustup target list --installed | grep -qx "$TARGET"; then
    echo "[riscv64-spectest] Installing Rust target $TARGET..."
    rustup target add "$TARGET"
fi

if [[ ! -d "$TESTSUITE_DIR" ]]; then
    echo "ERROR: Testsuite not found at $TESTSUITE_DIR" >&2
    echo "Download it or set TESTSUITE_DIR." >&2
    exit 1
fi

HOST_OS="$(uname -s)"
COLIMA_STARTED=0
USE_COLIMA=0
case "$HOST_OS" in
    Darwin)
        USE_COLIMA=1
        if ! command -v colima &>/dev/null; then
            echo "ERROR: colima not found. Install Colima or run on Linux/WSL." >&2
            exit 1
        fi
        if ! colima status &>/dev/null; then
            echo "[riscv64-spectest] Starting Colima..."
            colima start
            COLIMA_STARTED=1
        fi
        if ! colima ssh -- which qemu-riscv64-static &>/dev/null; then
            echo "[riscv64-spectest] Installing qemu-user-static in Colima VM..."
            colima ssh -- sudo apt-get update -qq
            colima ssh -- sudo apt-get install -y -qq qemu-user-static
        fi
        ;;
    Linux)
        if ! command -v qemu-riscv64-static &>/dev/null; then
            echo "ERROR: qemu-riscv64-static not found. Install qemu-user-static." >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: unsupported host OS '$HOST_OS'; use macOS+Colima or Linux/WSL." >&2
        exit 1
        ;;
esac

echo "[riscv64-spectest] Cross-compiling spectest for RV64GC..."
append_rustflags
cargo build --target "$TARGET" -p sf-nano-spectest --no-default-features --features jit

if [[ ! -f "$SPECTEST_BIN" ]]; then
    echo "ERROR: Build succeeded but binary not found at $SPECTEST_BIN" >&2
    exit 1
fi

echo "[riscv64-spectest] Running spectest under QEMU..."
echo
if [[ "$USE_COLIMA" -eq 1 ]]; then
    colima ssh -- env TESTSUITE_DIR="$TESTSUITE_DIR" \
        qemu-riscv64-static -cpu rv64 "$SPECTEST_BIN" --backend native ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    env TESTSUITE_DIR="$TESTSUITE_DIR" \
        qemu-riscv64-static -cpu rv64 "$SPECTEST_BIN" --backend native ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi
