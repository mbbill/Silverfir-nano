#!/usr/bin/env bash
#
# Deprecated direct entry point. Prefer `python3 scripts/check.py fast` or
# `python3 scripts/check.py full`; this script is kept as a low-level helper.
#
# Run the RV32GC spectest under QEMU user-mode emulation. On macOS this uses
# qemu-riscv32-static inside Colima; on Linux/WSL it uses qemu-riscv32-static
# directly from PATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=riscv32gc-unknown-linux-musl
SPECTEST_BIN="$REPO_ROOT/target/$TARGET/debug/sf-nano-spectest"
TESTSUITE_DIR="${TESTSUITE_DIR:-$REPO_ROOT/target/webassembly-testsuite}"
RV32_RUSTFLAGS="-C linker=$REPO_ROOT/scripts/zig-riscv32-linux-musl-cc.sh -C target-feature=+crt-static -C link-self-contained=no -C panic=abort"
export ZIG_GLOBAL_CACHE_DIR="${ZIG_GLOBAL_CACHE_DIR:-$REPO_ROOT/target/zig-cache}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [-- extra spectest args...]"; exit 1 ;;
    esac
done

append_rustflags() {
    if [[ -n "${RUSTFLAGS:-}" ]]; then
        export RUSTFLAGS="$RUSTFLAGS $RV32_RUSTFLAGS"
    else
        export RUSTFLAGS="$RV32_RUSTFLAGS"
    fi
}

cleanup() {
    if [[ "${COLIMA_STARTED:-}" = "1" ]]; then
        echo "[riscv32-spectest] Stopping Colima (we started it)..."
        colima stop 2>/dev/null || true
    fi
}
trap cleanup EXIT

for cmd in cargo zig; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. RV32 builds require nightly cargo and Zig." >&2
        exit 1
    fi
done

if ! cargo +nightly --version &>/dev/null; then
    echo "ERROR: cargo +nightly is required for RV32 build-std." >&2
    exit 1
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
            echo "[riscv32-spectest] Starting Colima..."
            colima start
            COLIMA_STARTED=1
        fi
        if ! colima ssh -- which qemu-riscv32-static &>/dev/null; then
            echo "[riscv32-spectest] Installing qemu-user-static in Colima VM..."
            colima ssh -- sudo apt-get update -qq
            colima ssh -- sudo apt-get install -y -qq qemu-user-static
        fi
        ;;
    Linux)
        if ! command -v qemu-riscv32-static &>/dev/null; then
            echo "ERROR: qemu-riscv32-static not found. Install qemu-user-static." >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: unsupported host OS '$HOST_OS'; use macOS+Colima or Linux/WSL." >&2
        exit 1
        ;;
esac

echo "[riscv32-spectest] Cross-compiling spectest for RV32GC..."
append_rustflags
cargo +nightly build -Z build-std=std,panic_abort --target "$TARGET" \
    -p sf-nano-spectest --no-default-features --features jit

if [[ ! -f "$SPECTEST_BIN" ]]; then
    echo "ERROR: Build succeeded but binary not found at $SPECTEST_BIN" >&2
    exit 1
fi

echo "[riscv32-spectest] Running spectest under QEMU..."
echo
if [[ "$USE_COLIMA" -eq 1 ]]; then
    colima ssh -- env TESTSUITE_DIR="$TESTSUITE_DIR" \
        qemu-riscv32-static -cpu rv32 "$SPECTEST_BIN" --backend native ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    env TESTSUITE_DIR="$TESTSUITE_DIR" \
        qemu-riscv32-static -cpu rv32 "$SPECTEST_BIN" --backend native ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi
