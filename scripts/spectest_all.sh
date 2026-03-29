#!/usr/bin/env bash
#
# Build and run spectest for all backends: arm64, emu64, emu32, then armv7.
# Stops on first failure.
#
# Usage:
#   ./scripts/spectest_all.sh [-- extra spectest args...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TESTSUITE_DIR="${TESTSUITE_DIR:-$REPO_ROOT/target/webassembly-testsuite-2.0}"
export TESTSUITE_DIR

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [-- extra spectest args...]"; exit 1 ;;
    esac
done

if [[ ! -d "$TESTSUITE_DIR" ]]; then
    echo "ERROR: Testsuite not found at $TESTSUITE_DIR" >&2
    exit 1
fi

# --- Build (one release build covers arm64/emu64/emu32) ---

echo "=== Building spectest (release) ==="
cargo build --release -p sf-nano-spectest --no-default-features --features micro-jit
SPECTEST="$REPO_ROOT/target/release/sf-nano-spectest"
echo

# --- arm64 ---

echo "=== spectest: arm64 ==="
"$SPECTEST" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
echo

# --- emu64 ---

echo "=== spectest: emu64 ==="
"$SPECTEST" --emu64 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
echo

# --- emu32 ---

echo "=== spectest: emu32 ==="
"$SPECTEST" --emu32 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
echo

# --- armv7 ---

echo "=== spectest: armv7 ==="
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    "$REPO_ROOT/scripts/armv7-spectest.sh" -- "${EXTRA_ARGS[@]}"
else
    "$REPO_ROOT/scripts/armv7-spectest.sh"
fi
echo

echo "=== All spectests passed ==="
