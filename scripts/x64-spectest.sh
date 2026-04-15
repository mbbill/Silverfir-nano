#!/usr/bin/env bash
#
# Run the x86_64 spectest under Rosetta on macOS.
#
# Usage:
#   ./scripts/x64-spectest.sh [-- extra spectest args...]
#
# Examples:
#   ./scripts/x64-spectest.sh
#   ./scripts/x64-spectest.sh -- if
#   ./scripts/x64-spectest.sh -- --log-level debug i32

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=x86_64-apple-darwin
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

for cmd in arch cargo; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first." >&2
        exit 1
    fi
done

echo "[x64-spectest] Cross-compiling spectest for x86_64..."
cargo build --target "$TARGET" -p sf-nano-spectest --no-default-features --features jit

if [[ ! -f "$SPECTEST_BIN" ]]; then
    echo "ERROR: Build succeeded but binary not found at $SPECTEST_BIN" >&2
    exit 1
fi

if [[ ! -d "$TESTSUITE_DIR" ]]; then
    echo "ERROR: Testsuite not found at $TESTSUITE_DIR" >&2
    echo "Download it or set TESTSUITE_DIR." >&2
    exit 1
fi

echo "[x64-spectest] Running spectest under Rosetta..."
echo
env TESTSUITE_DIR="$TESTSUITE_DIR" \
    arch -x86_64 "$SPECTEST_BIN" --backend native ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
