#!/usr/bin/env bash
#
# Deprecated direct entry point. Prefer `python3 scripts/check.py full`; this
# script is kept as a low-level helper.
#
# Run the WASI benchmark tests on x86_64 under Rosetta on macOS.
#
# Usage:
#   ./scripts/x64-run-tests.sh [--release]
#
# By default builds and runs in debug mode. Pass --release for an
# optimised build.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=x86_64-apple-darwin
PROFILE=debug
CARGO_PROFILE_FLAG=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release) PROFILE=release; CARGO_PROFILE_FLAG=(--release); shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--release]"; exit 1 ;;
    esac
done

CLI_BIN="$REPO_ROOT/target/$TARGET/$PROFILE/sf-nano-cli"
RUN_TESTS="$REPO_ROOT/benchmarks/wasi/run_tests.py"

for cmd in arch cargo python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first." >&2
        exit 1
    fi
done

echo "[x64-run-tests] Cross-compiling CLI ($PROFILE) for x86_64..."
cargo build --target "$TARGET" -p sf-nano-cli --no-default-features --features jit \
    ${CARGO_PROFILE_FLAG[@]+"${CARGO_PROFILE_FLAG[@]}"}

if [[ ! -f "$CLI_BIN" ]]; then
    echo "ERROR: Build succeeded but binary not found at $CLI_BIN" >&2
    exit 1
fi

echo "[x64-run-tests] Running benchmarks under Rosetta ($PROFILE)..."
echo
python3 "$RUN_TESTS" --exec "arch -x86_64 $CLI_BIN"
