#!/usr/bin/env bash
#
# Fast day-to-day validation gate.
#
# Coverage:
# - workspace release build (all packages compile in their default feature sets)
# - sf-nano-core integration tests in:
#     * the default release feature set
#     * the curated "full-runtime" release feature bundle
# - spectest in release mode across:
#     * native JIT
#     * emu64 build
#     * emu32 build
#
# Intentionally omitted to stay fast:
# - feature matrix
# - cross-target build matrix
# - x64 / armv7a / armv7m spectest runs
# - WASI testsuite
#
# Usage:
#   ./scripts/check_fast.sh
#   ./scripts/check_fast.sh -- i32
#   ./scripts/check_fast.sh -- array_copy --log-level info

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TESTSUITE_DIR="${TESTSUITE_DIR:-$REPO_ROOT/target/webassembly-testsuite}"
LOG_DIR="$REPO_ROOT/target/check-fast-logs"
mkdir -p "$LOG_DIR"

SPECTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            sed -n '3,24p' "$0"
            exit 0
            ;;
        --)
            shift
            SPECTEST_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [-- extra spectest args...]" >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "$TESTSUITE_DIR" ]]; then
    echo "ERROR: Testsuite not found at $TESTSUITE_DIR" >&2
    exit 1
fi

need_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: $cmd not found. Install it first." >&2
        exit 1
    fi
}

need_command cargo

run_step() {
    local name="$1"
    local log="$2"
    shift 2

    echo "=== $name ==="
    if "$@" 2>&1 | tee "$log"; then
        echo
        return 0
    fi

    local rc=$?
    echo
    echo "FAILED — last 20 log lines:"
    tail -20 "$log" | sed 's/^/  /'
    echo "full log: $log"
    return "$rc"
}

FULL_RUNTIME_FEATURES="jit,wasi,validator,guard-pages"

run_step \
    "cargo build (workspace, release)" \
    "$LOG_DIR/cargo-workspace-release.log" \
    cargo build --workspace --release

# Keep the fast gate on the user-facing integration tests rather than
# `cargo test --workspace --release`: the latter currently drags in a noisy
# set of lib-unit tests under `sf-nano-core/src/vm/arch/emulator/mod.rs`
# that are useful for deeper work but make a poor quick regression signal.
CORE_TEST_TARGETS=(
    smoke
    array_ops
    struct_new_wide
)

run_core_integration_tests() {
    local label="$1"
    shift

    local test_name
    for test_name in "${CORE_TEST_TARGETS[@]}"; do
        run_step \
            "cargo test (sf-nano-core ${label}, ${test_name})" \
            "$LOG_DIR/cargo-core-${label// /-}-${test_name}.log" \
            cargo test -p sf-nano-core --release "$@" --test "$test_name"
    done
}

run_core_integration_tests "default" 
run_core_integration_tests \
    "full-runtime" \
    --no-default-features \
    --features "$FULL_RUNTIME_FEATURES"

build_spectest_release() {
    local label="$1"
    local features="$2"
    run_step \
        "cargo build (sf-nano-spectest, release, $label)" \
        "$LOG_DIR/spectest-build-$label-release.log" \
        cargo build --release -p sf-nano-spectest --no-default-features --features "$features"
}

run_spectest() {
    local name="$1"
    local log="$2"
    local bin="$3"
    shift 3

    local -a cmd=("$bin" "$@")
    if [[ ${#SPECTEST_ARGS[@]} -gt 0 ]]; then
        cmd+=("${SPECTEST_ARGS[@]}")
    fi
    TESTSUITE_DIR="$TESTSUITE_DIR" run_step "$name" "$log" "${cmd[@]}"
}

build_spectest_release native jit
SPECTEST_BIN="$REPO_ROOT/target/release/sf-nano-spectest"
run_spectest \
    "spectest native (release)" \
    "$LOG_DIR/spectest-native-release.log" \
    "$SPECTEST_BIN" \
    --backend native

build_spectest_release emu64 jit,backend-emu64
SPECTEST_EMU64_BIN="$REPO_ROOT/target/release/sf-nano-spectest"
run_spectest \
    "spectest emu64 (release)" \
    "$LOG_DIR/spectest-emu64-release.log" \
    "$SPECTEST_EMU64_BIN"

build_spectest_release emu32 jit,backend-emu32
SPECTEST_EMU32_BIN="$REPO_ROOT/target/release/sf-nano-spectest"
run_spectest \
    "spectest emu32 (release)" \
    "$LOG_DIR/spectest-emu32-release.log" \
    "$SPECTEST_EMU32_BIN"

echo "=== Summary ==="
echo "Fast checks passed."
echo "Logs: $LOG_DIR"
