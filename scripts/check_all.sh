#!/usr/bin/env bash
#
# Run the feature matrix, target matrix, and full spectest suite, then report
# every failure instead of stopping at the first one.
#
# Usage:
#   ./scripts/check_all.sh
#   ./scripts/check_all.sh --strict --install-targets
#   ./scripts/check_all.sh --all-targets
#   ./scripts/check_all.sh -- --log-level info if
#
# Options:
#   --debug-only       Forward to check_features.sh
#   --release-only     Forward to check_features.sh
#   --strict           Forward to check_features.sh
#   --install-targets  Forward to check_targets.sh as --install
#   --all-targets      Forward to check_targets.sh as --all
#   --default-targets  Use the default check_targets.sh target set
#   --                 Forward the remaining args to spectest_all.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/target/check-all-logs"
mkdir -p "$LOG_DIR"

feature_args=()
target_args=()
spectest_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug-only|--release-only|--strict)
            feature_args+=("$1")
            ;;
        --install-targets)
            target_args+=("--install")
            ;;
        --all-targets)
            target_args+=("--all")
            ;;
        --default-targets)
            ;;
        --)
            shift
            spectest_args=("$@")
            break
            ;;
        -h|--help)
            sed -n '3,19p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
    shift
done

step_names=()
step_statuses=()
step_logs=()

run_step() {
    local name="$1"
    local log="$2"
    shift 2

    echo "=== $name ==="
    if "$@" 2>&1 | tee "$log"; then
        step_names+=("$name")
        step_statuses+=("ok")
        step_logs+=("$log")
        echo
        return 0
    fi

    step_names+=("$name")
    step_statuses+=("FAIL")
    step_logs+=("$log")
    echo
    return 1
}

overall_fail=0

run_step \
    "check_features.sh" \
    "$LOG_DIR/check-features.log" \
    "$REPO_ROOT/scripts/check_features.sh" "${feature_args[@]}" || overall_fail=1

run_step \
    "check_targets.sh" \
    "$LOG_DIR/check-targets.log" \
    "$REPO_ROOT/scripts/check_targets.sh" "${target_args[@]}" || overall_fail=1

run_step \
    "spectest_all.sh" \
    "$LOG_DIR/spectest-all.log" \
    "$REPO_ROOT/scripts/spectest_all.sh" -- "${spectest_args[@]}" || overall_fail=1

echo "=== Summary ==="
printf "  %-20s %-6s %s\n" "step" "status" "log"
printf "  %-20s %-6s %s\n" "--------------------" "------" "---"

for i in "${!step_names[@]}"; do
    printf "  %-20s %-6s %s\n" \
        "${step_names[$i]}" "${step_statuses[$i]}" "${step_logs[$i]}"
done

echo
echo "Nested logs:"
echo "  check_features.sh -> $REPO_ROOT/target/check-features-logs/"
echo "  check_targets.sh  -> $REPO_ROOT/target/check-targets-logs/"
echo "  spectest_all.sh   -> $LOG_DIR/spectest-all.log"

exit "$overall_fail"
