#!/usr/bin/env bash
#
# Run the feature matrix, target matrix, full spectest suite, and WASI
# testsuite, then report every failure and warning instead of stopping at the
# first issue.
#
# Usage:
#   ./scripts/check_all.sh
#   ./scripts/check_all.sh --strict --install-targets
#   ./scripts/check_all.sh --default-targets
#   ./scripts/check_all.sh -- --log-level info if
#
# Options:
#   --debug-only       Forward to check_features.sh, spectest_all.sh, wasitest_all.sh
#   --release-only     Forward to check_features.sh, spectest_all.sh, wasitest_all.sh
#   --strict           Forward to check_features.sh
#   --install-targets  Forward to check_targets.sh as --install
#   --all-targets      Use the full check_targets.sh target set (default)
#   --default-targets  Use the smaller check_targets.sh default target set
#   --                 Forward the remaining args to spectest_all.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/target/check-all-logs"
mkdir -p "$LOG_DIR"

feature_args=()
target_args=()
spectest_args=()
wasitest_args=()
target_mode="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug-only|--release-only)
            feature_args+=("$1")
            spectest_args+=("$1")
            wasitest_args+=("$1")
            ;;
        --strict)
            feature_args+=("$1")
            ;;
        --install-targets)
            target_args+=("--install")
            ;;
        --all-targets)
            target_mode="all"
            ;;
        --default-targets)
            target_mode="default"
            ;;
        --)
            shift
            spectest_args+=("$@")
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

if [[ "$target_mode" == "all" ]]; then
    target_args=("--all" ${target_args[@]+"${target_args[@]}"})
fi

step_names=()
step_statuses=()
step_logs=()
step_warning_counts=()
step_error_counts=()

count_step_warnings() {
    local name="$1"
    local log="$2"

    case "$name" in
        check_features.sh)
            awk '
                /^  warnings:/ {
                    print $2
                    found = 1
                    exit
                }
                END {
                    if (!found) {
                        print 0
                    }
                }
            ' "$log"
            ;;
        check_targets.sh)
            grep -Eo '\([0-9]+w\)' "$log" 2>/dev/null \
                | tr -dc '0-9\n' \
                | awk '{ total += $1 } END { print total + 0 }'
            ;;
        spectest_all.sh)
            grep -Eo 'generated [0-9]+ warnings?' "$log" 2>/dev/null \
                | awk '{ total += $2 } END { print total + 0 }'
            ;;
        wasitest_all.sh)
            grep -Eo 'generated [0-9]+ warnings?' "$log" 2>/dev/null \
                | awk '{ total += $2 } END { print total + 0 }'
            ;;
        *)
            echo 0
            ;;
    esac
}

count_step_errors() {
    local name="$1"
    local log="$2"

    case "$name" in
        check_features.sh)
            awk '
                /^  failed:/ {
                    print $2
                    found = 1
                    exit
                }
                END {
                    if (!found) {
                        print 0
                    }
                }
            ' "$log"
            ;;
        check_targets.sh)
            grep -c 'FAIL' "$log" 2>/dev/null || true
            ;;
        spectest_all.sh)
            grep -Ec '(^error:|^error\[| level=fatal )' "$log" 2>/dev/null || true
            ;;
        wasitest_all.sh)
            grep -Ec '(^error:|^error\[| level=fatal )' "$log" 2>/dev/null || true
            ;;
        *)
            echo 0
            ;;
    esac
}

# run_step:         stream child output to stdout AND log. Used for the
#                   matrix-style scripts (check_features.sh, check_targets.sh)
#                   whose own stdout already IS a compact, useful report.
# run_step_compact: redirect child output to log only; print a one-line
#                   verdict to stdout. On failure, dump matched error lines
#                   + tail of the log so the cause is visible without paging
#                   through thousands of per-test lines. Used for the chatty
#                   runtime test runners (spectest_all.sh, wasitest_all.sh).
run_step()         { _run_step loud    "$@"; }
run_step_compact() { _run_step compact "$@"; }

_run_step() {
    local mode="$1"
    local name="$2"
    local log="$3"
    shift 3

    local rc=0
    local warnings=0
    local errors=0
    local status="ok"

    echo "=== $name ==="
    if [[ "$mode" == "compact" ]]; then
        # Stream the child's own `=== section ===` heartbeat lines to
        # stdout as they appear so a long run still shows progress, while
        # the full child output is captured in $log. PIPESTATUS[0] gives
        # the child's real exit code regardless of whether grep matched.
        "$@" 2>&1 \
            | tee "$log" \
            | { grep --line-buffered -E '^===' || true; } \
            | awk '{ print "  " $0; fflush() }'
        rc="${PIPESTATUS[0]:-0}"
    else
        if "$@" 2>&1 | tee "$log"; then
            rc=0
        else
            rc=$?
        fi
    fi

    step_names+=("$name")
    warnings="$(count_step_warnings "$name" "$log")"
    errors="$(count_step_errors "$name" "$log")"
    if [[ $rc -ne 0 ]]; then
        status="FAIL"
    elif [[ ${warnings:-0} -gt 0 ]]; then
        status="WARN"
    fi

    if [[ "$mode" == "compact" ]]; then
        if [[ $rc -ne 0 ]]; then
            echo "  FAILED — relevant error lines:"
            grep -nE '^(error:|error\[| level=fatal |FAIL)' "$log" \
                | head -40 \
                | sed 's/^/    /'
            echo "  --- last 20 lines of log ---"
            tail -20 "$log" | sed 's/^/    /'
            echo "  full log: $log"
        else
            local detail="0 warnings"
            if [[ ${warnings:-0} -gt 0 ]]; then
                detail="${warnings} warnings"
            fi
            echo "  ok (${detail}, full log: $log)"
        fi
    fi

    step_statuses+=("$status")
    step_logs+=("$log")
    step_warning_counts+=("${warnings:-0}")
    step_error_counts+=("${errors:-0}")
    echo
    return "$rc"
}

overall_fail=0
overall_warn=0

run_step \
    "check_features.sh" \
    "$LOG_DIR/check-features.log" \
    "$REPO_ROOT/scripts/check_features.sh" \
    ${feature_args[@]+"${feature_args[@]}"} || overall_fail=1

run_step \
    "check_targets.sh" \
    "$LOG_DIR/check-targets.log" \
    "$REPO_ROOT/scripts/check_targets.sh" \
    ${target_args[@]+"${target_args[@]}"} || overall_fail=1

run_step_compact \
    "spectest_all.sh" \
    "$LOG_DIR/spectest-all.log" \
    "$REPO_ROOT/scripts/spectest_all.sh" -- \
    ${spectest_args[@]+"${spectest_args[@]}"} || overall_fail=1

run_step_compact \
    "wasitest_all.sh" \
    "$LOG_DIR/wasitest-all.log" \
    "$REPO_ROOT/scripts/wasitest_all.sh" \
    ${wasitest_args[@]+"${wasitest_args[@]}"} || overall_fail=1

for status in "${step_statuses[@]}"; do
    if [[ "$status" == "WARN" ]]; then
        overall_warn=1
        break
    fi
done

echo "=== Summary ==="
printf "  %-20s %-6s %-8s %-6s %s\n" "step" "status" "warnings" "errors" "log"
printf "  %-20s %-6s %-8s %-6s %s\n" "--------------------" "------" "--------" "------" "---"

for i in "${!step_names[@]}"; do
    printf "  %-20s %-6s %-8s %-6s %s\n" \
        "${step_names[$i]}" \
        "${step_statuses[$i]}" \
        "${step_warning_counts[$i]}" \
        "${step_error_counts[$i]}" \
        "${step_logs[$i]}"
done

echo
if [[ $overall_fail -ne 0 ]]; then
    echo "ERRORS found. Fix them before treating this run as clean."
fi
if [[ $overall_warn -ne 0 ]]; then
    echo "WARNINGS found. Fix them before treating this run as clean."
fi
if [[ $overall_fail -eq 0 && $overall_warn -eq 0 ]]; then
    echo "No warnings or errors found."
fi
echo
echo "Nested logs:"
echo "  check_features.sh -> $REPO_ROOT/target/check-features-logs/"
echo "  check_targets.sh  -> $REPO_ROOT/target/check-targets-logs/"
echo "  spectest_all.sh   -> $LOG_DIR/spectest-all.log"
echo "  wasitest_all.sh   -> $LOG_DIR/wasitest-all.log"

if [[ $overall_fail -ne 0 || $overall_warn -ne 0 ]]; then
    exit 1
fi
exit 0
