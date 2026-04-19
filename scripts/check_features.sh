#!/usr/bin/env bash
#
# check_features.sh — build-check the JIT-on feature surface in both debug and
# release profiles, then cross-check the dependent crates.
#
# What it does
# ------------
# 1. `cargo check -p sf-nano-core` with a JIT-on matrix in both dev and
#    release profiles. The matrix covers every single optional feature, every
#    pair of optional features, and a few real-world bundled configs.
# 2. `cargo check -p sf-nano-cli` with its optional feature surface in both
#    dev and release profiles.
# 3. `cargo check --workspace` with default features (sanity check that the
#    binaries still build against the default feature set).
# 4. `cargo check` in `sf-nano-cli-minimal` (excluded from workspace).
#
# Per-combination stderr is captured to `target/check-features-logs/<label>-<profile>.log`
# so failures and warnings can be inspected after the run.
#
# Usage
# -----
#   scripts/check_features.sh                # run everything
#   scripts/check_features.sh --debug-only   # skip release builds
#   scripts/check_features.sh --release-only # skip debug builds
#   scripts/check_features.sh --strict       # promote warnings to errors (-D warnings)
#
# Exit status is 0 only if every build passed. Warnings do not fail the run
# unless --strict is passed.
#
# Matrix note: only "jit-on" combinations are tested. Non-jit builds of
# sf-nano-core are not supported today (no execution backend), so this script
# focuses on wide JIT-on coverage instead of the full feature powerset.
#
# Note: written for bash 3.2+ (macOS default). Uses plain strings for flag
# lists rather than arrays to avoid `set -u` + empty-array quirks.

set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="target/check-features-logs"
mkdir -p "$LOG_DIR"

# ── Options ──────────────────────────────────────────────────────────────────

run_debug=1
run_release=1
deny_warnings=0

for arg in "$@"; do
    case "$arg" in
        --debug-only)   run_release=0 ;;
        --release-only) run_debug=0 ;;
        --strict)       deny_warnings=1 ;;
        -h|--help)
            sed -n '3,32p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# ── Matrix ───────────────────────────────────────────────────────────────────
#
# Each row: LABEL|FEATURE_SPEC
#
# FEATURE_SPEC values:
#   EMPTY      — `--no-default-features`
#   DEFAULT    — no explicit flag (uses Cargo.toml default)
#   ALL        — `--all-features`
#   <list>     — `--no-default-features --features <comma-list>`

CORE_OPTIONAL_FEATURES=(
    "backend-emu64"
    "backend-emu32"
    "wasi"
    "validator"
    "call-trace"
    "guard-pages"
    "ir-dump"
    "jitdump"
    "memprof"
    "thumb2-test"
)

CORE_COMBOS=(
    # Baseline profiles
    "default|DEFAULT"
    "all-features|ALL"

    # Minimum supported configuration
    "only-jit|jit"
)

for feature in "${CORE_OPTIONAL_FEATURES[@]}"; do
    CORE_COMBOS+=("jit+$feature|jit,$feature")
done

for ((i = 0; i < ${#CORE_OPTIONAL_FEATURES[@]}; i++)); do
    for ((j = i + 1; j < ${#CORE_OPTIONAL_FEATURES[@]}; j++)); do
        feature_a="${CORE_OPTIONAL_FEATURES[$i]}"
        feature_b="${CORE_OPTIONAL_FEATURES[$j]}"
        CORE_COMBOS+=("jit+$feature_a+$feature_b|jit,$feature_a,$feature_b")
    done
done

CORE_COMBOS+=(
    # Realistic bundled configurations
    "cli|jit,wasi,guard-pages"
    "cli+tracing|jit,wasi,guard-pages,call-trace"
    "cli+ir-dump|jit,wasi,guard-pages,ir-dump"
    "cli+jitdump|jit,wasi,guard-pages,jitdump"
    "cli+memprof|jit,wasi,guard-pages,memprof"
    "cli+emu64|jit,wasi,guard-pages,backend-emu64"
    "cli+emu32|jit,wasi,guard-pages,backend-emu32"
    "spectest|jit,validator,guard-pages"
    "spectest+emu64|jit,validator,guard-pages,backend-emu64"
    "spectest+emu32|jit,validator,guard-pages,backend-emu32"
    "full-runtime|jit,wasi,validator,guard-pages"
    "full-runtime+emu64|jit,wasi,validator,guard-pages,backend-emu64"
    "full-runtime+emu32|jit,wasi,validator,guard-pages,backend-emu32"
    "full-dev|jit,wasi,validator,call-trace,guard-pages,ir-dump,jitdump,memprof"

    # Thumb-2 ISA variant (arm32 backend emits Thumb-2 instead of A32). The
    # cfg gating (`sf_arm32_isa_thumb`) only has effect when the host arch
    # is arm, but the source must still typecheck on every host — so these
    # rows catch accidental cfg breakage during cross-crate edits.
    "cli+thumb2|jit,wasi,guard-pages,thumb2-test"
    "spectest+thumb2|jit,validator,guard-pages,thumb2-test"
)

CLI_OPTIONAL_FEATURES=(
    "call-trace"
    "memprof"
    "thumb2-test"
)

CLI_COMBOS=(
    "default|DEFAULT"
    "all-features|ALL"
    "only-jit|jit"
)

for feature in "${CLI_OPTIONAL_FEATURES[@]}"; do
    CLI_COMBOS+=("jit+$feature|jit,$feature")
done

for ((i = 0; i < ${#CLI_OPTIONAL_FEATURES[@]}; i++)); do
    for ((j = i + 1; j < ${#CLI_OPTIONAL_FEATURES[@]}; j++)); do
        feature_a="${CLI_OPTIONAL_FEATURES[$i]}"
        feature_b="${CLI_OPTIONAL_FEATURES[$j]}"
        CLI_COMBOS+=("jit+$feature_a+$feature_b|jit,$feature_a,$feature_b")
    done
done

# ── Helpers ──────────────────────────────────────────────────────────────────

# Translate a FEATURE_SPEC token into the feature portion of `cargo check`
# flags. Echoes a string that can be expanded word-split into the command.
feature_flags() {
    case "$1" in
        EMPTY)   echo "--no-default-features" ;;
        DEFAULT) echo "" ;;
        ALL)     echo "--all-features" ;;
        *)       echo "--no-default-features --features $1" ;;
    esac
}

# Run one check. Args: label, profile ("dev"|"release"), features spec.
# Writes stderr to the log, prints one-line status. Returns 0/1/2:
#   0 — clean pass
#   1 — failed
#   2 — warnings (only reachable with --no-warnings, since -D warnings
#       otherwise turns them into build errors)
package_name() {
    case "$1" in
        core) echo "sf-nano-core" ;;
        cli) echo "sf-nano-cli" ;;
        *)
            echo "internal error: unknown package key '$1'" >&2
            exit 2
            ;;
    esac
}

run_one() {
    local package_key="$1" label="$2" profile="$3" spec="$4"
    local package
    package="$(package_name "$package_key")"
    local fflags
    fflags="$(feature_flags "$spec")"

    local profile_flag=""
    if [[ "$profile" == "release" ]]; then
        profile_flag="--release"
    fi

    local rustflags=""
    if [[ $deny_warnings -eq 1 ]]; then
        rustflags="-D warnings"
    fi

    local log="$LOG_DIR/${package_key}-${label}-${profile}.log"

    # Intentional word splitting on $fflags/$profile_flag (plain strings,
    # controlled input).
    # shellcheck disable=SC2086
    if RUSTFLAGS="$rustflags" cargo check -p "$package" \
            $profile_flag $fflags \
            >"$log" 2>&1; then
        local wc_local
        wc_local=$(grep -c '^warning:' "$log" 2>/dev/null)
        [[ -z "$wc_local" ]] && wc_local=0
        if [[ $wc_local -gt 0 ]]; then
            printf "  %-8s %-22s %-8s  WARN  (%d)  %s\n" "$package_key" "$label" "$profile" "$wc_local" "$log"
            return 2
        else
            printf "  %-8s %-22s %-8s  ok\n" "$package_key" "$label" "$profile"
            return 0
        fi
    else
        local ec_local
        ec_local=$(grep -c '^error' "$log" 2>/dev/null)
        [[ -z "$ec_local" ]] && ec_local=0
        printf "  %-8s %-22s %-8s  FAIL  (%d)  %s\n" "$package_key" "$label" "$profile" "$ec_local" "$log"
        return 1
    fi
}

run_combo_matrix() {
    local heading="$1"
    local package_key="$2"
    shift 2
    local entries=("$@")

    echo "=== $heading ==="
    echo "  package  label                  profile   status"
    echo "  -------  ---------------------- --------  -------"

    for entry in "${entries[@]}"; do
        label="${entry%%|*}"
        spec="${entry#*|}"

        if [[ $run_debug -eq 1 ]]; then
            total_count=$((total_count + 1))
            run_one "$package_key" "$label" "dev" "$spec"
            rc=$?
            [[ $rc -eq 1 ]] && fail_count=$((fail_count + 1))
            [[ $rc -eq 2 ]] && warn_count=$((warn_count + 1))
        fi

        if [[ $run_release -eq 1 ]]; then
            total_count=$((total_count + 1))
            run_one "$package_key" "$label" "release" "$spec"
            rc=$?
            [[ $rc -eq 1 ]] && fail_count=$((fail_count + 1))
            [[ $rc -eq 2 ]] && warn_count=$((warn_count + 1))
        fi
    done
}

# ── Run matrix ───────────────────────────────────────────────────────────────

fail_count=0
warn_count=0
total_count=0

run_combo_matrix "sf-nano-core feature matrix" core "${CORE_COMBOS[@]}"

echo
run_combo_matrix "sf-nano-cli feature matrix" cli "${CLI_COMBOS[@]}"

# ── Workspace default build ──────────────────────────────────────────────────

echo
echo "=== workspace default features ==="

for profile in dev release; do
    [[ "$profile" == "dev" && $run_debug -eq 0 ]] && continue
    [[ "$profile" == "release" && $run_release -eq 0 ]] && continue

    total_count=$((total_count + 1))
    label="workspace"
    profile_flag=""
    [[ "$profile" == "release" ]] && profile_flag="--release"

    rustflags=""
    [[ $deny_warnings -eq 1 ]] && rustflags="-D warnings"

    log="$LOG_DIR/workspace-${profile}.log"
    # shellcheck disable=SC2086
    if RUSTFLAGS="$rustflags" cargo check --workspace $profile_flag >"$log" 2>&1; then
        wc_local=$(grep -c '^warning:' "$log" 2>/dev/null)
        [[ -z "$wc_local" ]] && wc_local=0
        if [[ $wc_local -gt 0 ]]; then
            printf "  %-22s %-8s  WARN  (%d)  %s\n" "$label" "$profile" "$wc_local" "$log"
            warn_count=$((warn_count + 1))
        else
            printf "  %-22s %-8s  ok\n" "$label" "$profile"
        fi
    else
        ec_local=$(grep -c '^error' "$log" 2>/dev/null)
        [[ -z "$ec_local" ]] && ec_local=0
        printf "  %-22s %-8s  FAIL  (%d)  %s\n" "$label" "$profile" "$ec_local" "$log"
        fail_count=$((fail_count + 1))
    fi
done

# ── cli-minimal (excluded from workspace) ────────────────────────────────────

echo
echo "=== sf-nano-cli-minimal ==="

if [[ -d "sf-nano-cli-minimal" ]]; then
    for profile in dev release; do
        [[ "$profile" == "dev" && $run_debug -eq 0 ]] && continue
        [[ "$profile" == "release" && $run_release -eq 0 ]] && continue

        total_count=$((total_count + 1))
        label="cli-minimal"
        profile_flag=""
        [[ "$profile" == "release" ]] && profile_flag="--release"

        rustflags=""
        [[ $deny_warnings -eq 1 ]] && rustflags="-D warnings"

        log="$LOG_DIR/cli-minimal-${profile}.log"
        # shellcheck disable=SC2086
        if (cd sf-nano-cli-minimal && RUSTFLAGS="$rustflags" cargo check $profile_flag) \
                >"$log" 2>&1; then
            wc_local=$(grep -c '^warning:' "$log" 2>/dev/null)
            [[ -z "$wc_local" ]] && wc_local=0
            if [[ $wc_local -gt 0 ]]; then
                printf "  %-22s %-8s  WARN  (%d)  %s\n" "$label" "$profile" "$wc_local" "$log"
                warn_count=$((warn_count + 1))
            else
                printf "  %-22s %-8s  ok\n" "$label" "$profile"
            fi
        else
            ec_local=$(grep -c '^error' "$log" 2>/dev/null)
            [[ -z "$ec_local" ]] && ec_local=0
            printf "  %-22s %-8s  FAIL  (%d)  %s\n" "$label" "$profile" "$ec_local" "$log"
            fail_count=$((fail_count + 1))
        fi
    done
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo
echo "=== summary ==="
printf "  total:    %d\n" "$total_count"
printf "  failed:   %d\n" "$fail_count"
printf "  warnings: %d\n" "$warn_count"
echo "  logs: $LOG_DIR/"

if [[ $fail_count -gt 0 ]]; then
    exit 1
fi
exit 0
