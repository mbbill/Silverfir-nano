#!/usr/bin/env bash
#
# check-targets.sh — cross-compile `cargo check` for the main binaries and test
# harnesses against every supported target in the host's installed rustup
# target set.
#
# What it does
# ------------
# Runs `cargo check --target <t>` for each target in the matrix below and
# reports per-package pass/fail plus warning counts. arm64 and x86_64 targets
# are included by default. Pass `--all` to include the armv7a / Thumb-2 rows
# as well.
#
# Usage
# -----
#   scripts/check-targets.sh            # arm64 + x64 targets (default)
#   scripts/check-targets.sh --all      # include armv7a as well
#   scripts/check-targets.sh --install  # rustup target add anything missing
#
# Exit status is 0 only if every selected package check passed.
#
# Target matrix
# -------------
# | target                             | pkg set                           | exercises                |
# |------------------------------------|-----------------------------------|---------------------------|
# | aarch64-apple-darwin (host)        | core + cli + spectest + wasitest  | runtime/os/macos.rs       |
# | aarch64-unknown-linux-gnu          | core + cli + spectest + wasitest  | runtime/os/linux.rs       |
# | aarch64-unknown-none               | bare-smoke (1)                    | arm64 backend × runtime/os/none.rs |
# | thumbv8m.main-none-eabihf          | bare-smoke (1)                    | arm32 (Thumb-2) × runtime/os/none.rs (Pico 2 / Cortex-M33) |
# | x86_64-apple-darwin                | core + cli + spectest + wasitest  | runtime/os/macos.rs + x64 backend |
# | x86_64-unknown-linux-gnu           | core + cli + spectest + wasitest  | runtime/os/linux.rs + x64 trap handler |
# | x86_64-pc-windows-gnu              | core + cli + spectest + wasitest  | runtime/os/windows.rs     |
# | armv7-unknown-linux-musleabihf     | core + cli + spectest + wasitest  | runtime/os/linux.rs (32-bit) |
#
# (1) Bare rows are run via the standalone `sf-nano-bare-smoke` crate, which
#     supplies stub implementations of the embedder-provided extern symbols
#     declared in `runtime/os/none.rs`. The stubs only have to type-check —
#     the smoke test proves the *contract* compiles, not that the runtime
#     functions. Both pointer widths are checked because `runtime/os/none.rs`
#     is the only place where a 32-bit-vs-64-bit signature drift in the
#     extern shims would be caught.
# (2) The standalone script keeps armv7 rows behind --all to preserve the
#     smaller default target set; check_all.sh opts into them by default.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BARE_SMOKE_DIR="$ROOT_DIR/sf-nano-bare-smoke"

INCLUDE_ALL=0
DO_INSTALL=0
for arg in "$@"; do
    case "$arg" in
        --all) INCLUDE_ALL=1 ;;
        --install) DO_INSTALL=1 ;;
        -h|--help)
            sed -n '3,/^set -u/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# Each entry: "label|target|kind|core_extra|cli_spectest_features_override"
#
# Fields
# ------
#   label                         — human-readable row name in the matrix
#   target                        — rustc target triple
#   kind                          — "hosted" (core+cli+spectest+wasitest) / "bare" (core only)
#   core_extra                    — extra cargo args forwarded only to the core check
#   cli_spectest_features_override— if non-empty, replaces the default
#                                   `--no-default-features --features jit` for the
#                                   cli and spectest cargo invocations. Used for
#                                   Thumb-2 variant rows so `thumb2-test` propagates
#                                   through all three packages.
DEFAULT_TARGETS=(
    "arm64-darwin|aarch64-apple-darwin|hosted||"
    "arm64-linux|aarch64-unknown-linux-gnu|hosted||"
    "arm64-none|aarch64-unknown-none|bare|--no-default-features --features jit|"
    "x64-darwin|x86_64-apple-darwin|hosted||"
    "x64-linux|x86_64-unknown-linux-gnu|hosted||"
    "x64-windows|x86_64-pc-windows-gnu|hosted||"
)

EXTRA_TARGETS=(
    "armv7-linux|armv7-unknown-linux-musleabihf|hosted||"
    # Same triple as armv7-linux, but builds with `--features thumb2-test` so
    # the arm32 backend emits Thumb-2 via `enc_t2.rs`. Cargo's feature
    # fingerprint distinguishes the two builds; we pin a separate
    # CARGO_TARGET_DIR per (label, package) in `run_package_check` so they
    # don't clobber each other.
    "armv7m-linux|armv7-unknown-linux-musleabihf|hosted|--no-default-features --features jit,thumb2-test|--no-default-features --features jit,thumb2-test"
)

if [[ $INCLUDE_ALL -eq 1 ]]; then
    TARGETS=("${DEFAULT_TARGETS[@]}" "${EXTRA_TARGETS[@]}")
else
    TARGETS=("${DEFAULT_TARGETS[@]}")
fi

ensure_installed() {
    local target="$1"
    if rustup target list --installed | grep -qx "$target"; then
        return 0
    fi
    if [[ $DO_INSTALL -eq 1 ]]; then
        echo "  installing rustup target $target ..."
        rustup target add "$target" >/dev/null 2>&1
    else
        return 1
    fi
}

LOG_DIR="$ROOT_DIR/target/check-targets-logs"
mkdir -p "$LOG_DIR"

BUILD_DIR="$ROOT_DIR/target/check-targets-build"
mkdir -p "$BUILD_DIR"

PACKAGES=(core cli spectest wasitest)

warning_count_from_log() {
    local log="$1"
    awk '
        match($0, /generated [0-9]+ warnings/) {
            text = substr($0, RSTART, RLENGTH)
            gsub(/[^0-9]/, "", text)
            total += text + 0
            found = 1
        }
        END {
            if (found) {
                print total
            } else {
                print 0
            }
        }
    ' "$log"
}

format_cell() {
    local status="$1"
    local warnings="${2:-0}"
    case "$status" in
        ok)
            if [[ "$warnings" -gt 0 ]]; then
                printf 'ok (%sw)' "$warnings"
            else
                printf 'ok'
            fi
            ;;
        fail)
            if [[ "$warnings" -gt 0 ]]; then
                printf 'FAIL (%sw)' "$warnings"
            else
                printf 'FAIL'
            fi
            ;;
        skip) printf 'SKIP' ;;
        na) printf 'n/a' ;;
        *) printf '%s' "$status" ;;
    esac
}

run_package_check() {
    local label="$1"
    local target="$2"
    local package_key="$3"
    local core_extra="$4"
    local cli_spectest_override="$5"
    local log="$LOG_DIR/$label.$package_key.log"
    local target_dir="$BUILD_DIR/$label-$package_key"
    local cargo_args=()

    case "$package_key" in
        core)
            cargo_args=(-p sf-nano-core)
            if [[ -n "$core_extra" ]]; then
                # shellcheck disable=SC2206  # split extra flags into argv
                cargo_args+=($core_extra)
            fi
            ;;
        cli)
            if [[ -n "$cli_spectest_override" ]]; then
                cargo_args=(-p sf-nano-cli)
                # shellcheck disable=SC2206
                cargo_args+=($cli_spectest_override)
            else
                cargo_args=(-p sf-nano-cli --no-default-features --features jit)
            fi
            ;;
        spectest)
            if [[ -n "$cli_spectest_override" ]]; then
                cargo_args=(-p sf-nano-spectest)
                # shellcheck disable=SC2206
                cargo_args+=($cli_spectest_override)
            else
                cargo_args=(-p sf-nano-spectest --no-default-features --features jit)
            fi
            ;;
        wasitest)
            cargo_args=(-p sf-nano-wasitest)
            ;;
        *)
            echo "internal error: unknown package key '$package_key'" >&2
            return 2
            ;;
    esac

    mkdir -p "$target_dir"
    if (cd "$ROOT_DIR" && CARGO_TARGET_DIR="$target_dir" cargo check --target "$target" "${cargo_args[@]}") >"$log" 2>&1; then
        return 0
    fi
    return 1
}

failure_details=()

printf '%-14s  %-32s  %-12s  %-12s  %-12s  %-12s\n' "label" "target" "core" "cli" "spectest" "wasitest"
printf '%-14s  %-32s  %-12s  %-12s  %-12s  %-12s\n' "-----" "------" "----" "---" "--------" "--------"

fail=0
for entry in "${TARGETS[@]}"; do
    IFS='|' read -r label target kind core_extra cli_spectest_override <<<"$entry"
    core_cell=""
    cli_cell=""
    spectest_cell=""
    wasitest_cell=""

    if ! ensure_installed "$target"; then
        core_cell=$(format_cell skip)
        if [[ "$kind" == "hosted" ]]; then
            cli_cell=$(format_cell skip)
            spectest_cell=$(format_cell skip)
            wasitest_cell=$(format_cell skip)
        else
            cli_cell=$(format_cell na)
            spectest_cell=$(format_cell na)
            wasitest_cell=$(format_cell na)
        fi
        printf '%-14s  %-32s  %-12s  %-12s  %-12s  %-12s\n' \
            "$label" "$target" "$core_cell" "$cli_cell" "$spectest_cell" "$wasitest_cell"
        continue
    fi

    if [[ "$kind" == "hosted" ]]; then
        for package_key in "${PACKAGES[@]}"; do
            if run_package_check "$label" "$target" "$package_key" "$core_extra" "$cli_spectest_override"; then
                pkg_status=ok
            else
                pkg_status=fail
                fail=1
                failure_details+=("$label/$package_key -> $LOG_DIR/$label.$package_key.log")
            fi
            pkg_warnings=$(warning_count_from_log "$LOG_DIR/$label.$package_key.log")
            case "$package_key" in
                core) core_cell=$(format_cell "$pkg_status" "$pkg_warnings") ;;
                cli) cli_cell=$(format_cell "$pkg_status" "$pkg_warnings") ;;
                spectest) spectest_cell=$(format_cell "$pkg_status" "$pkg_warnings") ;;
                wasitest) wasitest_cell=$(format_cell "$pkg_status" "$pkg_warnings") ;;
            esac
        done
    else
        if run_package_check "$label" "$target" core "$core_extra" ""; then
            core_cell=$(format_cell ok "$(warning_count_from_log "$LOG_DIR/$label.core.log")")
        else
            core_cell=$(format_cell fail "$(warning_count_from_log "$LOG_DIR/$label.core.log")")
            fail=1
            failure_details+=("$label/core -> $LOG_DIR/$label.core.log")
        fi
        cli_cell=$(format_cell na)
        spectest_cell=$(format_cell na)
        wasitest_cell=$(format_cell na)
    fi

    printf '%-14s  %-32s  %-12s  %-12s  %-12s  %-12s\n' \
        "$label" "$target" "$core_cell" "$cli_cell" "$spectest_cell" "$wasitest_cell"
done

# Bare-metal smoke crate: a standalone package that supplies stub
# implementations of the extern symbols declared in `runtime/os/none.rs`,
# letting `sf-nano-core` build cleanly on `target_os = "none"` triples.
# Checked against both pointer widths so a 32-bit-vs-64-bit signature
# drift in the embedder contract is caught — and so each (arch backend ×
# none.rs) combination compiles together:
#
#   aarch64-unknown-none       — arm64 backend  × runtime/os/none.rs
#   thumbv8m.main-none-eabihf  — arm32 (Thumb-2) backend × runtime/os/none.rs
#                                (Pico 2 / RP2350 Cortex-M33 and similar)
#
# Each row gets its own CARGO_TARGET_DIR so the two builds don't clobber
# each other's incremental cache.
BARE_SMOKE_TARGETS=(
    "aarch64-unknown-none"
    "thumbv8m.main-none-eabihf"
)
if [[ -d "$BARE_SMOKE_DIR" ]]; then
    for bare_target in "${BARE_SMOKE_TARGETS[@]}"; do
        log="$LOG_DIR/bare-smoke.$bare_target.log"
        target_dir="$BUILD_DIR/bare-smoke-$bare_target"
        if ! ensure_installed "$bare_target"; then
            printf '%-14s  %-32s  %-12s  %-12s  %-12s  %-12s\n' \
                "bare-smoke" "($bare_target)" "$(format_cell skip)" "$(format_cell na)" "$(format_cell na)" "$(format_cell na)"
            continue
        fi
        mkdir -p "$target_dir"
        if (cd "$BARE_SMOKE_DIR" && CARGO_TARGET_DIR="$target_dir" cargo check --target "$bare_target") >"$log" 2>&1; then
            printf '%-14s  %-32s  %-12s  %-12s  %-12s  %-12s\n' \
                "bare-smoke" "($bare_target)" "$(format_cell ok)" "$(format_cell na)" "$(format_cell na)" "$(format_cell na)"
        else
            printf '%-14s  %-32s  %-12s  %-12s  %-12s  %-12s\n' \
                "bare-smoke" "($bare_target)" "$(format_cell fail)" "$(format_cell na)" "$(format_cell na)" "$(format_cell na)"
            fail=1
            failure_details+=("bare-smoke/$bare_target -> $log")
        fi
    done
fi

if [[ ${#failure_details[@]} -gt 0 ]]; then
    echo
    echo "Failure logs:"
    for detail in "${failure_details[@]}"; do
        echo "  $detail"
    done
fi

exit $fail
