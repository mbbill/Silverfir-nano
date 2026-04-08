#!/usr/bin/env bash
#
# check-targets.sh — cross-compile `cargo check` sf-nano-core against every
# supported (target_arch × target_os) combination in the host's installed
# rustup target set.
#
# What it does
# ------------
# Runs `cargo check --target <t>` for each target in the matrix below and
# reports pass/fail. Non-arm64 targets are currently skipped by default
# because the x86_64 and armv7a backends on this branch lag behind the
# arm64 + common pipeline refactor and do not build — the failures swamp
# any real OS-layer regressions. Pass `--all` to exercise them anyway.
#
# Usage
# -----
#   scripts/check-targets.sh            # arm64 targets only (default)
#   scripts/check-targets.sh --all      # include non-arm64 backends (noisy)
#   scripts/check-targets.sh --install  # rustup target add anything missing
#
# Exit status is 0 only if every selected target passed.
#
# Target matrix
# -------------
# | target                             | clean today? | exercises                |
# |------------------------------------|--------------|---------------------------|
# | aarch64-apple-darwin (host)        | yes          | runtime/os/macos.rs       |
# | aarch64-unknown-linux-gnu          | yes          | runtime/os/linux.rs       |
# | aarch64-unknown-none               | yes (1)      | runtime/os/none.rs        |
# | x86_64-apple-darwin                | no (2)       | runtime/os/macos.rs       |
# | x86_64-unknown-linux-gnu           | no (2)       | runtime/os/linux.rs + x64 trap handler |
# | x86_64-pc-windows-gnu              | no (2)       | runtime/os/windows.rs     |
# | armv7-unknown-linux-musleabihf     | no (3)       | runtime/os/linux.rs (32-bit) |
#
# (1) Must be built with --no-default-features --features jit (no guard-pages on bare-metal).
# (2) Fails on pre-existing x86_64 backend drift, unrelated to the OS layer.
# (3) Fails on pre-existing armv7a backend drift, unrelated to the OS layer.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORE_DIR="$ROOT_DIR/sf-nano-core"
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

# Each entry: "label|target|extra cargo args"
ARM64_TARGETS=(
    "arm64-darwin|aarch64-apple-darwin|"
    "arm64-linux|aarch64-unknown-linux-gnu|"
    "arm64-none|aarch64-unknown-none|--no-default-features --features jit"
)

NON_ARM64_TARGETS=(
    "x64-darwin|x86_64-apple-darwin|"
    "x64-linux|x86_64-unknown-linux-gnu|"
    "x64-windows|x86_64-pc-windows-gnu|"
    "armv7-linux|armv7-unknown-linux-musleabihf|"
)

if [[ $INCLUDE_ALL -eq 1 ]]; then
    TARGETS=("${ARM64_TARGETS[@]}" "${NON_ARM64_TARGETS[@]}")
else
    TARGETS=("${ARM64_TARGETS[@]}")
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

printf '%-14s  %-32s  %s\n' "label" "target" "result"
printf '%-14s  %-32s  %s\n' "-----" "------" "------"

fail=0
for entry in "${TARGETS[@]}"; do
    IFS='|' read -r label target extra <<<"$entry"
    log="$LOG_DIR/$label.log"

    if ! ensure_installed "$target"; then
        printf '%-14s  %-32s  %s\n' "$label" "$target" "SKIP (target not installed; rerun with --install)"
        continue
    fi

    # shellcheck disable=SC2086  # $extra is intentionally word-split
    if (cd "$CORE_DIR" && cargo check --target "$target" $extra) >"$log" 2>&1; then
        printf '%-14s  %-32s  %s\n' "$label" "$target" "ok"
    else
        printf '%-14s  %-32s  %s\n' "$label" "$target" "FAIL (see $log)"
        fail=1
    fi
done

# Bare-metal smoke crate: a standalone package that pins its own target
# in `.cargo/config.toml` and exercises `runtime/os/none.rs` by providing
# the embedder-side extern symbols. Run as a sibling check so it's part
# of the same pass/fail gate.
if [[ -d "$BARE_SMOKE_DIR" ]]; then
    log="$LOG_DIR/bare-smoke.log"
    if (cd "$BARE_SMOKE_DIR" && cargo check) >"$log" 2>&1; then
        printf '%-14s  %-32s  %s\n' "bare-smoke" "(aarch64-unknown-none)" "ok"
    else
        printf '%-14s  %-32s  %s\n' "bare-smoke" "(aarch64-unknown-none)" "FAIL (see $log)"
        fail=1
    fi
fi

exit $fail
