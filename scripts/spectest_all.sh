#!/usr/bin/env bash
#
# Build and run spectest for all supported native backends in both debug and
# release profiles: arm64, x64, emu64, emu32, then armv7.
# Stops on first failure.
#
# Usage:
#   ./scripts/spectest_all.sh [-- extra spectest args...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TESTSUITE_DIR="${TESTSUITE_DIR:-$REPO_ROOT/target/webassembly-testsuite-2.0}"
X64_TARGET=x86_64-apple-darwin
ARMV7_TARGET=armv7-unknown-linux-musleabihf
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

for cmd in cargo arch colima qemu-system-arm; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first." >&2
        exit 1
    fi
done

cleanup() {
    if [[ "${COLIMA_STARTED:-0}" = "1" ]]; then
        echo "[spectest_all] Stopping Colima (we started it)..."
        colima stop 2>/dev/null || true
    fi
}
trap cleanup EXIT

run_with_extra_args() {
    local -a cmd=("$@")
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi
    "${cmd[@]}"
}

ensure_colima_ready() {
    COLIMA_STARTED=0
    if ! colima status &>/dev/null; then
        echo "[spectest_all] Starting Colima..."
        colima start
        COLIMA_STARTED=1
    fi

    if ! colima ssh -- which qemu-arm-static &>/dev/null; then
        echo "[spectest_all] Installing qemu-user-static in Colima VM..."
        colima ssh -- sudo apt-get update -qq
        colima ssh -- sudo apt-get install -y -qq qemu-user-static
    fi
}

build_spectest() {
    local profile="$1"
    shift

    local -a cargo_args=("$@" -p sf-nano-spectest --no-default-features --features jit)
    if [[ "$profile" == "release" ]]; then
        cargo_args=(--release "${cargo_args[@]}")
    fi
    cargo build "${cargo_args[@]}"
}

profile_bin() {
    local profile="$1"
    local target="$2"

    if [[ -n "$target" ]]; then
        echo "$REPO_ROOT/target/$target/$profile/sf-nano-spectest"
    else
        echo "$REPO_ROOT/target/$profile/sf-nano-spectest"
    fi
}

run_profile() {
    local profile="$1"
    local host_bin x64_bin armv7_bin

    echo "=== Profile: $profile ==="
    echo

    echo "=== Building spectest ($profile, host) ==="
    build_spectest "$profile"
    host_bin="$(profile_bin "$profile" "")"
    echo

    echo "=== spectest: arm64 ($profile) ==="
    run_with_extra_args "$host_bin"
    echo

    echo "=== Building spectest ($profile, x64) ==="
    build_spectest "$profile" --target "$X64_TARGET"
    x64_bin="$(profile_bin "$profile" "$X64_TARGET")"
    if [[ ! -f "$x64_bin" ]]; then
        echo "ERROR: Build succeeded but binary not found at $x64_bin" >&2
        exit 1
    fi
    echo

    echo "=== spectest: x64 ($profile) ==="
    run_with_extra_args arch -x86_64 "$x64_bin" --backend native
    echo

    echo "=== spectest: emu64 ($profile) ==="
    run_with_extra_args "$host_bin" --emu64
    echo

    echo "=== spectest: emu32 ($profile) ==="
    run_with_extra_args "$host_bin" --emu32
    echo

    echo "=== Building spectest ($profile, armv7) ==="
    build_spectest "$profile" --target "$ARMV7_TARGET"
    armv7_bin="$(profile_bin "$profile" "$ARMV7_TARGET")"
    if [[ ! -f "$armv7_bin" ]]; then
        echo "ERROR: Build succeeded but binary not found at $armv7_bin" >&2
        exit 1
    fi
    echo

    echo "=== spectest: armv7 ($profile) ==="
    local -a armv7_cmd=(
        colima ssh --
        env
        "TESTSUITE_DIR=$TESTSUITE_DIR"
        qemu-arm-static
        "$armv7_bin"
        --backend
        native
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        armv7_cmd+=("${EXTRA_ARGS[@]}")
    fi
    "${armv7_cmd[@]}"
    echo
}

ensure_colima_ready

run_profile debug
run_profile release

echo "=== All spectests passed (debug + release) ==="
