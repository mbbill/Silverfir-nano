#!/usr/bin/env bash
#
# Build and run spectest for all supported native backends: arm64, x64, emu64,
# emu32, armv7a (A32 JIT), then armv7m (Thumb-2 JIT). By default both debug
# and release profiles are exercised. Stops on first failure.
#
# The armv7a and armv7m rows share the same target triple
# (armv7-unknown-linux-musleabihf); armv7m adds `--features thumb2-test` so
# the arm32 backend emits Thumb-2 via `enc_t2.rs`. Cargo's feature
# fingerprint keeps the two binaries distinct, but they write to the same
# `sf-nano-spectest` path — so we capture each binary to a separate bin
# before the next build clobbers it.
#
# Usage:
#   ./scripts/spectest_all.sh
#   ./scripts/spectest_all.sh --debug-only
#   ./scripts/spectest_all.sh --release-only
#   ./scripts/spectest_all.sh -- extra spectest args...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TESTSUITE_DIR="${TESTSUITE_DIR:-$REPO_ROOT/target/webassembly-testsuite}"
X64_TARGET=x86_64-apple-darwin
ARMV7_TARGET=armv7-unknown-linux-musleabihf
export TESTSUITE_DIR

EXTRA_ARGS=()
RUN_DEBUG=1
RUN_RELEASE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug-only)
            RUN_RELEASE=0
            shift
            ;;
        --release-only)
            RUN_DEBUG=0
            shift
            ;;
        -h|--help)
            sed -n '3,18p' "$0"
            exit 0
            ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--debug-only|--release-only] [-- extra spectest args...]" >&2
            exit 1
            ;;
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
    local features="${2:-jit}"
    shift
    if [[ $# -gt 0 ]]; then
        shift
    fi

    local -a cargo_args=("$@" -p sf-nano-spectest --no-default-features --features "$features")
    if [[ "$profile" == "release" ]]; then
        cargo_args=(--release "${cargo_args[@]}")
    fi
    cargo build "${cargo_args[@]}"
}

# Variant of `build_spectest` that adds `thumb2-test` to the feature list so
# the arm32 backend emits Thumb-2 instead of A32. Only meaningful for the
# armv7 target.
build_spectest_thumb2() {
    local profile="$1"
    shift

    local -a cargo_args=(
        "$@"
        -p sf-nano-spectest
        --no-default-features
        --features jit,thumb2-test
    )
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
    build_spectest "$profile" jit
    host_bin="$(profile_bin "$profile" "")"
    echo

    echo "=== spectest: arm64 ($profile) ==="
    run_with_extra_args "$host_bin"
    echo

    echo "=== Building spectest ($profile, x64) ==="
    build_spectest "$profile" jit --target "$X64_TARGET"
    x64_bin="$(profile_bin "$profile" "$X64_TARGET")"
    if [[ ! -f "$x64_bin" ]]; then
        echo "ERROR: Build succeeded but binary not found at $x64_bin" >&2
        exit 1
    fi
    echo

    echo "=== spectest: x64 ($profile) ==="
    run_with_extra_args arch -x86_64 "$x64_bin" --backend native
    echo

    echo "=== Building spectest ($profile, emu64) ==="
    build_spectest "$profile" jit,backend-emu64
    host_bin="$(profile_bin "$profile" "")"
    echo

    echo "=== spectest: emu64 ($profile) ==="
    run_with_extra_args "$host_bin"
    echo

    echo "=== Building spectest ($profile, emu32) ==="
    build_spectest "$profile" jit,backend-emu32
    host_bin="$(profile_bin "$profile" "")"
    echo

    echo "=== spectest: emu32 ($profile) ==="
    run_with_extra_args "$host_bin"
    echo

    echo "=== Building spectest ($profile, armv7a / A32 JIT) ==="
    build_spectest "$profile" jit --target "$ARMV7_TARGET"
    armv7_bin="$(profile_bin "$profile" "$ARMV7_TARGET")"
    if [[ ! -f "$armv7_bin" ]]; then
        echo "ERROR: Build succeeded but binary not found at $armv7_bin" >&2
        exit 1
    fi
    # Capture the A32 binary before the Thumb-2 build overwrites it at
    # the same output path.
    local armv7a_saved="$armv7_bin.armv7a"
    cp -f "$armv7_bin" "$armv7a_saved"
    echo

    echo "=== spectest: armv7a / A32 JIT ($profile) ==="
    local -a armv7a_cmd=(
        colima ssh --
        env
        "TESTSUITE_DIR=$TESTSUITE_DIR"
        qemu-arm-static
        "$armv7a_saved"
        --backend
        native
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        armv7a_cmd+=("${EXTRA_ARGS[@]}")
    fi
    "${armv7a_cmd[@]}"
    echo

    echo "=== Building spectest ($profile, armv7m / Thumb-2 JIT) ==="
    build_spectest_thumb2 "$profile" --target "$ARMV7_TARGET"
    armv7_bin="$(profile_bin "$profile" "$ARMV7_TARGET")"
    if [[ ! -f "$armv7_bin" ]]; then
        echo "ERROR: Build succeeded but binary not found at $armv7_bin" >&2
        exit 1
    fi
    local armv7m_saved="$armv7_bin.armv7m"
    cp -f "$armv7_bin" "$armv7m_saved"
    echo

    echo "=== spectest: armv7m / Thumb-2 JIT ($profile) ==="
    local -a armv7m_cmd=(
        colima ssh --
        env
        "TESTSUITE_DIR=$TESTSUITE_DIR"
        qemu-arm-static
        "$armv7m_saved"
        --backend
        native
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        armv7m_cmd+=("${EXTRA_ARGS[@]}")
    fi
    "${armv7m_cmd[@]}"
    echo
}

if [[ $RUN_DEBUG -eq 0 && $RUN_RELEASE -eq 0 ]]; then
    echo "ERROR: nothing to run (both profiles disabled)" >&2
    exit 2
fi

ensure_colima_ready

if [[ $RUN_DEBUG -eq 1 ]]; then
    run_profile debug
fi

if [[ $RUN_RELEASE -eq 1 ]]; then
    run_profile release
fi

if [[ $RUN_DEBUG -eq 1 && $RUN_RELEASE -eq 1 ]]; then
    echo "=== All spectests passed (debug + release) ==="
elif [[ $RUN_DEBUG -eq 1 ]]; then
    echo "=== All spectests passed (debug) ==="
else
    echo "=== All spectests passed (release) ==="
fi
