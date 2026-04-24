#!/usr/bin/env bash
#
# Deprecated direct entry point. Prefer `python3 scripts/check.py full`; this
# script is kept as a low-level helper.
#
# Run the WASI benchmark tests on ARMv7 under QEMU inside Colima.
#
# Usage:
#   ./scripts/armv7a-run-tests.sh [--full] [--debug]
#
# By default builds a release binary and runs the reduced-workload
# (fast) benchmark suite. Pass --full to run the complete benchmark
# suite. Pass --debug to build in debug mode (much slower under QEMU).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=armv7-unknown-linux-musleabihf
PROFILE=release
CARGO_PROFILE_FLAG=(--release)
FAST_RUN=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) FAST_RUN=0; shift ;;
        --debug) PROFILE=debug; CARGO_PROFILE_FLAG=(); shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--full] [--debug]"; exit 1 ;;
    esac
done

CLI_BIN="$REPO_ROOT/target/$TARGET/$PROFILE/sf-nano-cli"
RUN_TESTS="$REPO_ROOT/benchmarks/wasi/run_tests.py"
STATE_DIR="$(mktemp -d /tmp/armv7a-run-tests.XXXXXX)"
WRAPPER=""

cleanup_remote_qemu_pattern() {
    if ! colima status &>/dev/null; then
        return
    fi

    colima ssh -- sh -lc '
pattern=$1
pids=$(ps -eo pid=,args= | awk -v needle="$pattern" '"'"'index($0, needle) { print $1 }'"'"')
if [ -n "$pids" ]; then
    kill $pids >/dev/null 2>&1 || true
    sleep 1
    kill -9 $pids >/dev/null 2>&1 || true
fi
' sh "qemu-arm-static -cpu cortex-a15 $CLI_BIN" 2>/dev/null || true
}

cleanup() {
    cleanup_remote_qemu_pattern
    if [[ "${COLIMA_STARTED:-}" = "1" ]]; then
        echo "[armv7-run-tests] Stopping Colima (we started it)..."
        colima stop 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --- Prerequisites ---

for cmd in colima cargo python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first." >&2
        exit 1
    fi
done

# --- Start Colima if needed ---

COLIMA_STARTED=0
if ! colima status &>/dev/null; then
    echo "[armv7-run-tests] Starting Colima..."
    colima start
    COLIMA_STARTED=1
fi

if ! colima ssh -- which qemu-arm-static &>/dev/null; then
    echo "[armv7-run-tests] Installing qemu-user-static in Colima VM..."
    colima ssh -- sudo apt-get update -qq
    colima ssh -- sudo apt-get install -y -qq qemu-user-static
fi

# --- Build ---

cleanup_remote_qemu_pattern

echo "[armv7-run-tests] Cross-compiling CLI ($PROFILE) for ARMv7..."
cargo build --target "$TARGET" -p sf-nano-cli --no-default-features --features jit \
    ${CARGO_PROFILE_FLAG[@]+"${CARGO_PROFILE_FLAG[@]}"}

if [[ ! -f "$CLI_BIN" ]]; then
    echo "ERROR: Build succeeded but binary not found at $CLI_BIN" >&2
    exit 1
fi

# --- Run ---

# run_tests.py runs on the host.  We point --exec at a small wrapper
# that invokes the ARMv7 binary under QEMU inside the Colima VM.
# Because Colima mounts the macOS home directory, host paths work directly.

WRAPPER=$(mktemp "$STATE_DIR/qemu-wrapper.XXXXXX")
cat > "$WRAPPER" <<'WRAPPER_EOF'
#!/usr/bin/env bash
set -euo pipefail

# Run qemu in the foreground via `exec` so stdin (needed by c-ray's scene
# input) flows through unchanged. Previously this wrapper backgrounded qemu
# with `& wait` to stash its pid for cross-process cleanup, but POSIX shells
# redirect backgrounded-job stdin from /dev/null in non-interactive contexts,
# which caused c-ray to see an empty scene and render a blank background.
# Orphan qemu processes are cleaned up via the pattern-based sweep in the
# parent script (cleanup_remote_qemu_pattern).
cli_bin="$1"
shift

exec colima ssh -- sh -lc 'exec qemu-arm-static -cpu cortex-a15 "$@"' \
    sh "$cli_bin" "$@"
WRAPPER_EOF
chmod +x "$WRAPPER"

FAST_FLAG=""
if [[ "$FAST_RUN" -eq 1 ]]; then
    FAST_FLAG="--fast"
    echo "[armv7-run-tests] Running benchmarks under QEMU ($PROFILE, fast)..."
else
    echo "[armv7-run-tests] Running benchmarks under QEMU ($PROFILE, full)..."
fi
echo
python3 "$RUN_TESTS" $FAST_FLAG --exec "$WRAPPER" --cli-args "$CLI_BIN"
