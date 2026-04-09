#!/usr/bin/env bash
#
# Run the WASI benchmark tests on ARMv7 under QEMU inside Colima.
#
# Usage:
#   ./scripts/armv7a-run-tests.sh [--fast] [--release]
#
# By default builds and runs in debug mode. Pass --release for an
# optimised build (much faster under QEMU). Pass --fast to use the
# reduced-workload benchmark suite for slower ARMv7a/QEMU iteration.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=armv7-unknown-linux-musleabihf
PROFILE=debug
CARGO_PROFILE_FLAG=()
FAST_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast) FAST_RUN=1; shift ;;
        --release) PROFILE=release; CARGO_PROFILE_FLAG=(--release); shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--fast] [--release]"; exit 1 ;;
    esac
done

CLI_BIN="$REPO_ROOT/target/$TARGET/$PROFILE/sf-nano-cli"
if [[ "$FAST_RUN" -eq 1 ]]; then
    RUN_TESTS="$REPO_ROOT/benchmarks/wasi/run_tests_fast.py"
else
    RUN_TESTS="$REPO_ROOT/benchmarks/wasi/run_tests.py"
fi
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
' sh "qemu-arm-static $CLI_BIN" 2>/dev/null || true
}

cleanup_remote_qemu_pidfile() {
    local pid_file="$1"
    local qemu_pid=""

    if [[ ! -f "$pid_file" ]]; then
        return
    fi

    qemu_pid="$(tr -d '[:space:]' < "$pid_file" 2>/dev/null || true)"

    if [[ -z "$qemu_pid" ]] || ! colima status &>/dev/null; then
        return
    fi

    colima ssh -- sh -lc '
pid=$1
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$pid" >/dev/null 2>&1 || true
fi
' sh "$qemu_pid" 2>/dev/null || true
}

cleanup_remote_qemu_state_dir() {
    if [[ ! -d "$STATE_DIR" ]]; then
        return
    fi

    shopt -s nullglob
    for pid_file in "$STATE_DIR"/qemu-pid.*; do
        cleanup_remote_qemu_pidfile "$pid_file"
    done
    shopt -u nullglob
}

cleanup() {
    cleanup_remote_qemu_state_dir
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

cleanup_remote_qemu_state_dir
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

cleanup_remote_qemu_pidfile() {
    local pid_file="$1"
    local qemu_pid=""

    if [[ ! -f "$pid_file" ]]; then
        return
    fi

    qemu_pid="$(tr -d '[:space:]' < "$pid_file" 2>/dev/null || true)"

    if [[ -z "$qemu_pid" ]]; then
        return
    fi

    colima ssh -- sh -lc '
pid=$1
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$pid" >/dev/null 2>&1 || true
fi
' sh "$qemu_pid" >/dev/null 2>&1 || true
}

cli_bin="$1"
shift
state_dir="$(cd "$(dirname "$0")" && pwd)"
pid_file="$(mktemp "$state_dir/qemu-pid.XXXXXX")"
watch_pid=$$

(
    parent_pid=$watch_pid
    while kill -0 "$parent_pid" 2>/dev/null; do
        sleep 1
    done
    cleanup_remote_qemu_pidfile "$pid_file"
) >/dev/null 2>&1 &

exec colima ssh -- sh -lc '
pid_file=$1
shift
qemu-arm-static "$@" &
pid=$!
printf "%s\n" "$pid" > "$pid_file"
wait "$pid"
status=$?
exit "$status"
' sh "$pid_file" "$cli_bin" "$@"
WRAPPER_EOF
chmod +x "$WRAPPER"

if [[ "$FAST_RUN" -eq 1 ]]; then
    echo "[armv7-run-tests] Running reduced benchmarks under QEMU ($PROFILE)..."
else
    echo "[armv7-run-tests] Running benchmarks under QEMU ($PROFILE)..."
fi
echo
if [[ "$FAST_RUN" -eq 1 ]]; then
    python3 "$RUN_TESTS" --exec "$WRAPPER" --cli-args "$CLI_BIN"
else
    python3 "$RUN_TESTS" --exec "$WRAPPER $CLI_BIN"
fi
