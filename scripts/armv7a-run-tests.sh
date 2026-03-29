#!/usr/bin/env bash
#
# Run the WASI benchmark tests on ARMv7 under QEMU inside Colima.
#
# Usage:
#   ./scripts/armv7a-run-tests.sh [--release]
#
# By default builds and runs in debug mode. Pass --release for an
# optimised build (much faster under QEMU).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=armv7-unknown-linux-musleabihf
PROFILE=debug
CARGO_PROFILE_FLAG=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release) PROFILE=release; CARGO_PROFILE_FLAG=(--release); shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--release]"; exit 1 ;;
    esac
done

CLI_BIN="$REPO_ROOT/target/$TARGET/$PROFILE/sf-nano-cli"
RUN_TESTS="$REPO_ROOT/benchmarks/wasi/run_tests.py"

cleanup() {
    if [[ "${COLIMA_STARTED:-}" = "1" ]]; then
        echo "[armv7-run-tests] Stopping Colima (we started it)..."
        colima stop 2>/dev/null || true
    fi
}
trap cleanup EXIT

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

echo "[armv7-run-tests] Cross-compiling CLI ($PROFILE) for ARMv7..."
cargo build --target "$TARGET" -p sf-nano-cli --no-default-features --features micro-jit \
    ${CARGO_PROFILE_FLAG[@]+"${CARGO_PROFILE_FLAG[@]}"}

if [[ ! -f "$CLI_BIN" ]]; then
    echo "ERROR: Build succeeded but binary not found at $CLI_BIN" >&2
    exit 1
fi

# --- Run ---

# run_tests.py runs on the host.  We point --exec at a small wrapper
# that invokes the ARMv7 binary under QEMU inside the Colima VM.
# Because Colima mounts the macOS home directory, host paths work directly.

WRAPPER=$(mktemp)
cat > "$WRAPPER" <<'WRAPPER_EOF'
#!/usr/bin/env bash
exec colima ssh -- qemu-arm-static "$@"
WRAPPER_EOF
chmod +x "$WRAPPER"
trap 'rm -f "$WRAPPER"; cleanup' EXIT

echo "[armv7-run-tests] Running benchmarks under QEMU ($PROFILE)..."
echo
python3 "$RUN_TESTS" --exec "$WRAPPER $CLI_BIN"
