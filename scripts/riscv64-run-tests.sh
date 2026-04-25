#!/usr/bin/env bash
#
# Deprecated direct entry point. Prefer `python3 scripts/check.py full`; this
# script is kept as a low-level helper.
#
# Run the WASI benchmark tests on RV64GC under QEMU. On macOS this uses
# qemu-riscv64-static inside Colima; on Linux/WSL it uses qemu-riscv64-static
# directly from PATH.
#
# Usage:
#   ./scripts/riscv64-run-tests.sh [--full] [--debug]
#
# By default builds a release binary and runs the reduced-workload
# (fast) benchmark suite. Pass --full to run the complete benchmark
# suite. Pass --debug to build in debug mode (much slower under QEMU).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=riscv64gc-unknown-linux-musl
PROFILE=release
CARGO_PROFILE_FLAG=(--release)
FAST_RUN=1
RV64_RUSTFLAGS="-C linker=rust-lld -C target-feature=+crt-static -C link-self-contained=yes -C panic=abort"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) FAST_RUN=0; shift ;;
        --debug) PROFILE=debug; CARGO_PROFILE_FLAG=(); shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--full] [--debug]"; exit 1 ;;
    esac
done

CLI_BIN="$REPO_ROOT/target/$TARGET/$PROFILE/sf-nano-cli"
RUN_TESTS="$REPO_ROOT/benchmarks/wasi/run_tests.py"
STATE_DIR="$(mktemp -d /tmp/riscv64-run-tests.XXXXXX)"
WRAPPER=""
USE_COLIMA=0
COLIMA_STARTED=0

append_rustflags() {
    if [[ -n "${RUSTFLAGS:-}" ]]; then
        export RUSTFLAGS="$RUSTFLAGS $RV64_RUSTFLAGS"
    else
        export RUSTFLAGS="$RV64_RUSTFLAGS"
    fi
}

cleanup_remote_qemu_pattern() {
    if [[ "$USE_COLIMA" -ne 1 ]]; then
        return
    fi
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
' sh "qemu-riscv64-static -cpu rv64 $CLI_BIN" 2>/dev/null || true
}

cleanup() {
    cleanup_remote_qemu_pattern
    if [[ "$STATE_DIR" == /tmp/riscv64-run-tests.* ]]; then
        rm -rf "$STATE_DIR"
    fi
    if [[ "$COLIMA_STARTED" = "1" ]]; then
        echo "[riscv64-run-tests] Stopping Colima (we started it)..."
        colima stop 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for cmd in cargo rustup python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first." >&2
        exit 1
    fi
done

if ! rustup target list --installed | grep -qx "$TARGET"; then
    echo "[riscv64-run-tests] Installing Rust target $TARGET..."
    rustup target add "$TARGET"
fi

HOST_OS="$(uname -s)"
case "$HOST_OS" in
    Darwin)
        USE_COLIMA=1
        if ! command -v colima &>/dev/null; then
            echo "ERROR: colima not found. Install Colima or run on Linux/WSL." >&2
            exit 1
        fi
        if ! colima status &>/dev/null; then
            echo "[riscv64-run-tests] Starting Colima..."
            colima start
            COLIMA_STARTED=1
        fi
        if ! colima ssh -- which qemu-riscv64-static &>/dev/null; then
            echo "[riscv64-run-tests] Installing qemu-user-static in Colima VM..."
            colima ssh -- sudo apt-get update -qq
            colima ssh -- sudo apt-get install -y -qq qemu-user-static
        fi
        ;;
    Linux)
        if ! command -v qemu-riscv64-static &>/dev/null; then
            echo "ERROR: qemu-riscv64-static not found. Install qemu-user-static." >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: unsupported host OS '$HOST_OS'; use macOS+Colima or Linux/WSL." >&2
        exit 1
        ;;
esac

cleanup_remote_qemu_pattern

echo "[riscv64-run-tests] Cross-compiling CLI ($PROFILE) for RV64GC..."
append_rustflags
cargo build --target "$TARGET" -p sf-nano-cli --no-default-features --features jit \
    ${CARGO_PROFILE_FLAG[@]+"${CARGO_PROFILE_FLAG[@]}"}

if [[ ! -f "$CLI_BIN" ]]; then
    echo "ERROR: Build succeeded but binary not found at $CLI_BIN" >&2
    exit 1
fi

WRAPPER=$(mktemp "$STATE_DIR/qemu-wrapper.XXXXXX")
if [[ "$USE_COLIMA" -eq 1 ]]; then
    HOST_HOME="${HOME:-$(cd ~ && pwd)}"
    cat > "$WRAPPER" <<WRAPPER_EOF
#!/usr/bin/env bash
set -euo pipefail
cli_bin="\$1"
shift

export HOME="$HOST_HOME"
env_args=()
while IFS= read -r key; do
    case "\$key" in
        _|HOME|PWD|SHLVL) continue ;;
    esac
    env_args+=("\$key=\${!key}")
done < <(compgen -e)

exec colima ssh -- env -i "\${env_args[@]}" /usr/bin/qemu-riscv64-static -cpu rv64 "\$cli_bin" "\$@"
WRAPPER_EOF
else
    cat > "$WRAPPER" <<'WRAPPER_EOF'
#!/usr/bin/env bash
set -euo pipefail
cli_bin="$1"
shift

exec qemu-riscv64-static -cpu rv64 "$cli_bin" "$@"
WRAPPER_EOF
fi
chmod +x "$WRAPPER"

FAST_FLAG=""
if [[ "$FAST_RUN" -eq 1 ]]; then
    FAST_FLAG="--fast"
    echo "[riscv64-run-tests] Running benchmarks under QEMU ($PROFILE, fast)..."
else
    echo "[riscv64-run-tests] Running benchmarks under QEMU ($PROFILE, full)..."
fi
echo
python3 "$RUN_TESTS" $FAST_FLAG --exec "$WRAPPER" --cli-args "$CLI_BIN"
