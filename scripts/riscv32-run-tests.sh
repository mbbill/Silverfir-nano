#!/usr/bin/env bash
#
# Deprecated direct entry point. Prefer `python3 scripts/check.py`; this
# script is kept as a low-level helper.
#
# Run the WASI benchmark tests on RV32GC under QEMU. On macOS this uses
# qemu-riscv32-static inside Colima; on Linux/WSL it uses qemu-riscv32-static
# directly from PATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=riscv32gc-unknown-linux-musl
PROFILE=release
CARGO_PROFILE_FLAG=(--release)
RV32_RUSTFLAGS="-C linker=$REPO_ROOT/scripts/zig-riscv32-linux-musl-cc.sh -C target-feature=+crt-static -C link-self-contained=no -C panic=abort"
export ZIG_GLOBAL_CACHE_DIR="${ZIG_GLOBAL_CACHE_DIR:-$REPO_ROOT/target/zig-cache}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) PROFILE=debug; CARGO_PROFILE_FLAG=(); shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--debug]"; exit 1 ;;
    esac
done

CLI_BIN="$REPO_ROOT/target/$TARGET/$PROFILE/sf-nano-cli"
RUN_TESTS="$REPO_ROOT/benchmarks/wasi/run_tests.py"
STATE_DIR="$(mktemp -d /tmp/riscv32-run-tests.XXXXXX)"
WRAPPER=""
USE_COLIMA=0
COLIMA_STARTED=0

append_rustflags() {
    if [[ -n "${RUSTFLAGS:-}" ]]; then
        export RUSTFLAGS="$RUSTFLAGS $RV32_RUSTFLAGS"
    else
        export RUSTFLAGS="$RV32_RUSTFLAGS"
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
' sh "qemu-riscv32-static -cpu rv32 $CLI_BIN" 2>/dev/null || true
}

cleanup() {
    cleanup_remote_qemu_pattern
    if [[ "$STATE_DIR" == /tmp/riscv32-run-tests.* ]]; then
        rm -rf "$STATE_DIR"
    fi
    if [[ "$COLIMA_STARTED" = "1" ]]; then
        echo "[riscv32-run-tests] Stopping Colima (we started it)..."
        colima stop 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for cmd in cargo zig python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. RV32 builds require nightly cargo, Zig, and Python." >&2
        exit 1
    fi
done

if ! cargo +nightly --version &>/dev/null; then
    echo "ERROR: cargo +nightly is required for RV32 build-std." >&2
    exit 1
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
            echo "[riscv32-run-tests] Starting Colima..."
            colima start
            COLIMA_STARTED=1
        fi
        if ! colima ssh -- which qemu-riscv32-static &>/dev/null; then
            echo "[riscv32-run-tests] Installing qemu-user-static in Colima VM..."
            colima ssh -- sudo apt-get update -qq
            colima ssh -- sudo apt-get install -y -qq qemu-user-static
        fi
        ;;
    Linux)
        if ! command -v qemu-riscv32-static &>/dev/null; then
            echo "ERROR: qemu-riscv32-static not found. Install qemu-user-static." >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: unsupported host OS '$HOST_OS'; use macOS+Colima or Linux/WSL." >&2
        exit 1
        ;;
esac

cleanup_remote_qemu_pattern

echo "[riscv32-run-tests] Cross-compiling CLI ($PROFILE) for RV32GC..."
append_rustflags
cargo +nightly build -Z build-std=std,panic_abort --target "$TARGET" \
    -p sf-nano-cli --no-default-features --features jit \
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

exec colima ssh -- env -i "\${env_args[@]}" /usr/bin/qemu-riscv32-static -cpu rv32 "\$cli_bin" "\$@"
WRAPPER_EOF
else
    cat > "$WRAPPER" <<'WRAPPER_EOF'
#!/usr/bin/env bash
set -euo pipefail
cli_bin="$1"
shift

exec qemu-riscv32-static -cpu rv32 "$cli_bin" "$@"
WRAPPER_EOF
fi
chmod +x "$WRAPPER"

echo "[riscv32-run-tests] Running benchmarks under QEMU ($PROFILE)..."
echo
python3 "$RUN_TESTS" --exec "$WRAPPER" --cli-args "$CLI_BIN"
