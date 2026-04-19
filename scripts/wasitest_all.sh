#!/usr/bin/env bash
#
# Build and run the WASI testsuite across the supported backends.
#
# Coverage policy:
# - debug: host native + emu64 + emu32
# - release: host native + x64 + emu64 + emu32 + armv7a + armv7m
#
# This keeps the default `check_all.sh` runtime gate practical while still
# exercising every supported execution target in at least one optimized run.
#
# Usage:
#   ./scripts/wasitest_all.sh
#   ./scripts/wasitest_all.sh --debug-only
#   ./scripts/wasitest_all.sh --release-only
#   ./scripts/wasitest_all.sh -- --log-level debug socket

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
X64_TARGET=x86_64-apple-darwin
ARMV7_TARGET=armv7-unknown-linux-musleabihf

RUN_DEBUG=1
RUN_RELEASE=1
EXTRA_ARGS=()
COLIMA_STARTED=0
TEMP_FILES=()

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
            sed -n '3,16p' "$0"
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--debug-only|--release-only] [-- extra wasitest args...]" >&2
            exit 1
            ;;
    esac
done

if [[ $RUN_DEBUG -eq 0 && $RUN_RELEASE -eq 0 ]]; then
    echo "ERROR: nothing to run (both profiles disabled)" >&2
    exit 2
fi

cleanup() {
    local path
    if [[ ${#TEMP_FILES[@]} -gt 0 ]]; then
        for path in "${TEMP_FILES[@]}"; do
            rm -f "$path" 2>/dev/null || true
        done
    fi
    if [[ "$COLIMA_STARTED" = "1" ]]; then
        echo "[wasitest_all] Stopping Colima (we started it)..."
        colima stop 2>/dev/null || true
    fi
}
trap cleanup EXIT

need_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: $cmd not found. Install it first." >&2
        exit 1
    fi
}

build_cli() {
    local profile="$1"
    local target="$2"
    local features="$3"
    local -a cargo_args=(-p sf-nano-cli --no-default-features --features "$features")

    if [[ "$profile" == "release" ]]; then
        cargo_args=(--release "${cargo_args[@]}")
    fi
    if [[ -n "$target" ]]; then
        cargo_args=(--target "$target" "${cargo_args[@]}")
    fi

    cargo build "${cargo_args[@]}"
}

build_wasitest() {
    local profile="$1"
    local target="$2"
    local -a cargo_args=(-p sf-nano-wasitest)

    if [[ "$profile" == "release" ]]; then
        cargo_args=(--release "${cargo_args[@]}")
    fi
    if [[ -n "$target" ]]; then
        cargo_args=(--target "$target" "${cargo_args[@]}")
    fi

    cargo build "${cargo_args[@]}"
}

profile_bin() {
    local profile="$1"
    local target="$2"
    local package="$3"

    if [[ -n "$target" ]]; then
        echo "$REPO_ROOT/target/$target/$profile/$package"
    else
        echo "$REPO_ROOT/target/$profile/$package"
    fi
}

run_with_extra_args() {
    local -a cmd=("$@")
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi
    "${cmd[@]}"
}

ensure_colima_ready() {
    if ! colima status >/dev/null 2>&1; then
        echo "[wasitest_all] Starting Colima..."
        colima start
        COLIMA_STARTED=1
    fi

    if ! colima ssh -- which qemu-arm-static >/dev/null 2>&1; then
        echo "[wasitest_all] Installing qemu-user-static in Colima VM..."
        colima ssh -- sudo apt-get update -qq
        colima ssh -- sudo apt-get install -y -qq qemu-user-static
    fi
}

make_arm_wrapper() {
    local label="$1"
    local cli_bin="$2"
    local shared_tmpdir="$3"
    local home_dir="${HOME:-$REPO_ROOT}"
    local ssh_config="${COLIMA_SSH_CONFIG:-$HOME/.colima/ssh_config}"
    local wrapper
    wrapper="$(mktemp "/tmp/${label}.XXXXXX")"
    TEMP_FILES+=("$wrapper")
    cat >"$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SSH_CONFIG="$ssh_config"
REMOTE_HOST="colima"
CLI_BIN="$cli_bin"
SHARED_TMPDIR="$shared_tmpdir"
export HOME="$home_dir"

cleanup() {
    if [[ -n "\${ENV_FILE:-}" ]]; then
        rm -f "\$ENV_FILE" 2>/dev/null || true
    fi
    if [[ -n "\${ARGS_FILE:-}" ]]; then
        rm -f "\$ARGS_FILE" 2>/dev/null || true
    fi
    if [[ \${#REMOTE_CLEANUPS[@]} -gt 0 ]]; then
        ssh -F "\$SSH_CONFIG" "\$REMOTE_HOST" rm -rf "\${REMOTE_CLEANUPS[@]}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

mkdir -p "\$SHARED_TMPDIR"
ENV_FILE="\$(mktemp "\$SHARED_TMPDIR/${label}.env.XXXXXX")"
ARGS_FILE="\$(mktemp "\$SHARED_TMPDIR/${label}.args.XXXXXX")"
REMOTE_CLEANUPS=()

# Forward the sanitized test environment to the remote CLI verbatim, including
# values that contain embedded newlines.
while IFS= read -r name; do
    case "\$name" in
        HOME|PATH|PWD|OLDPWD|SHLVL|_)
            continue
            ;;
    esac
    printf '%s=%s\0' "\$name" "\${!name}" >> "\$ENV_FILE"
done < <(compgen -e)

forwarded_args=()
while [[ \$# -gt 0 ]]; do
    case "\$1" in
        --dir)
            if [[ \$# -lt 2 ]]; then
                echo "missing argument for --dir" >&2
                exit 2
            fi
            spec="\$2"
            shift 2
            if [[ "\$spec" == *::* ]]; then
                guest="\${spec%%::*}"
                host_path="\${spec#*::}"
            else
                guest="."
                host_path="\$spec"
            fi
            remote_dir="/tmp/${label}-preopen-\$\$-\${#REMOTE_CLEANUPS[@]}"
            ssh -F "\$SSH_CONFIG" "\$REMOTE_HOST" mkdir -p "\$remote_dir"
            if [[ -d "\$host_path" ]]; then
                ssh -T -F "\$SSH_CONFIG" "\$REMOTE_HOST" bash -s -- "\$host_path" "\$remote_dir" <<'SSH_COPY_DIR'
set -euo pipefail
cp -a "\$1"/. "\$2"/
SSH_COPY_DIR
            elif [[ -e "\$host_path" ]]; then
                ssh -T -F "\$SSH_CONFIG" "\$REMOTE_HOST" bash -s -- "\$host_path" "\$remote_dir" <<'SSH_COPY_FILE'
set -euo pipefail
cp -a "\$1" "\$2"/
SSH_COPY_FILE
            fi
            REMOTE_CLEANUPS+=("\$remote_dir")
            forwarded_args+=("--dir" "\$guest::\$remote_dir")
            ;;
        *)
            forwarded_args+=("\$1")
            shift
            ;;
    esac
done

# Forward argv through a NUL-delimited sidecar file so spaces/quotes survive
# the ssh remote-command shell hop unchanged.
for arg in "\${forwarded_args[@]}"; do
    printf '%s\0' "\$arg" >> "\$ARGS_FILE"
done

exec ssh -T -F "\$SSH_CONFIG" "\$REMOTE_HOST" bash -s -- "\$ENV_FILE" "\$ARGS_FILE" "\$CLI_BIN" <<'SSH_RUN'
set -euo pipefail
env_args=()
while IFS= read -r -d "" kv; do
    env_args+=("\$kv")
done < "\$1"
shift
remote_args=()
while IFS= read -r -d "" arg; do
    remote_args+=("\$arg")
done < "\$1"
shift
exec env -i "\${env_args[@]}" qemu-arm-static -cpu cortex-a15 "\$1" "\${remote_args[@]}"
SSH_RUN
EOF
    chmod +x "$wrapper"
    echo "$wrapper"
}

run_host_profile() {
    local profile="$1"
    local cli_bin wasitest_bin

    echo "=== WASI profile: $profile (host + emulator builds) ==="
    echo

    echo "=== Building WASI host binaries ($profile) ==="
    build_cli "$profile" "" "jit"
    build_wasitest "$profile" ""
    cli_bin="$(profile_bin "$profile" "" "sf-nano-cli")"
    wasitest_bin="$(profile_bin "$profile" "" "sf-nano-wasitest")"
    echo

    echo "=== wasitest: arm64 host ($profile) ==="
    run_with_extra_args "$wasitest_bin" --cli-path "$cli_bin" --backend native
    echo

    echo "=== Building WASI emu64 CLI ($profile) ==="
    build_cli "$profile" "" "jit,backend-emu64"
    cli_bin="$(profile_bin "$profile" "" "sf-nano-cli")"
    echo

    echo "=== wasitest: emu64 ($profile) ==="
    run_with_extra_args "$wasitest_bin" --cli-path "$cli_bin"
    echo

    echo "=== Building WASI emu32 CLI ($profile) ==="
    build_cli "$profile" "" "jit,backend-emu32"
    cli_bin="$(profile_bin "$profile" "" "sf-nano-cli")"
    echo

    echo "=== wasitest: emu32 ($profile) ==="
    run_with_extra_args "$wasitest_bin" --cli-path "$cli_bin"
    echo
}

run_release_cross_targets() {
    local cli_bin wasitest_bin saved_bin wrapper arm_tmpdir

    need_command arch
    need_command colima
    need_command ssh
    ensure_colima_ready
    arm_tmpdir="$REPO_ROOT/tmp/wasitest-all"
    mkdir -p "$arm_tmpdir"

    echo "=== WASI profile: release (cross-target backends) ==="
    echo

    echo "=== Building WASI x64 binaries (release) ==="
    build_cli release "$X64_TARGET" "jit"
    build_wasitest release "$X64_TARGET"
    cli_bin="$(profile_bin release "$X64_TARGET" "sf-nano-cli")"
    wasitest_bin="$(profile_bin release "$X64_TARGET" "sf-nano-wasitest")"
    echo

    echo "=== wasitest: x64 (release) ==="
    run_with_extra_args arch -x86_64 "$wasitest_bin" --cli-path "$cli_bin" --backend native
    echo

    echo "=== Building WASI armv7a CLI (release) ==="
    build_cli release "$ARMV7_TARGET" "jit"
    cli_bin="$(profile_bin release "$ARMV7_TARGET" "sf-nano-cli")"
    saved_bin="$cli_bin.armv7a"
    cp -f "$cli_bin" "$saved_bin"
    TEMP_FILES+=("$saved_bin")
    wrapper="$(make_arm_wrapper wasitest-armv7a "$saved_bin" "$arm_tmpdir")"
    echo

    echo "=== wasitest: armv7a / A32 JIT (release) ==="
    run_with_extra_args env TMPDIR="$arm_tmpdir" \
        "$REPO_ROOT/target/release/sf-nano-wasitest" --cli-path "$wrapper" --backend native
    echo

    echo "=== Building WASI armv7m CLI (release) ==="
    build_cli release "$ARMV7_TARGET" "jit,thumb2-test"
    cli_bin="$(profile_bin release "$ARMV7_TARGET" "sf-nano-cli")"
    saved_bin="$cli_bin.armv7m"
    cp -f "$cli_bin" "$saved_bin"
    TEMP_FILES+=("$saved_bin")
    wrapper="$(make_arm_wrapper wasitest-armv7m "$saved_bin" "$arm_tmpdir")"
    echo

    echo "=== wasitest: armv7m / Thumb-2 JIT (release) ==="
    run_with_extra_args env TMPDIR="$arm_tmpdir" \
        "$REPO_ROOT/target/release/sf-nano-wasitest" --cli-path "$wrapper" --backend native
    echo
}

need_command cargo

if [[ $RUN_DEBUG -eq 1 ]]; then
    run_host_profile debug
fi

if [[ $RUN_RELEASE -eq 1 ]]; then
    run_host_profile release
    run_release_cross_targets
fi

if [[ $RUN_DEBUG -eq 1 && $RUN_RELEASE -eq 1 ]]; then
    echo "=== All WASI tests passed (debug host+emu, release full matrix) ==="
elif [[ $RUN_DEBUG -eq 1 ]]; then
    echo "=== All WASI tests passed (debug host+emu) ==="
else
    echo "=== All WASI tests passed (release full matrix) ==="
fi
