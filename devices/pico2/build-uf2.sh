#!/usr/bin/env bash
#
# Build a flashable UF2 from a pico2 release binary.
#
# RP2350 supports two boot architectures: 2× Cortex-M33 (ARM) or 2×
# Hazard3 (RV32IMAC). The IMAGE_DEF block tells the boot ROM which
# core type to start, and the UF2 file declares its family ID
# accordingly:
#   ARM (thumbv8m.main-none-eabihf)        → family rp2350-arm-s,  0xE48BFF59
#   RV  (riscv32imac-unknown-none-elf)     → family rp2350-riscv,  0x1C40F63F
#
# Converter preference: picotool > elf2uf2-rs. picotool is the
# reference tool from pico-sdk and understands the `.bi_entries`
# IMAGE_DEF block the firmware emits via rp235x-hal's "binary-info"
# feature. elf2uf2-rs is the pure-Rust fallback installable via
# `cargo install elf2uf2-rs`.
#
# Usage:
#   ./build-uf2.sh [BIN_NAME] [ARCH] [DEMO] [ENGINE]
#     BIN_NAME defaults to demo_host
#     ARCH     defaults to arm; pass "rv" for the RV32 build
#     DEMO     defaults to mandelbrot; pass "cube" for the cube demo
#     ENGINE   defaults to jit; pass "interp" for the interpreter build
#
# Examples:
#   ./build-uf2.sh                                # demo_host, arm, jit
#   ./build-uf2.sh native_demo                    # native_demo, arm
#   ./build-uf2.sh demo_host rv                   # demo_host, RV32
#   ./build-uf2.sh demo_host arm cube             # demo_host cube, ARM
#   ./build-uf2.sh demo_host arm mandelbrot interp   # interpreter engine
#
# Output: <bin>.uf2 next to the release ELF.

set -euo pipefail

RP2350_ARM_S_FAMILY_HEX="0xE48BFF59"
RP2350_RISCV_FAMILY_HEX="0x1C40F63F"

find_python() {
    if command -v python3 &>/dev/null; then
        echo python3
    elif command -v python &>/dev/null; then
        echo python
    else
        return 1
    fi
}

patch_uf2_family() {
    local path="$1"
    local family_hex="$2"
    local py
    py="$(find_python)" || {
        echo "ERROR: python is required to correct elf2uf2-rs output for RP2350." >&2
        exit 1
    }

    "$py" - "$path" "$family_hex" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
family = int(sys.argv[2], 16)
data = bytearray(path.read_bytes())
if len(data) % 512 != 0:
    raise SystemExit(f"ERROR: {path} is not a valid UF2 size")

for offset in range(0, len(data), 512):
    data[offset + 28:offset + 32] = family.to_bytes(4, "little")

path.write_bytes(data)
PY
}

verify_uf2_family() {
    local path="$1"
    local family_hex="$2"
    local py
    py="$(find_python)" || {
        echo "ERROR: python is required to verify UF2 family IDs." >&2
        exit 1
    }

    "$py" - "$path" "$family_hex" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2], 16)
data = path.read_bytes()
if len(data) % 512 != 0:
    raise SystemExit(f"ERROR: {path} is not a valid UF2 size")

families = {int.from_bytes(data[offset + 28:offset + 32], "little")
            for offset in range(0, len(data), 512)}
if families != {expected}:
    found = ", ".join(f"0x{family:08X}" for family in sorted(families))
    raise SystemExit(f"ERROR: expected UF2 family 0x{expected:08X}, found {found}")
PY
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_NAME="${1:-demo_host}"
ARCH="${2:-arm}"
DEMO="${3:-mandelbrot}"
ENGINE="${4:-jit}"

# The crate takes `default-features = false` below, so the engine has to be
# named explicitly -- without one, sf-nano-core compiles with no execution
# engine at all and the build fails deep inside the library.
case "$ENGINE" in
    jit)    ENGINE_FEATURE="engine-jit" ;;
    interp) ENGINE_FEATURE="engine-interp" ;;
    *)
        echo "ERROR: unknown ENGINE '$ENGINE' — expected 'jit' or 'interp'." >&2
        exit 1
        ;;
esac

case "$ARCH" in
    arm)
        TARGET="thumbv8m.main-none-eabihf"
        FAMILY_NAME="rp2350-arm-s"
        FAMILY_HEX="$RP2350_ARM_S_FAMILY_HEX"
        ;;
    rv|riscv|rv32)
        TARGET="riscv32imac-unknown-none-elf"
        FAMILY_NAME="rp2350-riscv"
        FAMILY_HEX="$RP2350_RISCV_FAMILY_HEX"
        ;;
    *)
        echo "ERROR: unknown ARCH '$ARCH' — expected 'arm' or 'rv'." >&2
        exit 1
        ;;
esac

case "$DEMO" in
    mandelbrot)
        DEMO_FEATURE="demo-mandelbrot"
        ;;
    cube)
        DEMO_FEATURE="demo-cube"
        ;;
    *)
        echo "ERROR: unknown DEMO '$DEMO' — expected 'mandelbrot' or 'cube'." >&2
        exit 1
        ;;
esac

ELF_PATH="$SCRIPT_DIR/target/$TARGET/release/$BIN_NAME"
UF2_PATH="$ELF_PATH.uf2"

cd "$SCRIPT_DIR"

echo "[pico2-uf2] Building release ELF: $BIN_NAME ($ARCH → $TARGET, demo=$DEMO, engine=$ENGINE)"
cargo build --release --bin "$BIN_NAME" --target "$TARGET" --no-default-features --features "$DEMO_FEATURE,$ENGINE_FEATURE"

if [[ ! -f "$ELF_PATH" ]]; then
    echo "ERROR: expected ELF not found at $ELF_PATH" >&2
    exit 1
fi

if command -v picotool &>/dev/null; then
    echo "[pico2-uf2] Converting via picotool -> $UF2_PATH (family $FAMILY_NAME)"
    picotool uf2 convert "$ELF_PATH" "$UF2_PATH" --family "$FAMILY_NAME"
elif command -v elf2uf2-rs &>/dev/null; then
    echo "[pico2-uf2] Converting via elf2uf2-rs -> $UF2_PATH"
    elf2uf2-rs "$ELF_PATH" "$UF2_PATH"
    echo "[pico2-uf2] Correcting UF2 family -> $FAMILY_HEX"
    patch_uf2_family "$UF2_PATH" "$FAMILY_HEX"
else
    echo "ERROR: no UF2 converter found." >&2
    echo "  Install picotool: https://github.com/raspberrypi/picotool" >&2
    echo "  Or run:           cargo install elf2uf2-rs" >&2
    exit 1
fi

verify_uf2_family "$UF2_PATH" "$FAMILY_HEX"
SIZE_BYTES=$(wc -c < "$UF2_PATH")
printf "[pico2-uf2] Done: %s (%s bytes)\n" "$UF2_PATH" "$SIZE_BYTES"
echo "[pico2-uf2] Flash by dropping the file onto the RP2350 BOOTSEL drive,"
echo "            or run: picotool load -fx \"$UF2_PATH\""
