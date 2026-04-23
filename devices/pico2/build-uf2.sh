#!/usr/bin/env bash
#
# Build a flashable UF2 from a pico2 release binary.
#
# The thumbv8m.main-none-eabihf release ELF is ~14 MiB on disk but
# almost all of that is DWARF debug info. The actual loadable image
# (.text + .rodata + .vector_table + .bi_entries + .data) is under
# 1 MiB. This script runs `cargo build --release` for the chosen
# binary, then converts the ELF to RP2350 UF2 format for USB-BOOTSEL
# drag-and-drop or `picotool load`.
#
# Converter preference: picotool > elf2uf2-rs. picotool is the
# reference tool from pico-sdk and understands the `.bi_entries`
# IMAGE_DEF block the firmware emits via rp235x-hal's "binary-info"
# feature. elf2uf2-rs is the pure-Rust fallback installable via
# `cargo install elf2uf2-rs`.
#
# Usage:
#   ./build-uf2.sh                     # mandelbrot_wasm (default-run)
#   ./build-uf2.sh heartbeat
#   ./build-uf2.sh lcd_demo
#   ./build-uf2.sh mandelbrot_native
#
# Output: <bin>.uf2 next to the release ELF.

set -euo pipefail

RP2350_ARM_S_FAMILY_HEX="0xE48BFF59"

find_python() {
    if command -v python3 &>/dev/null; then
        echo python3
    elif command -v python &>/dev/null; then
        echo python
    else
        return 1
    fi
}

patch_uf2_family_to_rp2350() {
    local path="$1"
    local py
    py="$(find_python)" || {
        echo "ERROR: python is required to correct elf2uf2-rs output for RP2350." >&2
        exit 1
    }

    "$py" - "$path" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = bytearray(path.read_bytes())
if len(data) % 512 != 0:
    raise SystemExit(f"ERROR: {path} is not a valid UF2 size")

rp2350_arm_s = 0xE48BFF59
for offset in range(0, len(data), 512):
    data[offset + 28:offset + 32] = rp2350_arm_s.to_bytes(4, "little")

path.write_bytes(data)
PY
}

verify_uf2_family() {
    local path="$1"
    local py
    py="$(find_python)" || {
        echo "ERROR: python is required to verify UF2 family IDs." >&2
        exit 1
    }

    "$py" - "$path" "$RP2350_ARM_S_FAMILY_HEX" <<'PY'
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
BIN_NAME="${1:-mandelbrot_wasm}"
TARGET="thumbv8m.main-none-eabihf"
ELF_PATH="$SCRIPT_DIR/target/$TARGET/release/$BIN_NAME"
UF2_PATH="$ELF_PATH.uf2"

cd "$SCRIPT_DIR"

echo "[pico2-uf2] Building release ELF: $BIN_NAME"
cargo build --release --bin "$BIN_NAME"

if [[ ! -f "$ELF_PATH" ]]; then
    echo "ERROR: expected ELF not found at $ELF_PATH" >&2
    exit 1
fi

if command -v picotool &>/dev/null; then
    echo "[pico2-uf2] Converting via picotool -> $UF2_PATH"
    picotool uf2 convert "$ELF_PATH" "$UF2_PATH" --family rp2350-arm-s
elif command -v elf2uf2-rs &>/dev/null; then
    echo "[pico2-uf2] Converting via elf2uf2-rs -> $UF2_PATH"
    elf2uf2-rs "$ELF_PATH" "$UF2_PATH"
    echo "[pico2-uf2] Correcting UF2 family -> $RP2350_ARM_S_FAMILY_HEX"
    patch_uf2_family_to_rp2350 "$UF2_PATH"
else
    echo "ERROR: no UF2 converter found." >&2
    echo "  Install picotool: https://github.com/raspberrypi/picotool" >&2
    echo "  Or run:           cargo install elf2uf2-rs" >&2
    exit 1
fi

verify_uf2_family "$UF2_PATH"
SIZE_BYTES=$(wc -c < "$UF2_PATH")
printf "[pico2-uf2] Done: %s (%s bytes)\n" "$UF2_PATH" "$SIZE_BYTES"
echo "[pico2-uf2] Flash by dropping the file onto the RP2350 BOOTSEL drive,"
echo "            or run: picotool load -fx \"$UF2_PATH\""
