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
else
    echo "ERROR: no UF2 converter found." >&2
    echo "  Install picotool: https://github.com/raspberrypi/picotool" >&2
    echo "  Or run:           cargo install elf2uf2-rs" >&2
    exit 1
fi

SIZE_BYTES=$(wc -c < "$UF2_PATH")
printf "[pico2-uf2] Done: %s (%s bytes)\n" "$UF2_PATH" "$SIZE_BYTES"
echo "[pico2-uf2] Flash by dropping the file onto the RP2350 BOOTSEL drive,"
echo "            or run: picotool load -fx \"$UF2_PATH\""
