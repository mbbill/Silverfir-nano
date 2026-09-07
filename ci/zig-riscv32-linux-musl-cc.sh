#!/usr/bin/env sh
set -eu

# Translate rustc's generic linker inputs to Zig's rv32 build-std toolchain.
# The adapter passes all linker output through unchanged.
exec python3 "$(dirname "$0")/zig_riscv32_linker.py" "$@"
