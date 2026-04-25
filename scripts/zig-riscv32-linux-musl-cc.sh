#!/usr/bin/env sh
set -eu

exec zig cc -target riscv32-linux-musl -mcpu=generic_rv32+m+a+f+d+c -mabi=ilp32d "$@"
