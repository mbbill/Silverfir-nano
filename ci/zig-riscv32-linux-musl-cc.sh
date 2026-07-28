#!/usr/bin/env sh
set -eu

# Do not filter linker output. Correctness must see every diagnostic emitted by
# rustc or Zig; CI decides whether it is actionable.
exec zig cc -target riscv32-linux-musl "$@"
