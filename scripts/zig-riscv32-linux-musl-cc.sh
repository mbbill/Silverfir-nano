#!/usr/bin/env sh
set -eu

# Zig supplies the musl C toolchain rust-lld cannot self-contain for rv32.
#
# One line of zig's stderr is dropped, and only this one:
#
#   unable to open library directory '<sysroot>/lib/rustlib/
#     riscv32gc-unknown-linux-musl/lib': FileNotFound
#
# rv32 is built with `-Z build-std`, which is precisely the case where no
# prebuilt std has been installed for the target, so that directory does not
# exist and cannot. rustc passes it as a `-L` anyway, zig reports it, and
# since nightly-2026-07-24 rustc relays linker stderr as a rustc `warning:`.
# The result was one warning on every rv32 row that links -- three of them in
# check-linux -- describing a condition that is both expected and harmless.
#
# The pattern is deliberately narrow: it requires the `rustlib` sysroot path,
# so a genuine "unable to open library directory" for any other path is still
# reported. zig's exit status is preserved, so a real link failure still fails.

err=$(mktemp)
# shellcheck disable=SC2064  # expand $err now: it is what needs cleaning up
trap "rm -f '$err'" EXIT

if zig cc -target riscv32-linux-musl -mcpu=generic_rv32+m+a+f+d+c \
        -mabi=ilp32d "$@" 2>"$err"; then
    status=0
else
    status=$?
fi

# grep exits 1 when everything was filtered (or there was nothing); that is a
# normal outcome here and must not become this script's status.
grep -Ev "unable to open library directory .*rustlib" "$err" >&2 || true

exit "$status"
