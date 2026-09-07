"""Adapt rustc's ELF linker arguments for the rv32 Zig build-std target."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


def absent_prebuilt_std_directory(value: str) -> bool:
    path = Path(value)
    # This tier-3 target builds std into Cargo's target directory. rustc also
    # supplies a prebuilt-std search path, although rustup ships no std there.
    # Keep existing directories and every other search path, including invalid
    # user paths: their diagnostics must remain visible.
    return (
        path.is_absolute()
        and path.parts[-4:]
        == ("lib", "rustlib", "riscv32gc-unknown-linux-musl", "lib")
        and not path.exists()
    )


def zig_arguments(arguments: Sequence[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-Wl,-O1":
            # rustc requests generic ELF linker optimization in release mode;
            # Zig ignores this deprecated setting. Do not send an unsupported
            # option to the linker, rather than filtering its resulting warning.
            index += 1
            continue
        if argument == "-L" and index + 1 < len(arguments):
            if not absent_prebuilt_std_directory(arguments[index + 1]):
                result.extend(arguments[index : index + 2])
            index += 2
            continue
        elif argument.startswith("-L") and absent_prebuilt_std_directory(argument[2:]):
            index += 1
            continue
        result.append(argument)
        index += 1
    return result


def main() -> None:
    os.execvp("zig", ["zig", "cc", "-target", "riscv32-linux-musl", *zig_arguments(sys.argv[1:])])


if __name__ == "__main__":
    main()
