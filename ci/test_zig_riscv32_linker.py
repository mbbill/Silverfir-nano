from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ci.zig_riscv32_linker import zig_arguments


class ZigRiscv32LinkerTests(unittest.TestCase):
    def test_build_std_keeps_real_libraries_and_removes_only_absent_prebuilt_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            absent = root / "sysroot/lib/rustlib/riscv32gc-unknown-linux-musl/lib"
            built = root / "cargo target/debug/deps"
            built.mkdir(parents=True)
            args = ["main.o", "-L", str(absent), "-L", str(built), "-lstd", "-o", "out"]
            self.assertEqual(zig_arguments(args), ["main.o", "-L", str(built), "-lstd", "-o", "out"])
            self.assertEqual(zig_arguments([f"-L{absent}", "-lstd"]), ["-lstd"])
            absent.mkdir(parents=True)
            self.assertEqual(zig_arguments(args), args)

    def test_unknown_inputs_remain_diagnosable(self) -> None:
        args = [
            "-L", "/missing/lib/rustlib/riscv64gc-unknown-linux-musl/lib",
            "-L/missing/user-library", "-Wl,--unknown-option", "-Wl,-O2",
            "-lmissing", "@link-arguments", "-L",
        ]
        self.assertEqual(zig_arguments(args), args)
        self.assertEqual(zig_arguments(["-L", "-Wl,-O1"]), ["-L", "-Wl,-O1"])

    def test_only_ignored_generic_optimization_setting_is_removed(self) -> None:
        self.assertEqual(zig_arguments(["-Wl,-O1", "main.o", "-O1", "-o", "out"]),
                         ["main.o", "-O1", "-o", "out"])


if __name__ == "__main__":
    unittest.main()
