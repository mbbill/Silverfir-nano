from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ci import performance_build


class PerformanceBuildTests(unittest.TestCase):
    def test_stable_and_rv32_commands_keep_the_same_feature_boundary(self) -> None:
        stable = performance_build.cargo_command(engine="jit", target="")
        self.assertEqual(stable[:3], ["cargo", "+1.97.0", "build"])
        self.assertEqual(stable[-2:], ["--features", "jit"])

        rv32 = performance_build.cargo_command(
            engine="interp",
            target=performance_build.RV32_LINUX,
        )
        self.assertEqual(
            rv32[:5],
            ["cargo", "+nightly", "build", "-Z", "build-std=std,panic_abort"],
        )
        self.assertEqual(rv32[-2:], ["--target", performance_build.RV32_LINUX])
        self.assertIn("interp", rv32)

    def test_source_paths_are_remapped_without_dropping_existing_flags(self) -> None:
        source = Path("checkout").resolve()
        env = performance_build.remapped_environment(
            source,
            {"RUSTFLAGS": "-C target-cpu=native"},
        )
        self.assertEqual(
            env["RUSTFLAGS"],
            (
                "-C target-cpu=native "
                f"--remap-path-prefix={source}={performance_build.VIRTUAL_SOURCE_ROOT}"
            ),
        )

        encoded = performance_build.remapped_environment(
            source,
            {"CARGO_ENCODED_RUSTFLAGS": "-C\x1fopt-level=2"},
        )
        self.assertEqual(
            encoded["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")[-1],
            f"--remap-path-prefix={source}={performance_build.VIRTUAL_SOURCE_ROOT}",
        )

    def test_sha256_file_records_the_copied_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "cli"
            binary.write_bytes(b"same runtime")
            self.assertEqual(
                performance_build.sha256_file(binary),
                hashlib.sha256(b"same runtime").hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
