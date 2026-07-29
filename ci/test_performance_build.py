from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ci import performance_build


class PerformanceBuildTests(unittest.TestCase):
    def test_stable_and_rv32_commands_keep_the_same_feature_boundary(self) -> None:
        stable = performance_build.cargo_command(engine="jit", target="")
        self.assertEqual(stable[:2], ["cargo", "build"])
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

    def test_source_paths_are_remapped_without_dropping_encoded_flags(self) -> None:
        source = Path("checkout").resolve()
        env = performance_build.remapped_environment(
            source,
            "",
            {"CARGO_ENCODED_RUSTFLAGS": "-C\x1ftarget-cpu=native"},
        )
        self.assertEqual(
            env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")[-1],
            f"--remap-path-prefix={source}={performance_build.VIRTUAL_SOURCE_ROOT}",
        )
        self.assertEqual(
            env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")[:-1],
            ["-C", "target-cpu=native"],
        )

    def test_target_config_rustflags_survive_path_remapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            config_dir = source / ".cargo"
            config_dir.mkdir()
            (config_dir / "config.toml").write_text(
                "[target.riscv64gc-unknown-linux-musl]\n"
                'rustflags = ["-C", "target-feature=+crt-static", '
                '"-C", "link-self-contained=yes", "-C", "panic=abort"]\n',
                encoding="utf-8",
            )
            env = performance_build.remapped_environment(
                source,
                "riscv64gc-unknown-linux-musl",
                {},
            )

        flags = env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        self.assertEqual(
            flags[:-1],
            [
                "-C",
                "target-feature=+crt-static",
                "-C",
                "link-self-contained=yes",
                "-C",
                "panic=abort",
            ],
        )
        self.assertEqual(
            flags[-1],
            f"--remap-path-prefix={source.resolve()}={performance_build.VIRTUAL_SOURCE_ROOT}",
        )

    def test_whitespace_rustflags_are_rejected_instead_of_guessed(self) -> None:
        with self.assertRaisesRegex(ValueError, "CARGO_ENCODED_RUSTFLAGS"):
            performance_build.remapped_environment(
                Path("checkout"),
                "",
                {"RUSTFLAGS": "-C target-cpu=native"},
            )

    def test_baseline_and_candidate_use_isolated_target_directories(
        self,
    ) -> None:
        source = Path("checkout").resolve()
        shared = str(Path("perf-target").resolve())
        baseline = performance_build.isolated_build_environment(
            source,
            label="baseline",
            target="",
            environ={"CARGO_TARGET_DIR": shared},
        )
        candidate = performance_build.isolated_build_environment(
            source,
            label="candidate",
            target="",
            environ={"CARGO_TARGET_DIR": shared},
        )

        self.assertEqual(
            Path(baseline["CARGO_TARGET_DIR"]),
            Path(shared) / "baseline",
        )
        self.assertEqual(
            Path(candidate["CARGO_TARGET_DIR"]),
            Path(shared) / "candidate",
        )
        self.assertNotEqual(
            baseline["CARGO_TARGET_DIR"],
            candidate["CARGO_TARGET_DIR"],
        )

    def test_sha256_file_records_the_copied_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "cli"
            binary.write_bytes(b"same runtime")
            self.assertEqual(
                performance_build.sha256_file(binary),
                hashlib.sha256(b"same runtime").hexdigest(),
            )

    def test_main_writes_versioned_build_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_path = Path(directory)
            out_dir = temp_path / "out"
            out_dir.mkdir()
            with patch.object(
                performance_build,
                "build_one",
                side_effect=[
                    {
                        "sha256": "a" * 64,
                        "size": 123,
                        "virtual_source_root": "/workspace",
                    },
                    {
                        "sha256": "b" * 64,
                        "size": 456,
                        "virtual_source_root": "/workspace",
                    },
                ],
            ):
                exit_code = performance_build.main([
                    "--baseline-source",
                    str(temp_path / "baseline"),
                    "--candidate-source",
                    str(temp_path / "candidate"),
                    "--out-dir",
                    str(out_dir),
                    "--engine",
                    "jit",
                    "--platform",
                    "x64-linux",
                    "--baseline-sha",
                    "base",
                    "--candidate-sha",
                    "head",
                ])

            metadata = json.loads(
                (out_dir / "build-metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["builds"]["baseline"]["revision"], "base")
        self.assertEqual(metadata["builds"]["candidate"]["revision"], "head")
        self.assertFalse(metadata["identical_binaries"])


if __name__ == "__main__":
    unittest.main()
