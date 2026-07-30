from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from ci import wasmi_performance


def criterion_message(
    *,
    benchmark_id: str,
    values: list[float] | None = None,
) -> str:
    measured = values or [float(index * 100) for index in range(1, 11)]
    return json.dumps({
        "reason": "benchmark-complete",
        "id": benchmark_id,
        "iteration_count": list(range(1, 11)),
        "measured_values": measured,
        "unit": "ns",
        "typical": {
            "estimate": 100.0,
            "lower_bound": 99.0,
            "upper_bound": 101.0,
            "unit": "ns",
        },
    })


def fake_run(value: float, version: str) -> dict:
    return {
        "version": version,
        "elapsed_seconds": 0.01,
        "benchmark_id": "synthetic",
        "iteration_count": [1.0] * 10,
        "measured_values": [value] * 10,
        "normalized_ns": [value] * 10,
        "unit": "ns",
        "typical": None,
    }


class WasmiPerformanceTests(unittest.TestCase):
    def test_manifest_has_all_criterion_groups_without_score_runner(
        self,
    ) -> None:
        self.assertEqual(len(wasmi_performance.BENCHMARK_GROUPS), 27)
        self.assertEqual(
            sum(
                group.startswith("execute/")
                for group in wasmi_performance.BENCHMARK_GROUPS
            ),
            20,
        )
        self.assertEqual(
            sum(
                group.startswith("startup/")
                for group in wasmi_performance.BENCHMARK_GROUPS
            ),
            7,
        )
        # startup/coremark measures compilation and instantiation. The
        # dedicated CoreMark score runner is deliberately not represented.
        self.assertIn(
            "startup/coremark", wasmi_performance.BENCHMARK_GROUPS
        )

    def test_patch_config_overrides_only_sf_nano_core(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            core = source / "sf-nano-core"
            core.mkdir(parents=True)
            (core / "Cargo.toml").write_text(
                "[package]\nname = \"sf-nano-core\"\n",
                encoding="utf-8",
            )
            config = root / "baseline.toml"

            wasmi_performance.write_patch_config(config, source)
            text = config.read_text(encoding="utf-8")

        self.assertIn(
            '[patch."https://github.com/mbbill/Silverfir-nano"]',
            text,
        )
        self.assertIn("sf-nano-core = { path =", text)
        self.assertNotIn("wasmi-v", text)

    def test_criterion_command_selects_exactly_one_nano_tier(self) -> None:
        context = wasmi_performance.CargoContext(
            version="candidate",
            source=Path("/source"),
            suite=Path("/suite"),
            config=Path("/suite/.cargo/config.toml"),
            target=Path("/target"),
            cargo="cargo",
            toolchain="1.97.0",
            feature="silverfir-nano-interp",
            runtime_id="silverfir-nano.interpreter",
            core_features="interp",
        )
        command = wasmi_performance.criterion_command(
            context,
            "--message-format=json",
            "execute/counter-local",
        )

        self.assertIn("--no-default-features", command)
        self.assertEqual(
            command[command.index("--features") + 1],
            "silverfir-nano-interp",
        )
        self.assertNotIn("silverfir-nano-jit", command)
        self.assertEqual(
            command[command.index("--plotting-backend") + 1],
            "disabled",
        )
        environment = wasmi_performance.cargo_environment(context)
        self.assertEqual(
            Path(environment["CRITERION_HOME"]).name,
            "criterion-home",
        )

    def test_core_identity_command_builds_only_the_measured_tier(
        self,
    ) -> None:
        context = wasmi_performance.CargoContext(
            version="candidate",
            source=Path("/source"),
            suite=Path("/suite"),
            config=Path("/suite/.cargo/config.toml"),
            target=Path("/target"),
            cargo="cargo",
            toolchain="1.97.0",
            feature="silverfir-nano-jit",
            runtime_id="silverfir-nano.jit",
            core_features="jit,guard-pages",
        )
        command = wasmi_performance.core_identity_command(context)

        self.assertIn("--no-default-features", command)
        self.assertEqual(
            command[command.index("--features") + 1],
            "jit,guard-pages",
        )
        self.assertEqual(
            command[command.index("--package") + 1],
            "sf-nano-core",
        )
        # The suite measures bench-profile code, so the identity build has to
        # use the same codegen settings.
        self.assertEqual(command[command.index("--profile") + 1], "bench")
        self.assertNotIn("interp", command[command.index("--features") + 1])

    def test_engine_core_features_cover_both_tiers_disjointly(self) -> None:
        self.assertEqual(
            set(wasmi_performance.ENGINE_CORE_FEATURES),
            set(wasmi_performance.ENGINE_FEATURE),
        )
        jit = set(wasmi_performance.ENGINE_CORE_FEATURES["jit"].split(","))
        interp = set(
            wasmi_performance.ENGINE_CORE_FEATURES["interp"].split(",")
        )
        self.assertFalse(jit & interp)

    def test_runtime_fingerprint_remaps_the_identity_target_directory(
        self,
    ) -> None:
        # build.rs writes interp_engine_cfg.rs into OUT_DIR and the crate
        # include!s it, so without this remap an interpreter build of identical
        # sources fingerprints differently per version and every benchmark
        # needlessly enters confirmation.
        context = wasmi_performance.CargoContext(
            version="candidate",
            source=Path("/source"),
            suite=Path("/suite"),
            config=Path("/suite/.cargo/config.toml"),
            target=Path("/target"),
            cargo="cargo",
            toolchain="1.97.0",
            feature="silverfir-nano-interp",
            runtime_id="silverfir-nano.interpreter",
            core_features="interp",
        )
        captured: dict[str, str] = {}

        def record(command, *, cwd, env, capture, check=True):
            captured.update(env)
            return subprocess.CompletedProcess(
                args=list(command),
                returncode=0,
                stdout=json.dumps({
                    "reason": "compiler-artifact",
                    "package_id": "path+file:///source#sf-nano-core@0.1.0",
                    "target": {"kind": ["lib"], "name": "sf_nano_core"},
                    "filenames": ["/artifact/libsf_nano_core-1.rlib"],
                }),
                stderr="",
            )

        with (
            patch.object(Path, "mkdir"),
            patch.object(wasmi_performance, "run_process", side_effect=record),
            patch("builtins.open", mock_open(read_data=b"engine")),
        ):
            wasmi_performance.runtime_fingerprint(context)

        flags = captured["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        identity_target = Path("/target/runtime-identity").resolve().as_posix()
        self.assertIn(
            f"--remap-path-prefix={identity_target}=/workspace/identity-target",
            flags,
        )
        # The measured checkout stays remapped as well.
        self.assertTrue(
            any(flag.endswith("=/workspace/sf-nano") for flag in flags)
        )

    def test_runtime_fingerprint_rejects_an_ambiguous_artifact_set(
        self,
    ) -> None:
        context = wasmi_performance.CargoContext(
            version="candidate",
            source=Path("/source"),
            suite=Path("/suite"),
            config=Path("/suite/.cargo/config.toml"),
            target=Path("/target"),
            cargo="cargo",
            toolchain="1.97.0",
            feature="silverfir-nano-jit",
            runtime_id="silverfir-nano.jit",
            core_features="jit,guard-pages",
        )
        # Two rlibs mean the filter stopped identifying one compilation unit;
        # hashing an arbitrary one would silently weaken the drift guard.
        emitted = json.dumps({
            "reason": "compiler-artifact",
            "package_id": "path+file:///source/sf-nano-core#sf-nano-core@0.1.0",
            "target": {"kind": ["lib"], "name": "sf_nano_core"},
            "filenames": ["/a/libsf_nano_core-1.rlib", "/a/libsf_nano_core-2.rlib"],
        })
        with (
            patch.object(Path, "mkdir"),
            patch.object(
                wasmi_performance,
                "run_process",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=emitted, stderr=""
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                wasmi_performance.runtime_fingerprint(context)

    def test_reachable_packages_ignore_disabled_runtime_adapters(
        self,
    ) -> None:
        packages = [
            {"id": "root", "name": "wasmi-benchmarks"},
            {"id": "nano", "name": "rt-silverfir-nano"},
            {"id": "wasmi", "name": "rt-wasmi-v2"},
        ]
        metadata = {
            "packages": packages,
            "resolve": {
                "root": "root",
                "nodes": [
                    {"id": "root", "deps": [{"pkg": "nano"}]},
                    {"id": "nano", "deps": []},
                    {"id": "wasmi", "deps": []},
                ],
            },
        }

        resolved = wasmi_performance.reachable_packages(
            metadata, label="candidate"
        )

        self.assertEqual(
            {package["name"] for package in resolved},
            {"wasmi-benchmarks", "rt-silverfir-nano"},
        )

    def test_runtime_dependency_closure_excludes_suite_parents(
        self,
    ) -> None:
        packages = [
            {"id": "suite", "name": "wasmi-benchmarks"},
            {"id": "adapter", "name": "rt-silverfir-nano"},
            {"id": "nano", "name": "sf-nano-core"},
            {"id": "alloc", "name": "tracked-alloc"},
            {"id": "utils", "name": "benchmark-utils"},
        ]
        metadata = {
            "packages": packages,
            "resolve": {
                "root": "suite",
                "nodes": [
                    {
                        "id": "suite",
                        "deps": [
                            {"pkg": "adapter"},
                            {"pkg": "utils"},
                        ],
                    },
                    {"id": "adapter", "deps": [{"pkg": "nano"}]},
                    {"id": "nano", "deps": [{"pkg": "alloc"}]},
                    {"id": "alloc", "deps": []},
                    {"id": "utils", "deps": []},
                ],
            },
        }

        resolved = wasmi_performance.dependency_closure_packages(
            metadata,
            "nano",
            label="candidate sf-nano-core",
        )

        self.assertEqual(
            {package["name"] for package in resolved},
            {"sf-nano-core", "tracked-alloc"},
        )

    def test_failed_process_can_be_recorded_before_raising(self) -> None:
        result = subprocess.CompletedProcess(
            ["cargo", "criterion"],
            returncode=3,
            stdout="partial result",
            stderr="criterion failed",
        )
        with self.assertRaisesRegex(RuntimeError, "criterion failed"):
            wasmi_performance.raise_for_process_failure(
                result, result.args
            )

    def test_cargo_criterion_version_is_exactly_pinned(self) -> None:
        wrong = subprocess.CompletedProcess(
            ["cargo", "criterion", "--version"],
            returncode=0,
            stdout="cargo-criterion 1.2.0\n",
            stderr="",
        )
        with (
            patch.object(
                wasmi_performance,
                "run_process",
                return_value=wrong,
            ),
            self.assertRaisesRegex(ValueError, "expected"),
        ):
            wasmi_performance.verify_cargo_criterion("cargo", "1.97.0")

    def test_parse_criterion_json_normalizes_samples(self) -> None:
        output = criterion_message(
            benchmark_id=(
                "execute/counter-local/"
                "silverfir-nano.jit/1000000"
            )
        )
        parsed = wasmi_performance.parse_criterion_json(
            output,
            group="execute/counter-local",
            runtime_id="silverfir-nano.jit",
        )
        self.assertEqual(parsed["normalized_ns"], [100.0] * 10)
        self.assertEqual(len(parsed["iteration_count"]), 10)

    def test_parse_criterion_json_rejects_wrong_engine(self) -> None:
        output = criterion_message(
            benchmark_id=(
                "startup/ffmpeg/silverfir-nano.interpreter"
            )
        )
        with self.assertRaisesRegex(ValueError, "unexpected benchmark id"):
            wasmi_performance.parse_criterion_json(
                output,
                group="startup/ffmpeg",
                runtime_id="silverfir-nano.jit",
            )

    def test_parse_criterion_json_rejects_missing_result(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "expected one benchmark-complete"
        ):
            wasmi_performance.parse_criterion_json(
                json.dumps({"reason": "group-complete"}),
                group="startup/ffmpeg",
                runtime_id="silverfir-nano.jit",
            )

    def test_summarize_pairs_uses_lower_is_better_direction(self) -> None:
        metric = wasmi_performance.summarize_pairs([{
            "order": ["baseline", "candidate"],
            "baseline": fake_run(100.0, "baseline"),
            "candidate": fake_run(102.0, "candidate"),
        }])
        self.assertLess(metric["delta_percent"], 0.0)
        self.assertEqual(metric["probability_regression"], 1.0)
        self.assertEqual(metric["pair_count"], 10)

    def test_main_freezes_candidates_and_reverses_confirmation(
        self,
    ) -> None:
        calls: list[tuple[str, str, bool]] = []
        regression = "execute/counter-local"

        def measure_pair(
            _contexts: object,
            **kwargs: object,
        ) -> dict:
            group = str(kwargs["group"])
            phase = str(kwargs["phase"])
            baseline_first = bool(kwargs["baseline_first"])
            calls.append((group, phase, baseline_first))
            candidate_value = 102.0 if group == regression else 100.0
            return {
                "order": (
                    ["baseline", "candidate"]
                    if baseline_first
                    else ["candidate", "baseline"]
                ),
                "baseline": fake_run(100.0, "baseline"),
                "candidate": fake_run(candidate_value, "candidate"),
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = root / "wasmi"
            suite.mkdir()
            (suite / "Cargo.toml").write_text(
                "[package]\nname = \"wasmi-benchmarks\"\n",
                encoding="utf-8",
            )
            (suite / "Cargo.lock").write_text(
                "# pinned fixture\n",
                encoding="utf-8",
            )
            for name in ("baseline", "candidate"):
                core = root / name / "sf-nano-core"
                core.mkdir(parents=True)
                (core / "Cargo.toml").write_text(
                    "[package]\nname = \"sf-nano-core\"\n",
                    encoding="utf-8",
                )
            out_dir = root / "out"
            argv = [
                "wasmi_performance.py",
                "--suite",
                str(suite),
                "--baseline-source",
                str(root / "baseline"),
                "--candidate-source",
                str(root / "candidate"),
                "--baseline-sha",
                "base0000",
                "--candidate-sha",
                "head0000",
                "--platform",
                "x64-linux",
                "--engine",
                "jit",
                "--out-dir",
                str(out_dir),
                "--target-root",
                str(root / "target"),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    wasmi_performance,
                    "git_revision",
                    side_effect=[
                        wasmi_performance.WASMI_BENCHMARKS_REVISION,
                        "base000000000000000000000000000000000000",
                        "head000000000000000000000000000000000000",
                    ],
                ),
                patch.object(
                    wasmi_performance,
                    "verify_cargo_criterion",
                    return_value="cargo-criterion 1.1.0",
                ),
                patch.object(
                    wasmi_performance,
                    "build_context",
                    side_effect=[
                        {"runtime_fingerprint": "baseline"},
                        {"runtime_fingerprint": "candidate"},
                    ],
                ),
                patch.object(
                    wasmi_performance,
                    "measure_pair",
                    side_effect=measure_pair,
                ),
            ):
                exit_code = wasmi_performance.main()

            document = json.loads(
                (out_dir / "comparison.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 1)
        pilot = [call for call in calls if call[1] == "pilot"]
        confirmation = [
            call for call in calls if call[1] == "confirmation"
        ]
        self.assertEqual(len(pilot), 27)
        self.assertEqual(confirmation, [(regression, "confirmation", False)])
        self.assertEqual(
            document["metrics"][regression]["status"], "REGRESSION"
        )
        self.assertEqual(
            {
                group
                for group, metric in document["metrics"].items()
                if metric["selected"]
            },
            {regression},
        )

    def test_main_skips_confirmation_for_identical_compiled_runtime(
        self,
    ) -> None:
        calls: list[tuple[str, str]] = []

        def measure_pair(
            _contexts: object,
            **kwargs: object,
        ) -> dict:
            group = str(kwargs["group"])
            phase = str(kwargs["phase"])
            calls.append((group, phase))
            return {
                "order": ["baseline", "candidate"],
                "baseline": fake_run(100.0, "baseline"),
                "candidate": fake_run(102.0, "candidate"),
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = root / "wasmi"
            suite.mkdir()
            (suite / "Cargo.toml").write_text(
                "[package]\nname = \"wasmi-benchmarks\"\n",
                encoding="utf-8",
            )
            (suite / "Cargo.lock").write_text(
                "# pinned fixture\n",
                encoding="utf-8",
            )
            for name in ("baseline", "candidate"):
                core = root / name / "sf-nano-core"
                core.mkdir(parents=True)
                (core / "Cargo.toml").write_text(
                    "[package]\nname = \"sf-nano-core\"\n",
                    encoding="utf-8",
                )
            out_dir = root / "out"
            argv = [
                "wasmi_performance.py",
                "--suite",
                str(suite),
                "--baseline-source",
                str(root / "baseline"),
                "--candidate-source",
                str(root / "candidate"),
                "--baseline-sha",
                "base0000",
                "--candidate-sha",
                "head0000",
                "--platform",
                "arm64-linux",
                "--engine",
                "interp",
                "--category",
                "startup",
                "--family-metric-count",
                "27",
                "--out-dir",
                str(out_dir),
                "--target-root",
                str(root / "target"),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    wasmi_performance,
                    "git_revision",
                    side_effect=[
                        wasmi_performance.WASMI_BENCHMARKS_REVISION,
                        "base000000000000000000000000000000000000",
                        "head000000000000000000000000000000000000",
                    ],
                ),
                patch.object(
                    wasmi_performance,
                    "verify_cargo_criterion",
                    return_value="cargo-criterion 1.1.0",
                ),
                patch.object(
                    wasmi_performance,
                    "build_context",
                    side_effect=[
                        {"runtime_fingerprint": "same"},
                        {"runtime_fingerprint": "same"},
                    ],
                ),
                patch.object(
                    wasmi_performance,
                    "measure_pair",
                    side_effect=measure_pair,
                ),
            ):
                exit_code = wasmi_performance.main()

            document = json.loads(
                (out_dir / "comparison.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        startup_groups = {
            group
            for group in wasmi_performance.BENCHMARK_GROUPS
            if group.startswith("startup/")
        }
        self.assertEqual(len(calls), 7)
        self.assertEqual({group for group, _ in calls}, startup_groups)
        self.assertEqual({phase for _, phase in calls}, {"pilot"})
        self.assertTrue(document["identical_runtime"])
        self.assertEqual(document["category"], "startup")
        self.assertEqual(document["shard_metric_count"], 7)
        self.assertEqual(document["family_metric_count"], 27)
        self.assertEqual(
            set(document["confirmation_pairs"]),
            startup_groups,
        )
        self.assertTrue(
            all(not pairs for pairs in document["confirmation_pairs"].values())
        )
        self.assertEqual(
            {
                metric["status"]
                for metric in document["metrics"].values()
            },
            {"UNSTABLE"},
        )
        self.assertTrue(
            all(
                metric["process_pairs"] == 1
                for metric in document["metrics"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
