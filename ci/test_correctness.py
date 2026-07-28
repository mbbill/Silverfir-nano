from __future__ import annotations

import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ci import correctness
from ci.runner import ROOT, Diagnostic, Result, Runner, parse_log


class WarningGateTests(unittest.TestCase):
    def test_compiler_warning_summary_is_not_double_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cargo.log"
            log.write_text(
                "warning: field `value` is never read\n"
                "  --> src/lib.rs:4:5\n"
                "warning: function `helper` is never used\n"
                "  --> src/lib.rs:8:4\n"
                "warning: `fixture` (lib) generated 2 warnings\n",
                encoding="utf-8",
            )
            errors, warnings, diagnostics = parse_log(log, 0)

        self.assertEqual(errors, 0)
        self.assertEqual(warnings, 2)
        self.assertEqual(len(diagnostics), 3)

    def test_warning_fails_the_final_gate(self) -> None:
        runner = Runner("unit/warning")
        runner.results.append(
            Result(
                name="cargo check",
                status="WARN",
                argv=("cargo", "check"),
                log=runner.log_dir / "cargo-check.log",
                warnings=1,
                diagnostics=(Diagnostic("warning", "warning: dead code", "src/lib.rs:1:1"),),
            )
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.finish(), 1)


class CoveragePlanTests(unittest.TestCase):
    def test_feature_matrices_cover_shipping_engine_boundaries(self) -> None:
        self.assertEqual(
            correctness.ENGINE_FEATURE_CONFIGS,
            (
                ("jit-only", "jit"),
                ("interp-only", "interp"),
                ("dual-engine", "jit,interp"),
            ),
        )

    @mock.patch.object(correctness, "cargo")
    def test_feature_coverage_uses_six_engine_boundaries_and_one_diagnostic_smoke(
        self,
        cargo: mock.Mock,
    ) -> None:
        correctness.run_host_feature_matrix(mock.Mock())

        self.assertEqual(cargo.call_count, 7)
        engine_calls = cargo.call_args_list[:6]
        self.assertEqual(
            [
                (call.kwargs["package"], call.kwargs["features"])
                for call in engine_calls
            ],
            [
                ("sf-nano-core", "jit"),
                ("sf-nano-core", "interp"),
                ("sf-nano-core", "jit,interp"),
                ("sf-nano-cli", "jit"),
                ("sf-nano-cli", "interp"),
                ("sf-nano-cli", "jit,interp"),
            ],
        )
        diagnostic = cargo.call_args_list[-1]
        self.assertIsNone(diagnostic.kwargs.get("package"))
        self.assertEqual(diagnostic.kwargs["features"], "all")
        self.assertEqual(diagnostic.kwargs["extra"], ("--workspace",))

    def test_cross_platforms_are_one_target_per_job(self) -> None:
        self.assertEqual(
            {config.target for config in correctness.CROSS_PLATFORMS.values()},
            {
                "armv7-unknown-linux-musleabihf",
                "riscv64gc-unknown-linux-musl",
                "riscv32gc-unknown-linux-musl",
            },
        )
        self.assertEqual(correctness.CROSS_PLATFORMS["armv7"].qemu, "qemu-arm-static")
        self.assertEqual(correctness.CROSS_PLATFORMS["riscv64"].qemu, "qemu-riscv64-static")
        self.assertEqual(correctness.CROSS_PLATFORMS["riscv32"].qemu, "qemu-riscv32-static")

    def test_interpreter_wast_runtime_keeps_pure_interp_compile_coverage(self) -> None:
        self.assertEqual(correctness.INTERP.features, "interp")
        self.assertEqual(correctness.spectest_features(correctness.INTERP), "jit,interp")
        self.assertEqual(correctness.spectest_features(correctness.JIT), "jit")

    def test_bare_targets_are_explicitly_separate_from_qemu_user(self) -> None:
        self.assertEqual(
            correctness.BARE_PLATFORMS,
            {
                "thumbv8m": "thumbv8m.main-none-eabihf",
                "riscv32": "riscv32imac-unknown-none-elf",
            },
        )

    def test_required_cli_shape(self) -> None:
        self.assertEqual(
            correctness.parse_args(["host", "x64-linux"]).platform,
            "x64-linux",
        )
        self.assertEqual(
            correctness.parse_args(["cross", "riscv32"]).suite,
            "cross",
        )
        self.assertEqual(
            correctness.parse_args(["bare", "thumbv8m"]).suite,
            "bare",
        )

    def test_workflow_has_exactly_one_job_for_every_logical_platform(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "correctness.yml").read_text(
            encoding="utf-8"
        )
        actual = {
            match.group(1).strip()
            for match in re.finditer(r"^\s+suite:\s+(.+)$", workflow, re.MULTILINE)
        }
        expected = {
            *(f"host {label}" for label in correctness.HOST_EXPECTATIONS),
            *(f"cross {label}" for label in correctness.CROSS_PLATFORMS),
            *(f"bare {label}" for label in correctness.BARE_PLATFORMS),
        }
        self.assertEqual(actual, expected)
        self.assertNotIn("scripts/check.py", workflow)
        self.assertIn("unittest discover -s ci -p 'test_*.py'", workflow)
        self.assertIn("if: ${{ always() && !cancelled() }}", workflow)
        for platform in ("armv7-linux", "riscv64-linux", "riscv32-linux"):
            self.assertIn(f"platform: cross-{platform}", workflow)

        performance_workflow = (
            ROOT / ".github" / "workflows" / "performance-regression.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("python3 -m ci.performance_build", performance_workflow)
        self.assertNotIn("build_one()", performance_workflow)
        self.assertIn("unittest discover -s ci -p 'test_*.py'", performance_workflow)
        for platform in ("armv7-a", "riscv64", "riscv32"):
            self.assertIn(f'"platform": "cross-{platform}"', performance_workflow)
        self.assertIn("id: lint-policy", performance_workflow)
        self.assertIn("continue-on-error: true", performance_workflow)
        self.assertIn(
            "if: steps.lint-policy.outcome == 'failure'",
            performance_workflow,
        )

    @mock.patch.object(correctness, "cargo")
    @mock.patch.object(correctness, "require_tools", return_value=True)
    @mock.patch.object(correctness.platform, "system", return_value="Linux")
    @mock.patch.object(correctness.platform, "machine", return_value="x86_64")
    def test_bare_job_builds_objects_for_all_engine_configs(
        self,
        _machine: mock.Mock,
        _system: mock.Mock,
        _tools: mock.Mock,
        cargo: mock.Mock,
    ) -> None:
        cargo.return_value = Result(
            name="build",
            status="OK",
            argv=("cargo", "build"),
            log=None,
        )
        with mock.patch.object(Runner, "finish", return_value=0):
            self.assertEqual(correctness.run_bare("thumbv8m"), 0)

        features = [
            call.kwargs.get("features")
            for call in cargo.call_args_list
            if call.kwargs.get("package") == "sf-nano-core"
        ]
        self.assertEqual(features, ["jit", "interp", "jit,interp"])
        self.assertTrue(all(call.args[2] == "build" for call in cargo.call_args_list))


if __name__ == "__main__":
    unittest.main()
