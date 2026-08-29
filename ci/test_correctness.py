from __future__ import annotations

import contextlib
import io
import re
import subprocess
import sys
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

    def test_build_std_linker_probe_does_not_fail_the_gate(self) -> None:
        """`-Z build-std` on a tier-3 target provokes linker chatter no change
        here can silence. It must stay visible without turning a job whose
        rv32 spectests all pass permanently red."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cargo.log"
            log.write_text(
                "warning: linker stderr: ignoring deprecated linker "
                "optimization setting '1'\n"
                "warning: linker stderr: unable to open library directory "
                "'/home/runner/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu"
                "/lib/rustlib/riscv32gc-unknown-linux-musl/lib': FileNotFound\n"
                "warning: `sf-nano-cli` (bin \"sf-nano-cli\") generated 2 warnings\n",
                encoding="utf-8",
            )
            errors, warnings, diagnostics = parse_log(log, 0)

        self.assertEqual(errors, 0)
        self.assertEqual(warnings, 0, "benign linker probe must not fail the gate")
        self.assertEqual(len(diagnostics), 3, "and must still be visible")

    def test_real_error_riding_a_benign_linker_diagnostic_still_fails(self) -> None:
        """rustc folds one linker invocation's whole stderr into ONE diagnostic:
        the continuation lines are indented and carry no `warning:` prefix. An
        `undefined reference` therefore arrives attached to a benign-looking
        warning, and matching only the first line hid it. Captured from real
        rustc output, not hand-written."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cargo.log"
            log.write_text(
                "warning: linker stderr: ignoring deprecated linker "
                "optimization setting '1'\n"
                "         undefined reference to `sf_os_alloc'\n"
                "  |\n"
                "  = note: `#[warn(linker_messages)]` on by default\n",
                encoding="utf-8",
            )
            errors, warnings, _ = parse_log(log, 0)
        self.assertEqual(
            (errors, warnings, 0)[:2] != (0, 0),
            True,
            "a real linker error must not be discounted with the benign line it rides on",
        )

    def test_benign_prefix_with_trailing_text_still_fails(self) -> None:
        """The patterns are fully anchored, so a benign prefix followed by
        anything else does not match."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cargo.log"
            log.write_text(
                "warning: linker stderr: ignoring deprecated linker optimization "
                "setting '1'; undefined reference to `sf_os_alloc'\n",
                encoding="utf-8",
            )
            errors, warnings, _ = parse_log(log, 0)
        self.assertNotEqual((errors, warnings), (0, 0))

    def test_rustlib_probe_is_scoped_to_the_build_std_target(self) -> None:
        """Only riscv32gc-unknown-linux-musl builds its std from source. Any
        other target missing its rustlib means a broken toolchain."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cargo.log"
            log.write_text(
                "warning: linker stderr: unable to open library directory "
                "'/x/lib/rustlib/x86_64-pc-windows-msvc/lib': FileNotFound\n",
                encoding="utf-8",
            )
            errors, warnings, _ = parse_log(log, 0)
        self.assertNotEqual((errors, warnings), (0, 0))

    def test_other_linker_warnings_still_fail_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cargo.log"
            log.write_text(
                "warning: linker stderr: undefined reference to `sf_os_alloc`\n"
                "warning: `sf-nano-cli` (bin \"sf-nano-cli\") generated 1 warning\n",
                encoding="utf-8",
            )
            errors, warnings, diagnostics = parse_log(log, 0)

        self.assertEqual(errors, 0)
        self.assertEqual(warnings, 1, "only the two named probe messages are benign")
        self.assertEqual(len(diagnostics), 2)

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

    def test_armv7_thumb2_variant_selects_the_core_owned_feature(self) -> None:
        self.assertEqual(
            correctness.CROSS_PLATFORMS["armv7"].jit_variants,
            (
                ("armv7a", ""),
                ("armv7m", "sf-nano-core/thumb2-test"),
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

    @mock.patch.object(correctness, "cargo")
    def test_native_workspace_tests_run_once_but_both_profiles_build(
        self,
        cargo: mock.Mock,
    ) -> None:
        correctness.run_host_builds_and_tests(mock.Mock())

        self.assertEqual(
            [
                (call.args[2], call.kwargs.get("profile", "debug"))
                for call in cargo.call_args_list
            ],
            [
                ("build", "debug"),
                ("test", "debug"),
                ("build", "release"),
            ],
        )

    @mock.patch.object(correctness, "ensure_testsuite", return_value=True)
    @mock.patch.object(correctness, "cargo")
    def test_native_runtime_suites_use_release_artifacts(
        self,
        cargo: mock.Mock,
        _testsuite: mock.Mock,
    ) -> None:
        cargo.return_value = Result(
            name="build",
            status="OK",
            argv=("cargo", "build"),
            log=None,
        )
        runner = mock.Mock()

        correctness.run_native_spectest(runner)
        spectest_calls = list(cargo.call_args_list)
        cargo.reset_mock()
        correctness.run_native_wasitest(runner)
        wasi_calls = list(cargo.call_args_list)

        self.assertEqual(len(spectest_calls), 2)
        self.assertEqual(len(wasi_calls), 3)
        self.assertTrue(
            all(call.kwargs["profile"] == "release" for call in spectest_calls + wasi_calls)
        )

    @mock.patch.object(correctness, "copy_built_binary", return_value=None)
    @mock.patch.object(correctness, "ensure_testsuite", return_value=True)
    @mock.patch.object(correctness, "cargo")
    def test_cross_spectest_does_not_repeat_debug_runtime(
        self,
        cargo: mock.Mock,
        _testsuite: mock.Mock,
        _copy: mock.Mock,
    ) -> None:
        cargo.return_value = Result(
            name="build",
            status="OK",
            argv=("cargo", "build"),
            log=None,
        )
        correctness.run_cross_spectest(
            mock.Mock(),
            correctness.CROSS_PLATFORMS["armv7"],
        )

        self.assertEqual(len(cargo.call_args_list), 3)
        self.assertTrue(
            all(call.kwargs["profile"] == "release" for call in cargo.call_args_list)
        )

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

    @mock.patch.object(correctness, "cargo")
    def test_riscv64_guard_regression_runs_as_a_focused_qemu_libtest(
        self,
        cargo: mock.Mock,
    ) -> None:
        for config in correctness.CROSS_PLATFORMS.values():
            correctness.run_cross_regression_tests(mock.Mock(), config)

        cargo.assert_called_once()
        call = cargo.call_args
        config = correctness.CROSS_PLATFORMS["riscv64"]
        self.assertEqual(call.args[2], "test")
        self.assertEqual(call.kwargs["package"], "sf-nano-core")
        self.assertEqual(call.kwargs["profile"], "release")
        self.assertEqual(call.kwargs["target"], config.target)
        self.assertEqual(call.kwargs["features"], "interp")
        self.assertEqual(
            call.kwargs["extra"],
            (
                "--config",
                f'target.{config.target}.rustflags=["-C","panic=unwind"]',
                "--config",
                f'target.{config.target}.runner=["{config.qemu}","-cpu","{config.cpu}"]',
                "--lib",
                correctness.SPECIAL_FUNCREF_GUARD_TEST,
                "--",
                "--exact",
                "--nocapture",
            ),
        )

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
        for step in (
            "Test CI implementation",
            "Enforce lint suppression policy",
            "Check Rust formatting",
        ):
            self.assertRegex(
                workflow,
                rf"- name: {re.escape(step)}\s+"
                r"if: \$\{\{ always\(\) && !cancelled\(\) \}\}",
            )
        for platform in ("armv7-linux", "riscv64-linux", "riscv32-linux"):
            self.assertIn(f"platform: cross-{platform}", workflow)
        self.assertNotIn("x86_64-pc-windows-gnu", workflow)

        performance_workflow = (
            ROOT / ".github" / "workflows" / "performance-regression.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("python3 -m ci.performance_build", performance_workflow)
        self.assertIn(
            '--build-metadata "perf-bin/build-metadata.json"',
            performance_workflow,
        )
        self.assertIn(
            'if [[ "$MODE" == "correctness" ]]',
            performance_workflow,
        )
        self.assertIn("args+=(--correctness-only)", performance_workflow)
        self.assertNotIn("--exclude stream/stream.wasm", performance_workflow)
        self.assertNotIn(
            '-n "$RUNNER_PREFIX" && "$ENGINE" == "interp"',
            performance_workflow,
        )
        self.assertIn("timeout-minutes: 30", performance_workflow)
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
        benchmark_step = performance_workflow.split(
            "- name: Run benchmark suite",
            maxsplit=1,
        )[1].split(
            "- uses: actions/upload-artifact",
            maxsplit=1,
        )[0]
        workflow_options = set(
            re.findall(r"--[a-z][a-z0-9-]*", benchmark_step)
        )
        performance_help = subprocess.run(
            [sys.executable, "-B", "-m", "ci.performance", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        unsupported_options = {
            option
            for option in workflow_options
            if option not in performance_help
        }
        self.assertEqual(unsupported_options, set())
        self.assertIn(
            "CARGO_TARGET_DIR: ${{ github.workspace }}/perf-target",
            performance_workflow,
        )

    def test_workflow_inventory_and_readme_badges_have_no_legacy_entries(self) -> None:
        workflows = ROOT / ".github" / "workflows"
        self.assertEqual(
            {path.name for path in workflows.glob("*.yml")},
            {
                "correctness.yml",
                "performance-regression.yml",
                "wasmi-startup-ranking.yml",
            },
        )

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("actions/workflows/correctness.yml", readme)
        self.assertIn("actions/workflows/performance-regression.yml", readme)
        self.assertNotIn("actions/workflows/check-", readme)

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
