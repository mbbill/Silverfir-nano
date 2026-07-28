from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from scripts.check import CheckRunner, Host, StepResult, parse_log


class WarningGateTests(unittest.TestCase):
    def test_parse_log_counts_every_compiler_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cargo.log"
            log.write_text(
                "warning: field `value` is never read\n"
                "  --> src/lib.rs:4:5\n"
                "warning: function `helper` is never used\n"
                "  --> src/lib.rs:8:4\n"
                "warning: `fixture` (lib) generated 2 warnings\n",
                encoding="utf-8",
            )
            errors, warnings, diagnostics = parse_log(log, 0, 10)

        self.assertEqual(errors, 0)
        self.assertEqual(warnings, 2)
        self.assertEqual(len(diagnostics), 3)

    def test_warning_status_fails_the_correctness_report(self) -> None:
        runner = CheckRunner(
            Host("linux", "x86_64"),
            install_targets=False,
            max_diagnostics=8,
            dry_run=True,
            phase="unit-test",
        )
        runner.results.append(
            StepResult(
                index=1,
                name="cargo check",
                status="WARN",
                command="cargo check",
                log_path=runner.log_dir / "cargo-check.log",
                warnings=1,
            )
        )

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.final_report(), 1)


if __name__ == "__main__":
    unittest.main()
