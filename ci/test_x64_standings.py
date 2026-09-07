"""Keep missing or invalid measurements from improving cross-engine rankings."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ci import x64_standings_report as corpus
from ci import x64_wasi_standings as wasi


class StandingsTests(unittest.TestCase):
    def test_requires_complete_common_corpus(self):
        row = {"silverfir-nano.jit": 100.0, "wasmtime.cranelift": 90.0, "v8": 80.0}
        corpus.validate_field({"sort": row}, 1)
        with self.assertRaisesRegex(ValueError, "expected 20"):
            corpus.validate_field({"sort": row}, 20)
        with self.assertRaisesRegex(ValueError, "incomplete runtime"):
            corpus.validate_field({"sort": {"silverfir-nano.jit": 100.0}}, 1)
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "invalid timing"):
                corpus.validate_field({"sort": {**row, "v8": invalid}}, 1)

    def test_numeric_input_is_part_of_case_and_old_results_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for phase, value in (("new", 100.0), ("base", 999.0)):
                path = root / "execute_counter-local" / "silverfir-nano.jit_1000000" / phase / "estimates.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"mean": {"point_estimate": value}}))
            self.assertEqual(corpus.collect_cases(root), {
                "counter-local/1000000": {"silverfir-nano.jit": 100.0}
            })

    def test_wasi_process_failure_cannot_pass_with_partial_output(self):
        output = "----\nlua/fib PASS 100 fib20/s\n"
        with patch.object(wasi.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            run.return_value.returncode = 1
            run.return_value.stdout = output
            run.return_value.stderr = "engine failed"
            rates, failures = wasi.run_v8("node", 2.0)
        self.assertEqual(rates["lua/fib"][0], 100.0)
        self.assertTrue(any("exited 1" in failure for failure in failures))

    def test_partial_wasi_run_writes_diagnostic_but_fails(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(wasi, "load_run_tests"), patch.object(wasi, "run_native") as native, patch.object(wasi, "run_v8") as v8:
                native.return_value = ({"lua/fib": (100.0, "fib20/s")}, [])
                v8.return_value = ({"lua/fib": (110.0, "fib20/s")}, [])
                result = wasi.main([
                    "--nano", "nano", "--wasmtime", "wasmtime",
                    "--out-md", f"{temp}/report.md", "--out-json", f"{temp}/report.json",
                ])
            self.assertEqual(result, 1)
            self.assertTrue(Path(temp, "report.md").is_file())
            self.assertEqual(len(json.loads(Path(temp, "report.json").read_text())["failures"]), 3)
