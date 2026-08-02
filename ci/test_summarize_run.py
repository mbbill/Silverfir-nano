import json
import tempfile
import unittest
from pathlib import Path

from ci.summarize_run import render_report


def write_doc(root: Path, artifact: str, relative: str, doc: dict) -> None:
    path = root / artifact / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def native_doc(statuses: dict[str, str]) -> dict:
    return {
        "tests": {
            name: {
                "metrics": {
                    name: {"status": status, "delta_percent": -5.0}
                }
            }
            for name, status in statuses.items()
        }
    }


def wasmi_doc(rows: dict[str, dict]) -> dict:
    return {"metrics": rows}


class RenderReportTests(unittest.TestCase):
    def test_confirm_artifact_annotates_and_is_not_a_job_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_doc(
                root,
                "performance-x64-linux-jit",
                "perf-output/jit/comparison.json",
                native_doc({"c-ray": "REGRESSION", "coremark": "PASS"}),
            )
            write_doc(
                root,
                "performance-confirm-x64-linux-jit",
                "comparison.json",
                native_doc({"c-ray": "PASS"}),
            )
            report = render_report(root)
        self.assertIn(
            "**c-ray** -5.00% — performance-x64-linux-jit "
            "— not reproduced on an independent runner",
            report,
        )
        self.assertNotIn("| performance-confirm-x64-linux-jit |", report)

    def test_confirmed_and_unconfirmed_regressions_are_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_doc(
                root,
                "wasmi-performance-x64-linux-jit-execute",
                "comparison.json",
                wasmi_doc({
                    "execute/nbody": {
                        "status": "REGRESSION",
                        "delta_percent": -8.9,
                    },
                }),
            )
            write_doc(
                root,
                "wasmi-performance-confirm-x64-linux-jit-execute",
                "comparison.json",
                wasmi_doc({
                    "execute/nbody": {
                        "status": "REGRESSION",
                        "delta_percent": -7.1,
                    },
                }),
            )
            write_doc(
                root,
                "wasmi-performance-arm64-linux-interp-startup",
                "comparison.json",
                wasmi_doc({
                    "startup/bz2": {
                        "status": "REGRESSION",
                        "delta_percent": -4.0,
                    },
                }),
            )
            report = render_report(root)
        self.assertIn(
            "**execute/nbody** -8.90% — "
            "wasmi-performance-x64-linux-jit-execute "
            "— confirmed on an independent runner",
            report,
        )
        self.assertIn(
            "**startup/bz2** -4.00% — "
            "wasmi-performance-arm64-linux-interp-startup "
            "— not re-measured (confirmation did not run)",
            report,
        )

    def test_cannot_run_rows_get_their_own_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_doc(
                root,
                "wasmi-performance-x64-linux-jit-startup",
                "comparison.json",
                wasmi_doc({
                    "startup/spidermonkey": {
                        "status": "CANNOT-RUN",
                        "failed_version": "baseline",
                        "error": "command failed with exit code 101",
                    },
                    "startup/bz2": {
                        "status": "NEGLIGIBLE",
                        "delta_percent": -0.2,
                    },
                }),
            )
            report = render_report(root)
        self.assertIn("### Rows that could not be measured", report)
        self.assertIn(
            "**startup/spidermonkey** — baseline process crashed "
            "— wasmi-performance-x64-linux-jit-startup",
            report,
        )
        self.assertIn("| 1 | 0 | 0 | 1 | 0 | 0 | 0 |", report)

    def test_failure_documents_list_jobs_with_no_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_doc(
                root,
                "wasmi-performance-x64-linux-jit-startup",
                "failure.json",
                {
                    "status": "ERROR",
                    "error": "wasmi-benchmarks revision mismatch\ndetail",
                },
            )
            report = render_report(root)
        self.assertIn("### Jobs with no verdict", report)
        self.assertIn(
            "**wasmi-performance-x64-linux-jit-startup** "
            "— wasmi-benchmarks revision mismatch",
            report,
        )

    def test_empty_artifacts_dir_reports_nothing_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = render_report(Path(temp_dir))
        self.assertIn("No result artifacts found.", report)


if __name__ == "__main__":
    unittest.main()
