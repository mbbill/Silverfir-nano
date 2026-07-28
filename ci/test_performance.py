import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ci.performance as bench_compare
from ci.performance import (
    classify_metrics,
    confirmation_candidates,
    fourth_round_candidates,
    fourth_round_improvement_candidates,
    regression_candidates,
    third_round_candidates,
    third_round_improvement_candidates,
)


def metric(delta_percent: float) -> dict:
    return {
        "unit": "units/s",
        "direction": "higher",
        "baseline_samples": [100.0, 100.0],
        "candidate_samples": [100.0, 100.0],
        "baseline_geomean": 100.0,
        "candidate_geomean": 100.0,
        "pair_deltas_percent": [delta_percent, delta_percent],
        "delta_percent": delta_percent,
    }


class StagedGateTests(unittest.TestCase):
    def test_regression_and_improvement_rules(self) -> None:
        initial = {
            "second_recovers": metric(-1.5),
            "third_recovers": metric(-1.2),
            "fourth_recovers": metric(-1.1),
            "persists": metric(-2.0),
            "new_second_regression": metric(-0.5),
            "new_second_improvement": metric(0.5),
            "confirmed_improvement": metric(3.1),
            "faded_improvement": metric(4.0),
            "third_faded_improvement": metric(3.5),
            "fourth_faded_improvement": metric(3.5),
        }
        confirmation = {
            "second_recovers": metric(-0.5),
            "third_recovers": metric(-1.1),
            "fourth_recovers": metric(-1.1),
            "persists": metric(-1.1),
            "confirmed_improvement": metric(3.2),
            "faded_improvement": metric(3.0),
            "third_faded_improvement": metric(3.2),
            "fourth_faded_improvement": metric(3.2),
            # This must be ignored because it passed the initial screen.
            "new_second_regression": metric(-9.0),
            "new_second_improvement": metric(9.0),
        }
        third_round = {
            "third_recovers": metric(-0.9),
            "fourth_recovers": metric(-1.1),
            "persists": metric(-1.01),
            "confirmed_improvement": metric(3.3),
            "third_faded_improvement": metric(3.0),
            "fourth_faded_improvement": metric(3.2),
            # This remains ignored despite sharing a rerun benchmark.
            "new_second_regression": metric(-20.0),
            "new_second_improvement": metric(20.0),
        }
        fourth_round = {
            "fourth_recovers": metric(-0.9),
            "persists": metric(-1.01),
            "confirmed_improvement": metric(3.4),
            "fourth_faded_improvement": metric(3.0),
            "new_second_improvement": metric(20.0),
        }

        self.assertEqual(
            regression_candidates(initial, 1.0),
            {
                "second_recovers",
                "third_recovers",
                "fourth_recovers",
                "persists",
            },
        )
        self.assertEqual(
            confirmation_candidates(initial, 1.0, 3.0),
            {
                "second_recovers",
                "third_recovers",
                "fourth_recovers",
                "persists",
                "confirmed_improvement",
                "faded_improvement",
                "third_faded_improvement",
                "fourth_faded_improvement",
            },
        )
        self.assertEqual(
            third_round_candidates(initial, confirmation, 1.0),
            {"third_recovers", "fourth_recovers", "persists"},
        )
        self.assertEqual(
            fourth_round_candidates(
                initial,
                confirmation,
                third_round,
                1.0,
            ),
            {"fourth_recovers", "persists"},
        )
        self.assertEqual(
            third_round_improvement_candidates(
                initial,
                confirmation,
                3.0,
            ),
            {
                "confirmed_improvement",
                "third_faded_improvement",
                "fourth_faded_improvement",
            },
        )
        self.assertEqual(
            fourth_round_improvement_candidates(
                initial,
                confirmation,
                third_round,
                3.0,
            ),
            {"confirmed_improvement", "fourth_faded_improvement"},
        )

        result = classify_metrics(
            initial=initial,
            confirmation=confirmation,
            third_round=third_round,
            fourth_round=fourth_round,
            regression_threshold=1.0,
            improvement_threshold=3.0,
        )

        self.assertEqual(result["second_recovers"]["status"], "RECOVERED")
        self.assertIsNone(result["second_recovers"]["third_round"])
        self.assertEqual(result["third_recovers"]["status"], "RECOVERED")
        self.assertIsNone(result["third_recovers"]["fourth_round"])
        self.assertEqual(result["fourth_recovers"]["status"], "RECOVERED")
        self.assertEqual(result["persists"]["status"], "REGRESSION")
        self.assertEqual(result["new_second_regression"]["status"], "PASS")
        self.assertIsNone(
            result["new_second_regression"]["confirmation"]
        )
        self.assertIsNone(result["new_second_regression"]["third_round"])
        self.assertIsNone(result["new_second_regression"]["fourth_round"])
        self.assertEqual(result["new_second_improvement"]["status"], "PASS")
        self.assertIsNone(
            result["new_second_improvement"]["confirmation"]
        )
        self.assertIsNone(
            result["new_second_improvement"]["third_round"]
        )
        self.assertIsNone(
            result["new_second_improvement"]["fourth_round"]
        )
        self.assertEqual(
            result["confirmed_improvement"]["status"],
            "IMPROVEMENT",
        )
        self.assertEqual(result["faded_improvement"]["status"], "PASS")
        self.assertEqual(
            result["third_faded_improvement"]["status"],
            "PASS",
        )
        self.assertIsNone(
            result["third_faded_improvement"]["fourth_round"]
        )
        self.assertEqual(
            result["fourth_faded_improvement"]["status"],
            "PASS",
        )
        self.assertIsNotNone(
            result["confirmed_improvement"]["third_round"]
        )
        self.assertIsNotNone(
            result["confirmed_improvement"]["fourth_round"]
        )
        self.assertIsNone(
            result["faded_improvement"]["third_round"]
        )

    def test_threshold_boundaries_are_inclusive_passes(self) -> None:
        initial = {
            "negative_boundary": metric(-1.0),
            "positive_boundary": metric(3.0),
        }
        result = classify_metrics(
            initial=initial,
            confirmation={},
            third_round={},
            fourth_round={},
            regression_threshold=1.0,
            improvement_threshold=3.0,
        )

        self.assertEqual(result["negative_boundary"]["status"], "PASS")
        self.assertEqual(result["positive_boundary"]["status"], "PASS")

    def test_later_round_boundaries_recover(self) -> None:
        result = classify_metrics(
            initial={
                "second_boundary": metric(-1.01),
                "third_boundary": metric(-1.01),
                "fourth_boundary": metric(-1.01),
            },
            confirmation={
                "second_boundary": metric(-1.0),
                "third_boundary": metric(-1.01),
                "fourth_boundary": metric(-1.01),
            },
            third_round={
                "third_boundary": metric(-1.0),
                "fourth_boundary": metric(-1.01),
            },
            fourth_round={"fourth_boundary": metric(-1.0)},
            regression_threshold=1.0,
            improvement_threshold=3.0,
        )

        self.assertEqual(result["second_boundary"]["status"], "RECOVERED")
        self.assertEqual(result["third_boundary"]["status"], "RECOVERED")
        self.assertEqual(result["fourth_boundary"]["status"], "RECOVERED")

    def test_later_improvement_boundaries_pass(self) -> None:
        result = classify_metrics(
            initial={
                "second_boundary": metric(3.01),
                "third_boundary": metric(3.01),
                "fourth_boundary": metric(3.01),
            },
            confirmation={
                "second_boundary": metric(3.0),
                "third_boundary": metric(3.01),
                "fourth_boundary": metric(3.01),
            },
            third_round={
                "third_boundary": metric(3.0),
                "fourth_boundary": metric(3.01),
            },
            fourth_round={"fourth_boundary": metric(3.0)},
            regression_threshold=1.0,
            improvement_threshold=3.0,
        )

        self.assertEqual(result["second_boundary"]["status"], "PASS")
        self.assertEqual(result["third_boundary"]["status"], "PASS")
        self.assertEqual(result["fourth_boundary"]["status"], "PASS")

    def test_selected_metric_requires_second_round(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "selected metric has no second round",
        ):
            classify_metrics(
                initial={"missing": metric(-1.01)},
                confirmation={},
                third_round={},
                fourth_round={},
                regression_threshold=1.0,
                improvement_threshold=3.0,
            )

    def test_persistent_regression_requires_third_round(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "persistent metric has no third round",
        ):
            classify_metrics(
                initial={"missing": metric(-1.01)},
                confirmation={"missing": metric(-1.01)},
                third_round={},
                fourth_round={},
                regression_threshold=1.0,
                improvement_threshold=3.0,
            )

    def test_persistent_regression_requires_fourth_round(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "persistent metric has no fourth round",
        ):
            classify_metrics(
                initial={"missing": metric(-1.01)},
                confirmation={"missing": metric(-1.01)},
                third_round={"missing": metric(-1.01)},
                fourth_round={},
                regression_threshold=1.0,
                improvement_threshold=3.0,
            )

    def test_persistent_improvement_requires_fourth_round(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "persistent improvement has no fourth round",
        ):
            classify_metrics(
                initial={"missing": metric(3.01)},
                confirmation={"missing": metric(3.01)},
                third_round={"missing": metric(3.01)},
                fourth_round={},
                regression_threshold=1.0,
                improvement_threshold=3.0,
            )


class ScriptIntegrationTests(unittest.TestCase):
    def test_main_runs_four_rounds_and_ignores_late_regression(self) -> None:
        tests = [{"name": "synthetic"}]
        extractors = {
            "synthetic": [
                ("score", None, "units/s", "higher"),
                ("late", None, "units/s", "higher"),
            ]
        }

        def fake_run_once(**kwargs: object) -> dict:
            block = int(kwargs["block"])
            slot = int(kwargs["slot"])
            version = str(kwargs["version"])
            baseline = version == "baseline"
            return {
                "block": block,
                "slot": slot,
                "version": version,
                "elapsed_seconds": 0.01,
                "metric_text": "synthetic",
                "metrics": {
                    "score": {
                        "value": 100.0 if baseline else 98.0,
                        "unit": "units/s",
                        "direction": "higher",
                    },
                    "late": {
                        "value": (
                            100.0
                            if baseline or block == 0
                            else 80.0
                        ),
                        "unit": "units/s",
                        "direction": "higher",
                    },
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            argv = [
                "bench_compare.py",
                "--baseline-exec",
                "baseline",
                "--candidate-exec",
                "candidate",
                "--engine",
                "interp",
                "--platform",
                "synthetic",
                "--warmup-time",
                "0",
                "--time",
                "0.01",
                "--out-dir",
                str(out_dir),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bench_compare.run_tests, "TESTS", tests),
                patch.object(bench_compare, "METRIC_EXTRACTORS", extractors),
                patch.object(
                    bench_compare,
                    "command_for",
                    return_value=("fake", []),
                ),
                patch.object(
                    bench_compare,
                    "run_once",
                    side_effect=fake_run_once,
                ),
            ):
                exit_code = bench_compare.main()

            document = json.loads(
                (out_dir / "comparison.json").read_text(encoding="utf-8")
            )
            result = document["tests"]["synthetic"]

        self.assertEqual(exit_code, 1)
        self.assertEqual(document["schema_version"], 4)
        self.assertEqual(len(result["runs"]), 16)
        schedules = [
            [
                run["version"]
                for run in result["runs"]
                if run["block"] == block
            ]
            for block in range(4)
        ]
        self.assertEqual(
            schedules,
            [
                ["baseline", "candidate", "baseline", "candidate"],
                ["candidate", "baseline", "candidate", "baseline"],
                ["baseline", "candidate", "baseline", "candidate"],
                ["candidate", "baseline", "candidate", "baseline"],
            ],
        )
        self.assertEqual(result["third_round_metrics"], ["score"])
        self.assertEqual(result["fourth_round_metrics"], ["score"])
        self.assertEqual(result["metrics"]["score"]["status"], "REGRESSION")
        self.assertEqual(result["metrics"]["late"]["status"], "PASS")
        self.assertIsNone(
            result["metrics"]["late"]["confirmation"]
        )
        self.assertIsNone(result["metrics"]["late"]["third_round"])
        self.assertIsNone(result["metrics"]["late"]["fourth_round"])

    def test_main_confirms_improvement_in_four_rounds(self) -> None:
        tests = [{"name": "synthetic"}]
        extractors = {
            "synthetic": [("score", None, "units/s", "higher")]
        }

        def fake_run_once(**kwargs: object) -> dict:
            version = str(kwargs["version"])
            return {
                "block": int(kwargs["block"]),
                "slot": int(kwargs["slot"]),
                "version": version,
                "elapsed_seconds": 0.01,
                "metric_text": "synthetic",
                "metrics": {
                    "score": {
                        "value": 100.0 if version == "baseline" else 104.0,
                        "unit": "units/s",
                        "direction": "higher",
                    }
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            argv = [
                "bench_compare.py",
                "--baseline-exec",
                "baseline",
                "--candidate-exec",
                "candidate",
                "--engine",
                "jit",
                "--platform",
                "synthetic",
                "--warmup-time",
                "0",
                "--time",
                "0.01",
                "--out-dir",
                str(out_dir),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bench_compare.run_tests, "TESTS", tests),
                patch.object(bench_compare, "METRIC_EXTRACTORS", extractors),
                patch.object(
                    bench_compare,
                    "command_for",
                    return_value=("fake", []),
                ),
                patch.object(
                    bench_compare,
                    "run_once",
                    side_effect=fake_run_once,
                ),
            ):
                exit_code = bench_compare.main()

            document = json.loads(
                (out_dir / "comparison.json").read_text(encoding="utf-8")
            )
            result = document["tests"]["synthetic"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(result["runs"]), 16)
        self.assertEqual(result["third_round_metrics"], ["score"])
        self.assertEqual(result["fourth_round_metrics"], ["score"])
        self.assertEqual(
            result["metrics"]["score"]["status"],
            "IMPROVEMENT",
        )
        self.assertIsNotNone(
            result["metrics"]["score"]["confirmation"]
        )
        self.assertIsNotNone(result["metrics"]["score"]["third_round"])
        self.assertIsNotNone(result["metrics"]["score"]["fourth_round"])


if __name__ == "__main__":
    unittest.main()
