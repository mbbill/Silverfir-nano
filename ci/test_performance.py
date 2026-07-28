import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ci.performance as bench_compare
from ci.performance import (
    classify_metrics,
    family_adjusted_probability,
    metric_plans,
    probability_summary,
    required_pairs,
    required_pairs_for_direction,
    student_t_cdf,
)


def measured_metric(deltas_percent: list[float]) -> dict:
    ratios = [1.0 + delta / 100.0 for delta in deltas_percent]
    statistics = probability_summary(ratios)
    return {
        "unit": "units/s",
        "direction": "higher",
        "baseline_samples": [100.0] * len(ratios),
        "candidate_samples": [100.0 * ratio for ratio in ratios],
        "baseline_geomean": 100.0,
        "candidate_geomean": 100.0 * math.exp(
            statistics["mean_log_ratio"]
        ),
        **statistics,
    }


class ProbabilityGateTests(unittest.TestCase):
    def test_coremark_preserves_official_invocation(self) -> None:
        coremark = bench_compare.run_tests.TESTS[0]
        sha256 = bench_compare.run_tests.TESTS[1]

        self.assertEqual(
            bench_compare.run_tests.program_args(coremark, 2.0),
            ["coremark.wasm"],
        )
        self.assertEqual(
            bench_compare.run_tests.program_args(
                coremark, 2.0, correctness_only=True
            ),
            ["coremark.wasm", "0", "0", "102", "1"],
        )
        self.assertEqual(
            bench_compare.run_tests.program_args(sha256, 2.0),
            ["sha256.wasm", "2.0"],
        )
        self.assertEqual(
            bench_compare.run_tests.program_args(
                sha256, 2.0, correctness_only=True
            ),
            ["sha256.wasm", "--bench-correctness"],
        )

    def test_student_t_cdf_known_values_and_symmetry(self) -> None:
        self.assertAlmostEqual(student_t_cdf(0.0, 1), 0.5, places=12)
        self.assertAlmostEqual(student_t_cdf(1.0, 1), 0.75, places=12)
        for degrees in (2, 5, 30):
            for value in (0.25, 1.0, 3.0):
                self.assertAlmostEqual(
                    student_t_cdf(-value, degrees),
                    1.0 - student_t_cdf(value, degrees),
                    places=12,
                )

    def test_probability_summary_recognizes_direction(self) -> None:
        regression = measured_metric([-2.0] * 6)
        improvement = measured_metric([2.0] * 6)
        stable = measured_metric([-2.0, 2.0, -1.5, 1.5])

        self.assertEqual(regression["probability_regression"], 1.0)
        self.assertEqual(improvement["probability_improvement"], 1.0)
        self.assertLess(
            abs(stable["probability_regression"] - 0.5),
            0.03,
        )

    def test_required_pairs_grows_with_volatility(self) -> None:
        stable = measured_metric([-2.0, -2.1, -1.9, -2.0])
        noisy = measured_metric([-6.0, 2.0, -4.0, 0.0])

        stable_pairs = required_pairs(
            stable,
            probability=0.999,
            minimum_pairs=6,
            maximum_pairs=24,
        )
        noisy_pairs = required_pairs(
            noisy,
            probability=0.999,
            minimum_pairs=6,
            maximum_pairs=24,
        )

        self.assertEqual(stable_pairs, 6)
        self.assertTrue(noisy_pairs is None or noisy_pairs > stable_pairs)

    def test_family_probability_covers_metrics_jobs_and_adaptive_looks(
        self,
    ) -> None:
        adjusted = family_adjusted_probability(
            0.9999,
            metric_count=15,
            job_count=8,
            maximum_looks=10,
        )
        self.assertAlmostEqual(adjusted, 1.0 - 0.0001 / 1200, places=15)

    def test_one_percent_regression_is_detectable_when_stable(self) -> None:
        pilot = measured_metric([-0.8, -1.2, -0.9, -1.1])
        confirmation = measured_metric(
            [-0.8, -1.2, -0.9, -1.1] * 6
        )
        adjusted = family_adjusted_probability(
            0.9999,
            metric_count=15,
            job_count=8,
            maximum_looks=10,
        )

        target_pairs = required_pairs(
            pilot,
            probability=adjusted,
            minimum_pairs=6,
            maximum_pairs=24,
        )

        self.assertIsNotNone(target_pairs)
        self.assertLessEqual(target_pairs, 24)
        self.assertGreaterEqual(
            confirmation["probability_regression"],
            adjusted,
        )

    def test_directional_pilot_can_confirm_when_variance_forecast_cannot(
        self,
    ) -> None:
        initial = {
            "small": measured_metric([-0.3, -1.8, -1.0, 0.2])
        }
        plans = metric_plans(
            initial,
            regression_probability=0.9999,
            improvement_probability=0.999,
            minimum_pairs=6,
            maximum_pairs=24,
            pilot_probability=0.8,
        )

        self.assertIsNone(plans["small"]["target_pairs"])
        self.assertGreater(
            plans["small"]["pilot_direction_probability"],
            0.8,
        )
        self.assertTrue(plans["small"]["selected"])

    def test_classification_freezes_initial_direction(self) -> None:
        initial = {
            "regression": measured_metric([-2.0] * 4),
            "late": measured_metric([0.0] * 4),
            "improvement": measured_metric([3.0] * 4),
        }
        plans = metric_plans(
            initial,
            regression_probability=0.9999,
            improvement_probability=0.999,
            minimum_pairs=6,
            maximum_pairs=24,
        )
        final = {
            "regression": measured_metric([-2.0] * 6),
            "improvement": measured_metric([3.0] * 6),
            # This must be ignored because it had no initial direction.
            "late": measured_metric([-20.0] * 6),
        }
        result = classify_metrics(
            initial=initial,
            final=final,
            plans=plans,
            regression_probability=0.9999,
            improvement_probability=0.999,
        )

        self.assertEqual(result["regression"]["status"], "REGRESSION")
        self.assertEqual(result["improvement"]["status"], "IMPROVEMENT")
        self.assertEqual(result["late"]["status"], "PASS")
        self.assertEqual(result["late"]["pair_count"], 4)

    def test_unstable_signal_is_not_selected_within_budget(self) -> None:
        initial = {
            "unstable": measured_metric([-20.0, 20.0, -18.0, 18.0])
        }
        plans = metric_plans(
            initial,
            regression_probability=0.9999,
            improvement_probability=0.999,
            minimum_pairs=6,
            maximum_pairs=24,
        )
        self.assertFalse(plans["unstable"]["selected"])

    def test_calibrated_false_improvement_stays_below_symmetric_gate(
        self,
    ) -> None:
        measured = measured_metric([
            1.5037593984962518,
            1.5037593984962518,
            1.4925373134328404,
            -0.7462686567164202,
            1.4925373134328404,
            0.746268656716409,
            1.5037593984962518,
            0.0,
            0.746268656716409,
            0.746268656716409,
            0.7518796992481259,
            0.746268656716409,
            0.7518796992481259,
            1.5037593984962518,
        ])
        initial = {"score": measured_metric([1.0] * 4)}
        plans = metric_plans(
            initial,
            regression_probability=0.9999,
            improvement_probability=0.9999,
            minimum_pairs=6,
            maximum_pairs=24,
        )
        result = classify_metrics(
            initial=initial,
            final={"score": measured},
            plans=plans,
            regression_probability=0.9999,
            improvement_probability=0.9999,
        )

        self.assertGreater(measured["probability_improvement"], 0.999)
        self.assertLess(measured["probability_improvement"], 0.9999)
        self.assertEqual(result["score"]["status"], "PASS")

    def test_calibrated_false_regression_stays_below_family_gate(
        self,
    ) -> None:
        measured = measured_metric([
            -1.5625,
            -2.3622047244094557,
            -0.78125,
            -0.7874015748031482,
            0.0,
            -3.1007751937984556,
            -2.3255813953488413,
            -0.78125,
            -0.7751937984496138,
            0.0,
            -0.78125,
            -2.34375,
            -1.5748031496062964,
            -0.7874015748031482,
            -3.0534351145038214,
            -2.3076923076923106,
        ])
        adjusted = family_adjusted_probability(
            0.9999,
            metric_count=15,
            job_count=8,
            maximum_looks=10,
        )
        initial = {"score": measured_metric([-1.0] * 4)}
        plans = metric_plans(
            initial,
            regression_probability=adjusted,
            improvement_probability=adjusted,
            minimum_pairs=6,
            maximum_pairs=24,
        )
        result = classify_metrics(
            initial=initial,
            final={"score": measured},
            plans=plans,
            regression_probability=adjusted,
            improvement_probability=adjusted,
        )

        self.assertGreater(measured["probability_regression"], 0.9999)
        self.assertLess(measured["probability_regression"], adjusted)
        self.assertEqual(result["score"]["status"], "RECOVERED")

    def test_confirmation_reestimates_after_pilot_underestimates_pairs(
        self,
    ) -> None:
        confirmation = measured_metric(
            [-2.0, -2.0, -2.0, -2.0, -2.0, 0.5]
        )

        next_pairs = required_pairs_for_direction(
            confirmation,
            direction="regression",
            probability=0.9999,
            minimum_pairs=8,
            maximum_pairs=24,
        )
        opposite = required_pairs_for_direction(
            measured_metric([1.0] * 6),
            direction="regression",
            probability=0.9999,
            minimum_pairs=8,
            maximum_pairs=24,
        )

        self.assertEqual(next_pairs, 14)
        self.assertIsNone(opposite)


class ScriptIntegrationTests(unittest.TestCase):
    def test_correctness_mode_runs_a_and_b_without_metrics_gate(self) -> None:
        calls = []

        def fake_run_test(*args: object, **kwargs: object) -> tuple:
            calls.append((args, kwargs))
            return "synthetic", "PASS", "validated", 0.01

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            with patch.object(
                bench_compare.run_tests,
                "run_test",
                side_effect=fake_run_test,
            ):
                exit_code = bench_compare.run_correctness_suite(
                    baseline_command=("baseline", []),
                    candidate_command=("candidate", []),
                    selected_tests=[{"name": "synthetic"}],
                    time_target=2.0,
                    platform="rv32",
                    engine="interp",
                    baseline_sha="a",
                    candidate_sha="b",
                    out_dir=out_dir,
                )
            document = json.loads(
                (out_dir / "correctness.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[1]["correctness_only"] for call in calls))
        self.assertEqual(document["mode"], "correctness")
        self.assertEqual(
            document["tests"]["synthetic"]["baseline"]["status"],
            "PASS",
        )
        self.assertEqual(
            document["tests"]["synthetic"]["candidate"]["status"],
            "PASS",
        )

    def test_main_targets_selected_metric_and_ignores_late_change(self) -> None:
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
                            if baseline or block < 2
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
                "--blocks",
                "2",
                "--min-pairs",
                "6",
                "--max-pairs",
                "24",
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
        self.assertEqual(document["schema_version"], 7)
        self.assertEqual(len(result["runs"]), 20)
        schedules = [
            [
                run["version"]
                for run in result["runs"]
                if run["block"] == block
            ]
            for block in range(5)
        ]
        self.assertEqual(
            schedules,
            [
                ["baseline", "candidate", "baseline", "candidate"],
                ["candidate", "baseline", "candidate", "baseline"],
                ["baseline", "candidate", "baseline", "candidate"],
                ["candidate", "baseline", "candidate", "baseline"],
                ["baseline", "candidate", "baseline", "candidate"],
            ],
        )
        self.assertEqual(result["candidate_metrics"], ["score"])
        self.assertEqual(result["metrics"]["score"]["status"], "REGRESSION")
        self.assertEqual(result["metrics"]["score"]["pair_count"], 6)
        self.assertEqual(result["metrics"]["late"]["status"], "PASS")
        self.assertEqual(result["metrics"]["late"]["pair_count"], 4)
        self.assertEqual(result["confirmation_looks"], [6])

    def test_main_confirms_improvement_at_planned_pair_count(self) -> None:
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
                "--blocks",
                "2",
                "--min-pairs",
                "6",
                "--max-pairs",
                "24",
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
        self.assertEqual(len(result["runs"]), 20)
        self.assertEqual(
            result["metrics"]["score"]["status"],
            "IMPROVEMENT",
        )
        self.assertEqual(result["metrics"]["score"]["pair_count"], 6)
        self.assertEqual(result["confirmation_looks"], [6])

    def test_main_extends_confirmation_when_pilot_plan_is_too_short(
        self,
    ) -> None:
        tests = [{"name": "synthetic"}]
        extractors = {
            "synthetic": [("score", None, "units/s", "higher")]
        }
        confirmation_deltas = [
            -2.0,
            -2.0,
            -2.0,
            -2.0,
            -2.0,
            0.5,
        ] + [-2.0] * 18

        def fake_run_once(**kwargs: object) -> dict:
            block = int(kwargs["block"])
            slot = int(kwargs["slot"])
            version = str(kwargs["version"])
            if version == "baseline":
                value = 100.0
            elif block < 2:
                value = 98.0
            else:
                pair_index = (block - 2) * 2 + slot // 2
                value = 100.0 * (
                    1.0 + confirmation_deltas[pair_index] / 100.0
                )
            return {
                "block": block,
                "slot": slot,
                "version": version,
                "elapsed_seconds": 0.01,
                "metric_text": "synthetic",
                "metrics": {
                    "score": {
                        "value": value,
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
                "--blocks",
                "2",
                "--min-pairs",
                "6",
                "--max-pairs",
                "24",
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
        self.assertEqual(result["confirmation_looks"], [6, 16])
        self.assertEqual(result["metrics"]["score"]["pair_count"], 16)
        self.assertEqual(result["metrics"]["score"]["status"], "REGRESSION")


if __name__ == "__main__":
    unittest.main()
