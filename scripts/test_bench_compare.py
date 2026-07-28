import unittest

from scripts.bench_compare import classify_metrics, regression_candidates


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


class TwoStageGateTests(unittest.TestCase):
    def test_confirmation_and_improvement_rules(self) -> None:
        initial = {
            "recovers": metric(-1.5),
            "persists": metric(-1.2),
            "new_confirmation_regression": metric(-0.5),
            "improves": metric(3.1),
        }
        confirmation = {
            "recovers": metric(-0.5),
            "persists": metric(-1.1),
            # This must be ignored because it passed the initial screen.
            "new_confirmation_regression": metric(-9.0),
        }

        self.assertEqual(
            regression_candidates(initial, 1.0),
            {"recovers", "persists"},
        )
        result = classify_metrics(
            initial=initial,
            confirmation=confirmation,
            regression_threshold=1.0,
            improvement_threshold=3.0,
        )

        self.assertEqual(result["recovers"]["status"], "RECOVERED")
        self.assertEqual(result["persists"]["status"], "REGRESSION")
        self.assertEqual(
            result["new_confirmation_regression"]["status"],
            "PASS",
        )
        self.assertIsNone(
            result["new_confirmation_regression"]["confirmation"]
        )
        self.assertEqual(result["improves"]["status"], "IMPROVEMENT")

    def test_threshold_boundaries_are_inclusive_passes(self) -> None:
        initial = {
            "negative_boundary": metric(-1.0),
            "positive_boundary": metric(3.0),
        }
        result = classify_metrics(
            initial=initial,
            confirmation={},
            regression_threshold=1.0,
            improvement_threshold=3.0,
        )

        self.assertEqual(result["negative_boundary"]["status"], "PASS")
        self.assertEqual(result["positive_boundary"]["status"], "PASS")

    def test_selected_metric_requires_confirmation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "selected metric has no confirmation",
        ):
            classify_metrics(
                initial={"missing": metric(-1.01)},
                confirmation={},
                regression_threshold=1.0,
                improvement_threshold=3.0,
            )


if __name__ == "__main__":
    unittest.main()
