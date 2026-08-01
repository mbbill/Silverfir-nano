import unittest

from ci.perf_confirm import regression_rows, verdict_lines


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


def wasmi_doc(statuses: dict[str, str]) -> dict:
    return {
        "metrics": {
            name: {"status": status, "delta_percent": -5.0}
            for name, status in statuses.items()
        }
    }


class RegressionRowTests(unittest.TestCase):
    def test_native_shape_keys_rows_by_test_and_metric(self) -> None:
        rows = regression_rows(
            native_doc({"c-ray": "REGRESSION", "coremark": "PASS"})
        )
        self.assertEqual(sorted(rows), ["c-ray:c-ray"])

    def test_wasmi_shape_keys_rows_by_metric(self) -> None:
        rows = regression_rows(
            wasmi_doc(
                {
                    "execute/fibonacci-tail": "REGRESSION",
                    "execute/mandelbrot": "NEGLIGIBLE",
                }
            )
        )
        self.assertEqual(sorted(rows), ["execute/fibonacci-tail"])

    def test_unknown_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            regression_rows({"schema_version": 9})


class VerdictTests(unittest.TestCase):
    def test_only_reproduced_rows_fail(self) -> None:
        primary = regression_rows(
            wasmi_doc(
                {
                    "execute/fibonacci-tail": "REGRESSION",
                    "execute/matrix_mul": "REGRESSION",
                }
            )
        )
        confirm = regression_rows(
            wasmi_doc(
                {
                    "execute/fibonacci-tail": "REGRESSION",
                    "execute/matrix_mul": "PASS",
                }
            )
        )
        lines, failed = verdict_lines(primary, confirm)
        self.assertTrue(failed)
        text = "\n".join(lines)
        self.assertIn("execute/fibonacci-tail", text)
        self.assertIn("primary run only", text)

    def test_nothing_reproduced_passes(self) -> None:
        primary = regression_rows(wasmi_doc({"a": "REGRESSION"}))
        confirm = regression_rows(wasmi_doc({"a": "PASS", "b": "REGRESSION"}))
        lines, failed = verdict_lines(primary, confirm)
        self.assertFalse(failed)
        self.assertIn("environment", "\n".join(lines))

    def test_clean_primary_passes(self) -> None:
        lines, failed = verdict_lines({}, {})
        self.assertFalse(failed)
        self.assertIn("flagged nothing", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
