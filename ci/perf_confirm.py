"""Cross-run confirmation verdict for the performance gate.

A performance comparison runs on a dynamically allocated runner, and three
runs of byte-identical sources have measured three disjoint sets of
statistically certain 1-9% regressions; a single run's verdict therefore
confounds the source change with the runner draw. The primary run screens
at the family-corrected probability; when it flags rows, a second run on a
fresh runner re-measures, and only a row the confirmation run also flags
-- a pre-registered, single hypothesis on independent hardware -- fails
the gate.

This module intersects the two runs' result documents. It understands
both document shapes: the native driver's ``tests.{name}.metrics`` and
the wasmi driver's flat ``metrics``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def regression_rows(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Metric rows with status REGRESSION, keyed test:metric."""
    rows: dict[str, dict[str, Any]] = {}
    tests = document.get("tests")
    if isinstance(tests, dict):
        for test_name, test in tests.items():
            for metric_name, metric in test.get("metrics", {}).items():
                if metric.get("status") == "REGRESSION":
                    rows[f"{test_name}:{metric_name}"] = metric
        return rows
    metrics = document.get("metrics")
    if isinstance(metrics, dict):
        for metric_name, metric in metrics.items():
            if metric.get("status") == "REGRESSION":
                rows[metric_name] = metric
        return rows
    raise ValueError("document has neither 'tests' nor 'metrics'")


def verdict_lines(
    primary_rows: dict[str, dict[str, Any]],
    confirm_rows: dict[str, dict[str, Any]],
) -> tuple[list[str], bool]:
    confirmed = sorted(set(primary_rows) & set(confirm_rows))
    unreproduced = sorted(set(primary_rows) - set(confirm_rows))
    lines = ["## Cross-run confirmation", ""]
    if not primary_rows:
        lines.append("Primary run flagged nothing; confirmation not needed.")
        return lines, False
    if confirmed:
        lines.append(
            "The following regressions reproduced on an independent "
            "runner and fail the gate:"
        )
        lines.extend(
            f"- **{name}**: primary "
            f"{primary_rows[name]['delta_percent']:+.2f}%, confirmation "
            f"{confirm_rows[name]['delta_percent']:+.2f}%"
            for name in confirmed
        )
    if unreproduced:
        lines.append("")
        lines.append(
            "Flagged by the primary run only -- an independent runner did "
            "not reproduce these, so they read as that run's environment:"
        )
        lines.extend(
            f"- {name}: primary {primary_rows[name]['delta_percent']:+.2f}%"
            for name in unreproduced
        )
    return lines, bool(confirmed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--confirm", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    confirm = json.loads(args.confirm.read_text(encoding="utf-8"))
    lines, failed = verdict_lines(
        regression_rows(primary), regression_rows(confirm)
    )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.summary_out is not None:
        args.summary_out.write_text(text, encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
