"""One consolidated report for a whole performance run.

Walks every job's result artifact and emits a single markdown summary:
a per-job verdict table plus flat lists of the rows worth reading --
regressions, improvements, noisy-floor and placement rows -- so the
run's outcome is legible without opening any job page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INTERESTING = (
    "REGRESSION",
    "IMPROVEMENT",
    "NOISY-FLOOR",
    "PLACEMENT",
    "UNSTABLE",
)


def iter_documents(root: Path):
    for path in sorted(root.rglob("comparison.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        label = path.parent.name
        artifact = next(
            (p.name for p in path.parents if p.name.startswith(
                ("performance-", "wasmi-performance-")
            )),
            path.parent.name,
        )
        yield artifact, label, doc


def document_rows(doc):
    tests = doc.get("tests")
    if isinstance(tests, dict):
        for test_name, test in tests.items():
            for metric_name, metric in test.get("metrics", {}).items():
                yield metric_name, metric
        return
    metrics = doc.get("metrics")
    if isinstance(metrics, dict):
        yield from metrics.items()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()

    per_job: dict[str, dict[str, list]] = {}
    for artifact, label, doc in iter_documents(args.artifacts_dir):
        bucket = per_job.setdefault(artifact, {k: [] for k in INTERESTING})
        bucket.setdefault("PASS-COUNT", [])
        for name, metric in document_rows(doc):
            status = metric.get("status", "")
            if status in INTERESTING:
                bucket[status].append((name, metric.get("delta_percent")))
            else:
                bucket["PASS-COUNT"].append(name)

    lines = ["## Run report", ""]
    if not per_job:
        lines.append("No result artifacts found.")
        print("\n".join(lines))
        return 0

    lines.append(
        "| job | pass | regression | improvement | noisy-floor "
        "| placement | unstable |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for job in sorted(per_job):
        b = per_job[job]
        lines.append(
            f"| {job} | {len(b['PASS-COUNT'])} | {len(b['REGRESSION'])} "
            f"| {len(b['IMPROVEMENT'])} | {len(b['NOISY-FLOOR'])} "
            f"| {len(b['PLACEMENT'])} | {len(b['UNSTABLE'])} |"
        )

    titles = (
        ("REGRESSION", "Regressions"),
        ("IMPROVEMENT", "Improvements"),
        ("NOISY-FLOOR", "Noisy-floor rows (measured, not gated)"),
        ("PLACEMENT", "Placement-classified rows"),
        ("UNSTABLE", "Runner drift on identical binaries"),
    )
    for status, title in titles:
        rows = [
            (job, name, delta)
            for job, b in sorted(per_job.items())
            for name, delta in sorted(b[status], key=lambda r: r[1] or 0)
        ]
        if not rows:
            continue
        lines.extend(["", f"### {title}", ""])
        for job, name, delta in rows:
            shown = f"{delta:+.2f}%" if isinstance(delta, (int, float)) else "?"
            lines.append(f"- **{name}** {shown} — {job}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
