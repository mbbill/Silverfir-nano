"""One consolidated report for a whole performance run.

Walks every job's result artifact and emits a single markdown summary:
a per-job verdict table plus flat lists of the rows worth reading --
regressions, improvements, cannot-run, noisy-floor and placement rows --
so the run's outcome is legible without opening any job page.

Confirm-job artifacts are not separate jobs: each is matched back to its
primary job and its verdict is stamped onto that job's regression rows,
so the report says whether a listed regression reproduced on independent
hardware, was dismissed there, or was never re-measured at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INTERESTING = (
    "REGRESSION",
    "IMPROVEMENT",
    "CANNOT-RUN",
    "NOISY-FLOOR",
    "PLACEMENT",
    "UNSTABLE",
)


def iter_documents(root: Path, name: str = "comparison.json"):
    for path in sorted(root.rglob(name)):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        artifact = next(
            (p.name for p in path.parents if p.name.startswith(
                ("performance-", "wasmi-performance-")
            )),
            path.parent.name,
        )
        yield artifact, doc


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


def confirm_note(name: str, confirm_statuses: dict[str, str] | None) -> str:
    """How the confirmation layer adjudicated one flagged regression."""
    if confirm_statuses is None:
        return "not re-measured (confirmation did not run)"
    status = confirm_statuses.get(name)
    if status == "REGRESSION":
        return "confirmed on an independent runner"
    if status == "CANNOT-RUN":
        return "crashed on the confirmation runner (fails closed)"
    if status is None:
        return "not re-measured (confirmation did not run)"
    return "not reproduced on an independent runner"


def render_report(artifacts_dir: Path) -> str:
    per_job: dict[str, dict[str, list]] = {}
    confirm_rows: dict[str, dict[str, str]] = {}
    for artifact, doc in iter_documents(artifacts_dir):
        if "-confirm-" in artifact:
            primary = artifact.replace("-confirm-", "-", 1)
            statuses = confirm_rows.setdefault(primary, {})
            for name, metric in document_rows(doc):
                statuses[name] = metric.get("status", "")
            continue
        bucket = per_job.setdefault(
            artifact, {status: [] for status in INTERESTING}
        )
        bucket.setdefault("PASS-COUNT", [])
        for name, metric in document_rows(doc):
            status = metric.get("status", "")
            if status in INTERESTING:
                bucket[status].append((name, metric))
            else:
                bucket["PASS-COUNT"].append(name)

    failures = [
        (artifact, doc)
        for artifact, doc in iter_documents(artifacts_dir, "failure.json")
        if artifact not in per_job
    ]

    lines = ["## Run report", ""]
    if not per_job and not failures:
        lines.append("No result artifacts found.")
        return "\n".join(lines)

    lines.append(
        "| job | pass | regression | improvement | cannot-run "
        "| noisy-floor | placement | unstable |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for job in sorted(per_job):
        bucket = per_job[job]
        lines.append(
            f"| {job} | {len(bucket['PASS-COUNT'])} "
            f"| {len(bucket['REGRESSION'])} "
            f"| {len(bucket['IMPROVEMENT'])} "
            f"| {len(bucket['CANNOT-RUN'])} "
            f"| {len(bucket['NOISY-FLOOR'])} "
            f"| {len(bucket['PLACEMENT'])} "
            f"| {len(bucket['UNSTABLE'])} |"
        )

    def rows_for(status: str):
        return [
            (job, name, metric)
            for job, bucket in sorted(per_job.items())
            for name, metric in sorted(
                bucket[status],
                key=lambda row: row[1].get("delta_percent") or 0,
            )
        ]

    regressions = rows_for("REGRESSION")
    if regressions:
        lines.extend(["", "### Regressions", ""])
        for job, name, metric in regressions:
            note = confirm_note(name, confirm_rows.get(job))
            lines.append(
                f"- **{name}** {metric['delta_percent']:+.2f}% — {job} "
                f"— {note}"
            )

    cannot_run = rows_for("CANNOT-RUN")
    if cannot_run:
        lines.extend(["", "### Rows that could not be measured", ""])
        for job, name, metric in cannot_run:
            lines.append(
                f"- **{name}** — {metric.get('failed_version', '?')} "
                f"process crashed — {job}"
            )

    titles = (
        ("IMPROVEMENT", "Improvements"),
        ("NOISY-FLOOR", "Noisy-floor rows (measured, not gated)"),
        ("PLACEMENT", "Placement-classified rows"),
        ("UNSTABLE", "Runner drift on identical binaries"),
    )
    for status, title in titles:
        rows = rows_for(status)
        if not rows:
            continue
        lines.extend(["", f"### {title}", ""])
        for job, name, metric in rows:
            delta = metric.get("delta_percent")
            shown = f"{delta:+.2f}%" if isinstance(delta, (int, float)) else "?"
            lines.append(f"- **{name}** {shown} — {job}")

    if failures:
        lines.extend(["", "### Jobs with no verdict", ""])
        for artifact, doc in failures:
            error = str(doc.get("error", "")).strip().splitlines()
            head = error[0] if error else "setup or measurement error"
            lines.append(f"- **{artifact}** — {head}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render_report(args.artifacts_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
