"""One consolidated report for a whole performance run.

Walks every job's result artifact and emits a single markdown summary:
a per-job verdict table plus flat lists of the rows worth reading --
regressions, improvements, cannot-run, noisy-floor and placement rows --
so the run's outcome is legible without opening any job page.

Confirm-job artifacts are not separate jobs: each is matched back to its
primary job, its confirmed count appears in the verdict table, and the
flagged regressions are split by adjudication -- confirmed on an
independent runner, not reproduced there, or never adjudicated at all.
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


def format_delta(metric) -> str:
    delta = metric.get("delta_percent") if isinstance(metric, dict) else None
    return f"{delta:+.2f}%" if isinstance(delta, (int, float)) else "?"


def render_report(artifacts_dir: Path) -> str:
    per_job: dict[str, dict[str, list]] = {}
    confirm_rows: dict[str, dict[str, dict]] = {}
    for artifact, doc in iter_documents(artifacts_dir):
        if "-confirm-" in artifact:
            primary = artifact.replace("-confirm-", "-", 1)
            rows = confirm_rows.setdefault(primary, {})
            for name, metric in document_rows(doc):
                rows[name] = metric
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

    def confirm_status(job: str, name: str) -> str | None:
        row = confirm_rows.get(job, {}).get(name)
        return row.get("status", "") if isinstance(row, dict) else None

    lines.append(
        "| job | pass | regression | confirmed | improvement | cannot-run "
        "| noisy-floor | placement | unstable |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for job in sorted(per_job):
        bucket = per_job[job]
        confirmed = sum(
            confirm_status(job, name) == "REGRESSION"
            for name, _ in bucket["REGRESSION"]
        )
        lines.append(
            f"| {job} | {len(bucket['PASS-COUNT'])} "
            f"| {len(bucket['REGRESSION'])} "
            f"| {confirmed} "
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

    confirmed_rows: list[tuple[str, str, dict, dict]] = []
    dismissed_rows: list[tuple[str, str, dict, dict]] = []
    unadjudicated_rows: list[tuple[str, str, dict, str]] = []
    for job, name, metric in rows_for("REGRESSION"):
        row_confirm = confirm_rows.get(job, {}).get(name)
        status = (
            row_confirm.get("status", "")
            if isinstance(row_confirm, dict)
            else None
        )
        if status == "REGRESSION":
            confirmed_rows.append((job, name, metric, row_confirm))
        elif status == "CANNOT-RUN":
            unadjudicated_rows.append((
                job, name, metric,
                "crashed on the confirmation runner; fails closed",
            ))
        elif status is None:
            unadjudicated_rows.append((
                job, name, metric,
                "not re-measured; confirmation did not run",
            ))
        else:
            dismissed_rows.append((job, name, metric, row_confirm))

    if confirmed_rows:
        lines.extend([
            "", "### Regressions confirmed on an independent runner", "",
        ])
        for job, name, metric, row_confirm in confirmed_rows:
            lines.append(
                f"- **{name}** primary {format_delta(metric)}, "
                f"confirmation {format_delta(row_confirm)} — {job}"
            )

    if dismissed_rows:
        lines.extend([
            "",
            "### Flagged by one runner, not reproduced on re-measure",
            "",
        ])
        for job, name, metric, row_confirm in dismissed_rows:
            lines.append(
                f"- **{name}** primary {format_delta(metric)}, "
                f"re-measure {format_delta(row_confirm)} — {job}"
            )

    if unadjudicated_rows:
        lines.extend(["", "### Flagged regressions without a verdict", ""])
        for job, name, metric, reason in unadjudicated_rows:
            lines.append(
                f"- **{name}** {format_delta(metric)} — {job} — {reason}"
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
            lines.append(f"- **{name}** {format_delta(metric)} — {job}")

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
