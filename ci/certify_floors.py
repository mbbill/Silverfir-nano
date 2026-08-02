"""Generate certified measurement floors from identical-binary runs.

Feed it one or more downloaded run directories (gh run download) whose
runs compared identical binaries under the pinned harness; every row's
floor is the largest magnitude those runs measured. Rows below the
threshold are omitted: the global minimum effect already covers them.

    python3 -m ci.certify_floors --run-dir A --run-dir B \
        --out ci/measurement_floors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLATFORMS = ("x64-linux", "arm64-linux", "arm64-darwin", "x64-windows")
ENGINES = ("interp", "jit")


def canonical_job(raw: str) -> str | None:
    platform = next((p for p in PLATFORMS if p in raw), None)
    engine = next(
        (e for e in ENGINES if f"-{e}" in raw or f"/ {e}" in raw or raw.endswith(e)),
        None,
    )
    if platform is None or engine is None:
        return None
    family = "wasmi / " if "wasmi" in raw else ""
    return f"{family}{platform} / {engine}"


def document_rows(doc):
    tests = doc.get("tests")
    if isinstance(tests, dict):
        for test in tests.values():
            yield from test.get("metrics", {}).items()
        return
    metrics = doc.get("metrics")
    if isinstance(metrics, dict):
        yield from metrics.items()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="omit rows whose certified floor is below this percent",
    )
    args = parser.parse_args()

    floors: dict[str, float] = {}
    samples = 0
    for root in args.run_dir:
        for path in sorted(Path(root).rglob("comparison.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            artifact = next(
                (p.name for p in path.parents
                 if p.name.startswith(("performance-", "wasmi-performance-"))),
                path.parent.name,
            )
            job = canonical_job(artifact)
            if job is None:
                continue
            samples += 1
            for name, metric in document_rows(doc):
                delta = metric.get("delta_percent")
                if delta is None:
                    continue
                key = f"{job}|{name}"
                floors[key] = max(floors.get(key, 0.0), abs(float(delta)))

    kept = {
        key: round(value, 2)
        for key, value in sorted(floors.items())
        if value >= args.threshold
    }
    payload = {
        "_comment": (
            "Certified per-row measurement floors, percent: the largest "
            "magnitude identical binaries have measured on the pinned "
            "harness. A listed row's regression gate is 1.5x its floor "
            "and reports NOISY-FLOOR below it. Regenerate with "
            "ci/certify_floors.py from identical-binary run downloads "
            "after harness changes."
        ),
        "floors": kept,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"certified {len(kept)} floors from {samples} job documents "
        f"({len(floors)} rows seen)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
