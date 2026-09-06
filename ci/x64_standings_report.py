"""Cross-engine standings table for the x64 tuning branch.

Diagnostic tooling for .github/workflows/x64-standings.yml: reads the
Criterion estimates one wasmi-benchmarks run left behind and renders a
single within-run comparison table (Silverfir-nano JIT vs the comparison
engines). No statistical win claim: this is a same-run snapshot, and the paired
revision performance workflow remains the regression gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def collect_cases(criterion_dir: Path) -> dict[str, dict[str, float]]:
    """Return {case: {runtime: mean_ns}} for every execute group found."""
    cases: dict[str, dict[str, float]] = {}
    for estimates in sorted(criterion_dir.rglob("estimates.json")):
        rel = estimates.relative_to(criterion_dir)
        parts = rel.parts
        # <group>/<runtime...>/new/estimates.json
        if len(parts) < 4 or parts[-2] != "new":
            continue
        group = parts[0]
        if not group.startswith("execute"):
            continue
        # <group>/<runtime-and-input>/new/estimates.json — Criterion
        # flattens "<runtime>/<input>" bench ids into one directory name
        # joined by "_" (e.g. silverfir-nano.jit_1000000). The numeric
        # suffix is a benchmark input and belongs to the case, not the
        # engine column.
        component = parts[1]
        if component == "report":
            continue
        prefix, sep, suffix = component.rpartition("_")
        if sep and suffix.isdigit():
            runtime, case_input = prefix, suffix
        else:
            runtime, case_input = component, ""
        doc = json.loads(estimates.read_text(encoding="utf-8"))
        mean = (doc.get("mean") or {}).get("point_estimate")
        if mean is None:
            continue
        case = group[len("execute"):].lstrip("_/") or group
        inputs = "/".join(part for part in (case_input, *parts[2:-2]) if part)
        if inputs:
            case = f"{case}/{inputs}"
        value = float(mean)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid timing in {estimates}: {value}")
        row = cases.setdefault(case, {})
        if runtime in row:
            raise ValueError(f"duplicate timing for {case}/{runtime}")
        row[runtime] = value
    return cases


def pick_nano(runtimes: list[str]) -> str:
    silverfir = [name for name in runtimes if "silverfir" in name]
    if not silverfir:
        raise SystemExit(
            "no silverfir runtime found in the Criterion results; "
            "the [patch] to the local checkout likely did not take effect"
        )
    jit = [name for name in silverfir if "jit" in name]
    return (jit or silverfir)[0]


def fmt_time(ns: float) -> str:
    ms = ns / 1e6
    if ms >= 100:
        return f"{ms:.0f} ms"
    if ms >= 10:
        return f"{ms:.1f} ms"
    if ms >= 1:
        return f"{ms:.2f} ms"
    return f"{ns / 1e3:.1f} µs"


def geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def render(cases: dict[str, dict[str, float]], notes: list[str]) -> str:
    runtimes = sorted({name for row in cases.values() for name in row})
    nano = pick_nano(runtimes)
    others = [name for name in runtimes if name != nano]

    def worst_gap(row: dict[str, float]) -> float:
        gaps = [
            row[nano] / row[name]
            for name in others
            if name in row and nano in row
        ]
        return max(gaps, default=0.0)

    lines = ["# x64 cross-engine standings", ""]
    lines += [f"- {note}" for note in notes]
    lines += [
        "",
        f"Times are Criterion mean point estimates; lower is better. "
        f"`nano/<engine>` above 1.00 means `{nano}` is slower than that "
        f"engine on that case. Rows are sorted widest gap first.",
        "",
    ]
    header = (
        ["case", nano]
        + others
        + [f"nano/{name}" for name in others]
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    ratios: dict[str, list[float]] = {name: [] for name in others}
    for case in sorted(cases, key=lambda c: worst_gap(cases[c]), reverse=True):
        row = cases[case]
        cells = [case]
        cells.append(fmt_time(row[nano]) if nano in row else "—")
        for name in others:
            cells.append(fmt_time(row[name]) if name in row else "—")
        for name in others:
            if nano in row and name in row:
                gap = row[nano] / row[name]
                ratios[name].append(gap)
                cells.append(f"{gap:.2f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    geo_cells = ["**geomean**", ""] + [""] * len(others)
    for name in others:
        mean = geomean(ratios[name])
        geo_cells.append(f"**{mean:.2f}**" if mean is not None else "—")
    lines.append("| " + " | ".join(geo_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def validate_field(cases: dict[str, dict[str, float]], expected_cases: int) -> None:
    if len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} cases, found {len(cases)}")
    runtimes = sorted({name for row in cases.values() for name in row})
    nano = pick_nano(runtimes)
    expected = (
        {nano, "wasmtime.cranelift", "v8"}
        if "jit" in nano else
        {nano, "stitch", "wasm3.eager", "wasmi-v2.eager.checked"}
    )
    for case, row in cases.items():
        if set(row) != expected:
            raise ValueError(f"incomplete runtime field for {case}: {sorted(row)}")
        if any(not math.isfinite(value) or value <= 0 for value in row.values()):
            raise ValueError(f"invalid timing for {case}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterion-dir", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args(argv)

    if not args.criterion_dir.is_dir():
        raise SystemExit(f"criterion dir not found: {args.criterion_dir}")
    cases = collect_cases(args.criterion_dir)
    if not cases:
        raise SystemExit(
            f"no execute estimates under {args.criterion_dir}; "
            "did the bench run finish?"
        )

    validate_field(cases, args.expected_cases)
    report = render(cases, args.note)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report, encoding="utf-8")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {"notes": args.note, "mean_ns": cases},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
