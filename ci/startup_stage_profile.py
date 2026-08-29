#!/usr/bin/env python3
"""Run and reduce the temporary interpreter startup stage profiler.

The Rust driver emits one JSON object per instantiate-and-drop sample. This
reducer keeps those raw samples, derives mutually exclusive pipeline buckets
per sample, and reports medians plus both equal-workload and time-weighted
shares. Timing is intended for dev CI runners only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable


CASES = (
    ("bz2", "res/wasm/bz2.wasm"),
    ("pulldown-cmark", "res/wasm/pulldown-cmark.wasm"),
    ("spidermonkey", "res/wasm/spidermonkey.wasm"),
    ("ffmpeg", "res/wasm/ffmpeg.wasm"),
    ("coremark", "res/wasm/coremark.wasm"),
    ("argon2", "res/rust/cases/argon2/out.wasm"),
    ("erc20", "res/wasm/erc20.wasm"),
)

PARSER_CHILDREN = (
    "parser.header",
    "parser.plan",
    "parser.section.custom",
    "parser.section.type",
    "parser.section.import",
    "parser.section.function",
    "parser.section.table",
    "parser.section.memory",
    "parser.section.tag",
    "parser.section.global",
    "parser.section.export",
    "parser.section.start",
    "parser.section.element",
    "parser.section.data_count",
    "parser.section.code",
    "parser.section.data",
    "parser.finalize",
)

EXCLUSIVE_ORDER = (
    *PARSER_CHILDREN,
    "parser.other",
    "predecode.decode",
    "predecode.scratch",
    "predecode.pinned_census",
    "predecode.lowering_control",
    "instance.setup",
    "instance.memories",
    "instance.globals",
    "instance.tables",
    "instance.stack_deferred",
    "instance.build.other",
    "link.handler_selection",
    "link.cell_transform",
    "link.call_fixup",
    "link.finalize",
    "link.other",
    "instance.element_segments",
    "instance.data_segments",
    "instance.lease",
    "drop",
    "startup.other",
)


def median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(statistics.median(values))


def nanos(sample: dict[str, Any], name: str) -> int:
    return int(sample["stages"].get(name, {}).get("nanos", 0))


def residual(parent: int, *children: int) -> tuple[int, int]:
    value = parent - sum(children)
    return max(0, value), min(0, value)


def derive(sample: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    exclusive: dict[str, int] = {
        name: nanos(sample, name) for name in PARSER_CHILDREN
    }
    negative: dict[str, int] = {}

    parser_other, under = residual(
        nanos(sample, "parser.total"),
        *(exclusive[name] for name in PARSER_CHILDREN),
    )
    exclusive["parser.other"] = parser_other
    negative["parser.other"] = under

    predecode_total = nanos(sample, "predecode.total")
    decode = nanos(sample, "predecode.decode")
    scratch = nanos(sample, "predecode.scratch")
    pinned = nanos(sample, "predecode.pinned_census")
    lowering, under = residual(predecode_total, decode, scratch, pinned)
    exclusive.update(
        {
            "predecode.decode": decode,
            "predecode.scratch": scratch,
            "predecode.pinned_census": pinned,
            "predecode.lowering_control": lowering,
        }
    )
    negative["predecode.lowering_control"] = under

    setup = nanos(sample, "instance.setup")
    memories = nanos(sample, "instance.memories")
    globals_ns = nanos(sample, "instance.globals")
    tables = nanos(sample, "instance.tables")
    stack = nanos(sample, "instance.stack_deferred")
    build_other, under = residual(
        nanos(sample, "instance.build.total"),
        predecode_total,
        setup,
        memories,
        globals_ns,
        tables,
        stack,
    )
    exclusive.update(
        {
            "instance.setup": setup,
            "instance.memories": memories,
            "instance.globals": globals_ns,
            "instance.tables": tables,
            "instance.stack_deferred": stack,
            "instance.build.other": build_other,
        }
    )
    negative["instance.build.other"] = under

    cells_total = nanos(sample, "link.cells.total")
    handler = nanos(sample, "link.handler_selection")
    cell_transform, under = residual(cells_total, handler)
    exclusive["link.handler_selection"] = handler
    exclusive["link.cell_transform"] = cell_transform
    negative["link.cell_transform"] = under

    call_fixup = nanos(sample, "link.call_fixup")
    finalize = nanos(sample, "link.finalize")
    link_other, under = residual(
        nanos(sample, "link.total"), cells_total, call_fixup, finalize
    )
    exclusive.update(
        {
            "link.call_fixup": call_fixup,
            "link.finalize": finalize,
            "link.other": link_other,
        }
    )
    negative["link.other"] = under

    element = nanos(sample, "instance.element_segments")
    data = nanos(sample, "instance.data_segments")
    lease = nanos(sample, "instance.lease")
    drop_ns = nanos(sample, "drop")
    startup_other, under = residual(
        nanos(sample, "startup.total"),
        nanos(sample, "parser.total"),
        nanos(sample, "instance.build.total"),
        nanos(sample, "link.total"),
        element,
        data,
        lease,
        drop_ns,
    )
    exclusive.update(
        {
            "instance.element_segments": element,
            "instance.data_segments": data,
            "instance.lease": lease,
            "drop": drop_ns,
            "startup.other": startup_other,
        }
    )
    negative["startup.other"] = under
    return exclusive, {name: value for name, value in negative.items() if value}


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_case.setdefault(str(sample["case"]), []).append(sample)

    cases: dict[str, Any] = {}
    all_negative: list[dict[str, Any]] = []
    for case, rows in by_case.items():
        derived = []
        for row in rows:
            exclusive, negative = derive(row)
            total = nanos(row, "startup.total")
            derived.append((exclusive, total))
            if negative:
                all_negative.append(
                    {
                        "case": case,
                        "iteration": row["iteration"],
                        "residuals_ns": negative,
                    }
                )
        total_median = median(total for _, total in derived)
        stage_summary = {}
        for stage in EXCLUSIVE_ORDER:
            stage_ns = median(exclusive[stage] for exclusive, _ in derived)
            shares = [
                exclusive[stage] / total if total else 0.0
                for exclusive, total in derived
            ]
            stage_summary[stage] = {
                "median_ns": stage_ns,
                "median_share": median(shares),
            }

        raw_names = sorted(rows[0]["stages"])
        raw_summary = {
            name: {
                "median_ns": median(nanos(row, name) for row in rows),
                "median_calls": median(
                    int(row["stages"].get(name, {}).get("calls", 0))
                    for row in rows
                ),
            }
            for name in raw_names
        }
        cases[case] = {
            "sample_count": len(rows),
            "median_total_ns": total_median,
            "exclusive": stage_summary,
            "raw": raw_summary,
        }

    expected = {name for name, _ in CASES}
    if set(cases) != expected:
        raise ValueError(
            f"profile cases differ: got {sorted(cases)}, expected {sorted(expected)}"
        )

    aggregate: dict[str, Any] = {}
    sum_total = sum(case["median_total_ns"] for case in cases.values())
    for stage in EXCLUSIVE_ORDER:
        sum_stage = sum(
            case["exclusive"][stage]["median_ns"] for case in cases.values()
        )
        aggregate[stage] = {
            "equal_workload_share": statistics.fmean(
                case["exclusive"][stage]["median_share"]
                for case in cases.values()
            ),
            "time_weighted_share": sum_stage / sum_total if sum_total else 0.0,
            "sum_case_medians_ns": sum_stage,
        }

    ranked = sorted(
        aggregate,
        key=lambda stage: aggregate[stage]["equal_workload_share"],
        reverse=True,
    )
    return {
        "schema": 1,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git_sha": os.environ.get("GITHUB_SHA", ""),
        "cases": cases,
        "aggregate": aggregate,
        "ranked_stages": ranked,
        "negative_residuals": all_negative,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Interpreter eager-startup stage profile",
        "",
        f"- Platform: `{summary['platform']['machine']}`",
        f"- Revision: `{summary['git_sha'] or 'local-compile-only'}`",
        f"- Samples: median of {next(iter(summary['cases'].values()))['sample_count']} per workload",
        "",
        "## Largest exclusive stages",
        "",
        "| Stage | Equal-workload share | Time-weighted share |",
        "|---|---:|---:|",
    ]
    for stage in summary["ranked_stages"][:12]:
        item = summary["aggregate"][stage]
        lines.append(
            f"| `{stage}` | {item['equal_workload_share'] * 100:.2f}% | "
            f"{item['time_weighted_share'] * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Per-workload bottleneck",
            "",
            "| Workload | Median total | Largest stage | Share |",
            "|---|---:|---|---:|",
        ]
    )
    for name, case in summary["cases"].items():
        largest = max(
            EXCLUSIVE_ORDER,
            key=lambda stage: case["exclusive"][stage]["median_share"],
        )
        lines.append(
            f"| `{name}` | {case['median_total_ns'] / 1_000_000:.3f} ms | "
            f"`{largest}` | {case['exclusive'][largest]['median_share'] * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Parser sections",
            "",
            "| Section/stage | Equal-workload share |",
            "|---|---:|",
        ]
    )
    for stage in sorted(
        PARSER_CHILDREN,
        key=lambda item: summary["aggregate"][item]["equal_workload_share"],
        reverse=True,
    ):
        lines.append(
            f"| `{stage}` | "
            f"{summary['aggregate'][stage]['equal_workload_share'] * 100:.2f}% |"
        )

    if summary["negative_residuals"]:
        lines.extend(
            [
                "",
                "> [!WARNING]",
                "> Nested timers exceeded a parent in at least one raw sample. "
                "Residuals were clamped to zero; inspect `summary.json`.",
            ]
        )
    lines.extend(
        [
            "",
            "`predecode.lowering_control` is `predecode.total` minus decode, "
            "scratch, and incremental pinned-census timers. "
            "`link.cell_transform` is the linked-cell pass minus nested handler selection.",
            "",
        ]
    )
    return "\n".join(lines)


def run_driver(args: argparse.Namespace) -> list[dict[str, Any]]:
    suite = args.suite.resolve()
    cases = []
    for name, relative in CASES:
        path = suite / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned benchmark input: {path}")
        cases.append(f"{name}={path}")
    command = [
        str(args.binary.resolve()),
        str(args.iterations),
        str(args.warmups),
        *cases,
    ]
    result = subprocess.run(
        command,
        cwd=suite,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        check=True,
    )
    samples = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    expected = args.iterations * len(CASES)
    if len(samples) != expected:
        raise ValueError(f"driver emitted {len(samples)} samples, expected {expected}")
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmups < 0:
        parser.error("iterations must be positive and warmups non-negative")
    return args


def main() -> int:
    args = parse_args()
    samples = run_driver(args)
    summary = summarize(samples)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "samples.json").write_text(
        json.dumps({"schema": 1, "samples": samples}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    report = markdown(summary)
    (args.out_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
