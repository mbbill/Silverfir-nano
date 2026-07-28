#!/usr/bin/env python3
"""Compare two sf-nano-cli builds with a staged ABBA performance gate.

Both binaries run on the same machine against this checkout's benchmark
corpus. One ABBA block contains two adjacent baseline/candidate pairs:
baseline/candidate/candidate/baseline. The next block reverses that order.

The initial block produces two paired deltas for every metric. Initial
regressions and improvements are selected for a second ABBA round. A regression
that remains below the threshold receives a third and final ABBA round and
fails only if it remains below the threshold again. An improvement is reported
only if it remains above its threshold in the second round.

Only metrics selected by the initial screen can affect later classification.
New changes observed while rerunning a multi-metric benchmark are ignored.
Raw samples from every executed round are retained in the output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "wasi"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_tests  # noqa: E402
from bench_publish import METRIC_EXTRACTORS, extract_metrics, write_json  # noqa: E402


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive samples")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def command_for(binary: Path, runner_prefix: str, engine: str) -> tuple[str, list[str]]:
    binary = binary.resolve()
    if not binary.is_file():
        raise ValueError(f"runtime binary not found: {binary}")

    engine_args = ["--engine", engine]
    if not runner_prefix:
        return str(binary), engine_args

    prefix = shlex.split(runner_prefix, posix=os.name != "nt")
    if not prefix:
        raise ValueError("runner prefix is empty")
    runner = shutil.which(prefix[0])
    if runner is None:
        raise ValueError(f"runner not found on PATH: {prefix[0]}")
    return runner, prefix[1:] + [str(binary)] + engine_args


def run_once(
    *,
    version: str,
    command: tuple[str, list[str]],
    test: dict[str, Any],
    time_target: float,
    block: int,
    slot: int,
) -> dict[str, Any]:
    cli, cli_extra = command
    name, status, metric_str, elapsed = run_tests.run_test(
        cli,
        test,
        cli_extra,
        time_target=time_target,
    )
    entries = extract_metrics(name, metric_str) if status == "PASS" else []
    expected = len(METRIC_EXTRACTORS.get(name, []))
    if status != "PASS":
        raise RuntimeError(f"{version} {name}: {status}: {metric_str}")
    if len(entries) != expected:
        raise RuntimeError(
            f"{version} {name}: parsed {len(entries)}/{expected} metrics "
            f"from {metric_str!r}"
        )
    metrics = {
        entry["name"]: {
            "value": entry["value"],
            "unit": entry["unit"],
            "direction": entry["direction"],
        }
        for entry in entries
    }
    print(
        (
            "  warm-up: "
            if block < 0
            else f"  block {block + 1} slot {slot + 1}: "
        )
        + f"{version:<9} {elapsed or 0:.2f}s  {metric_str}",
        flush=True,
    )
    return {
        "block": block,
        "slot": slot,
        "version": version,
        "elapsed_seconds": elapsed,
        "metric_text": metric_str,
        "metrics": metrics,
    }


def pair_block(runs: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    versions = [run["version"] for run in runs]
    if versions == ["baseline", "candidate", "candidate", "baseline"]:
        return [(runs[0], runs[1]), (runs[3], runs[2])]
    if versions == ["candidate", "baseline", "baseline", "candidate"]:
        return [(runs[1], runs[0]), (runs[2], runs[3])]
    raise ValueError(f"unexpected ABBA schedule: {versions}")


def analyze_phase(
    *,
    test_name: str,
    runs: list[dict[str, Any]],
    blocks: Iterable[int],
) -> dict[str, dict[str, Any]]:
    block_indices = list(blocks)
    if not block_indices:
        raise ValueError(f"{test_name}: phase has no ABBA blocks")

    expected_names = [
        label for label, _regex, _unit, _direction
        in METRIC_EXTRACTORS[test_name]
    ]
    metrics: dict[str, Any] = {}

    for metric_name in expected_names:
        baseline_values: list[float] = []
        candidate_values: list[float] = []
        pair_ratios: list[float] = []
        unit = ""
        direction = ""

        for block in block_indices:
            block_runs = sorted(
                (run for run in runs if run["block"] == block),
                key=lambda run: run["slot"],
            )
            if len(block_runs) != 4:
                raise ValueError(
                    f"{test_name}: block {block + 1} has "
                    f"{len(block_runs)}/4 runs"
                )
            for baseline, candidate in pair_block(block_runs):
                base_metric = baseline["metrics"][metric_name]
                candidate_metric = candidate["metrics"][metric_name]
                if base_metric["unit"] != candidate_metric["unit"]:
                    raise ValueError(f"unit changed for {test_name}/{metric_name}")
                if base_metric["direction"] != candidate_metric["direction"]:
                    raise ValueError(
                        f"direction changed for {test_name}/{metric_name}"
                    )
                unit = base_metric["unit"]
                direction = base_metric["direction"]
                base_value = float(base_metric["value"])
                candidate_value = float(candidate_metric["value"])
                baseline_values.append(base_value)
                candidate_values.append(candidate_value)
                ratio = candidate_value / base_value
                if direction == "lower":
                    ratio = base_value / candidate_value
                pair_ratios.append(ratio)

        baseline_mean = geometric_mean(baseline_values)
        candidate_mean = geometric_mean(candidate_values)
        relative = geometric_mean(pair_ratios)
        delta_percent = (relative - 1.0) * 100.0
        pair_deltas = [(ratio - 1.0) * 100.0 for ratio in pair_ratios]

        metrics[metric_name] = {
            "unit": unit,
            "direction": direction,
            "baseline_samples": baseline_values,
            "candidate_samples": candidate_values,
            "baseline_geomean": baseline_mean,
            "candidate_geomean": candidate_mean,
            "pair_deltas_percent": pair_deltas,
            "delta_percent": delta_percent,
        }

    return metrics


def regression_candidates(
    metrics: dict[str, dict[str, Any]],
    threshold_percent: float,
) -> set[str]:
    return {
        name
        for name, metric in metrics.items()
        if metric["delta_percent"] < -threshold_percent
    }


def improvement_candidates(
    metrics: dict[str, dict[str, Any]],
    threshold_percent: float,
) -> set[str]:
    return {
        name
        for name, metric in metrics.items()
        if metric["delta_percent"] > threshold_percent
    }


def confirmation_candidates(
    metrics: dict[str, dict[str, Any]],
    regression_threshold: float,
    improvement_threshold: float,
) -> set[str]:
    return regression_candidates(
        metrics, regression_threshold
    ) | improvement_candidates(metrics, improvement_threshold)


def third_round_candidates(
    initial: dict[str, dict[str, Any]],
    confirmation: dict[str, dict[str, Any]],
    regression_threshold: float,
) -> set[str]:
    selected = regression_candidates(initial, regression_threshold)
    missing = selected - confirmation.keys()
    if missing:
        raise ValueError(
            f"{sorted(missing)[0]}: selected metric has no second round"
        )
    return {
        name
        for name in selected
        if confirmation[name]["delta_percent"] < -regression_threshold
    }


def classify_metrics(
    *,
    initial: dict[str, dict[str, Any]],
    confirmation: dict[str, dict[str, Any]],
    third_round: dict[str, dict[str, Any]],
    regression_threshold: float,
    improvement_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Apply the staged gate to candidates frozen by the initial screen.

    `confirmation` contains initial regressions and improvements. `third_round`
    contains only initial regressions that were still negative in confirmation.
    Other measurements produced by multi-metric benchmarks are omitted.
    """
    regressions = regression_candidates(initial, regression_threshold)
    improvements = improvement_candidates(initial, improvement_threshold)
    selected = regressions | improvements
    metrics: dict[str, dict[str, Any]] = {}

    for name, initial_metric in initial.items():
        confirmation_metric = confirmation.get(name) if name in selected else None
        third_metric = None

        if name in selected:
            if confirmation_metric is None:
                raise ValueError(f"{name}: selected metric has no second round")

        if name in regressions:
            if confirmation_metric["delta_percent"] >= -regression_threshold:
                status = "RECOVERED"
            else:
                third_metric = third_round.get(name)
                if third_metric is None:
                    raise ValueError(
                        f"{name}: persistent metric has no third round"
                    )
                status = (
                    "REGRESSION"
                    if third_metric["delta_percent"] < -regression_threshold
                    else "RECOVERED"
                )
        elif name in improvements:
            status = (
                "IMPROVEMENT"
                if confirmation_metric["delta_percent"] > improvement_threshold
                else "PASS"
            )
        else:
            status = "PASS"

        metrics[name] = {
            **initial_metric,
            "confirmation": confirmation_metric,
            "third_round": third_metric,
            "regression_threshold_percent": regression_threshold,
            "improvement_threshold_percent": improvement_threshold,
            "status": status,
        }

    return metrics


def format_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def render_summary(
    *,
    platform: str,
    engine: str,
    baseline_sha: str,
    candidate_sha: str,
    regression_threshold: float,
    improvement_threshold: float,
    results: dict[str, Any],
) -> str:
    lines = [
        f"## Performance: {platform} / {engine}",
        "",
        f"`{baseline_sha[:12]}` -> `{candidate_sha[:12]}`",
        "",
        (
            f"> Gate: initial delta below `-{regression_threshold:.2f}%` "
            "gets a second ABBA round. If it remains below the threshold, "
            "a third round decides REGRESSION versus RECOVERED. Initial "
            f"improvements above `+{improvement_threshold:.2f}%` are reported "
            "only when the second round also exceeds that threshold."
        ),
        "",
        (
            "| Metric | Baseline | Candidate | "
            "Initial delta (pairs) | Second delta (pairs) | "
            "Third delta (pairs) | Gate |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for test in results.values():
        for name, metric in test["metrics"].items():
            initial_pairs = ", ".join(
                f"{delta:+.2f}%"
                for delta in metric["pair_deltas_percent"]
            )
            initial = (
                f"{metric['delta_percent']:+.2f}% "
                f"({initial_pairs})"
            )
            confirmation_metric = metric["confirmation"]
            confirmation = "—"
            if confirmation_metric is not None:
                confirmation_pairs = ", ".join(
                    f"{delta:+.2f}%"
                    for delta in confirmation_metric["pair_deltas_percent"]
                )
                confirmation = (
                    f"{confirmation_metric['delta_percent']:+.2f}% "
                    f"({confirmation_pairs})"
                )
            third_metric = metric["third_round"]
            third = "—"
            if third_metric is not None:
                third_pairs = ", ".join(
                    f"{delta:+.2f}%"
                    for delta in third_metric["pair_deltas_percent"]
                )
                third = (
                    f"{third_metric['delta_percent']:+.2f}% "
                    f"({third_pairs})"
                )
            gate = metric["status"]
            if gate in {"REGRESSION", "IMPROVEMENT"}:
                gate = f"**{gate}**"
            lines.append(
                "| "
                f"{name} | "
                f"{format_number(metric['baseline_geomean'])} | "
                f"{format_number(metric['candidate_geomean'])} | "
                f"{initial} | {confirmation} | {third} | {gate} |"
            )
    lines.append("")
    lines.append(
        "> Later-round measurements from metrics that passed the initial "
        "screen are ignored by design. Improvement candidates stop after "
        "the second round."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-exec", required=True, type=Path)
    parser.add_argument("--candidate-exec", required=True, type=Path)
    parser.add_argument(
        "--runner-prefix",
        default="",
        help="Optional emulator command placed before each runtime binary",
    )
    parser.add_argument("--engine", choices=("jit", "interp"), required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--baseline-sha", default="")
    parser.add_argument("--candidate-sha", default="")
    parser.add_argument("--time", type=float, default=run_tests.DEFAULT_TARGET)
    parser.add_argument("--warmup-time", type=float, default=0.5)
    parser.add_argument(
        "--blocks",
        type=int,
        default=1,
        help="Initial ABBA blocks per benchmark (one block is two pairs)",
    )
    parser.add_argument(
        "--confirmation-blocks",
        type=int,
        default=1,
        help="ABBA blocks in each follow-up round (default: 1)",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=1.0,
        metavar="PERCENT",
        help="Initial and confirmation regression threshold (default: 1.0)",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=3.0,
        metavar="PERCENT",
        help="Initial and confirmation improvement threshold (default: 3.0)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="TEST_OR_METRIC",
        help="Run only a named test or metric; may be repeated",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="TEST_OR_METRIC",
        help=(
            "Skip a named test, or the test containing a named metric; "
            "may be repeated"
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.time <= 0:
        parser.error("--time must be positive")
    if args.warmup_time < 0:
        parser.error("--warmup-time cannot be negative")
    if args.blocks <= 0:
        parser.error("--blocks must be positive")
    if args.confirmation_blocks <= 0:
        parser.error("--confirmation-blocks must be positive")
    if args.regression_threshold <= 0:
        parser.error("--regression-threshold must be positive")
    if args.improvement_threshold <= 0:
        parser.error("--improvement-threshold must be positive")

    selected_tests = []
    selectors = set(args.only)
    exclusions = set(args.exclude)
    matched_exclusions: set[str] = set()
    for test in run_tests.TESTS:
        metric_names = {
            label
            for label, _regex, _unit, _direction
            in METRIC_EXTRACTORS.get(test["name"], [])
        }
        test_selectors = {test["name"]} | metric_names
        matched_exclusions.update(exclusions & test_selectors)
        if exclusions & test_selectors:
            continue
        if not selectors or selectors & test_selectors:
            selected_tests.append(test)
    if selectors and not selected_tests:
        parser.error(
            "--only did not select a test after exclusions: "
            + ", ".join(sorted(selectors))
        )
    unknown_exclusions = exclusions - matched_exclusions
    if unknown_exclusions:
        parser.error(
            "--exclude did not match a test or metric: "
            + ", ".join(sorted(unknown_exclusions))
        )

    try:
        baseline_command = command_for(
            args.baseline_exec, args.runner_prefix, args.engine
        )
        candidate_command = command_for(
            args.candidate_exec, args.runner_prefix, args.engine
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.warmup_time:
        print(
            f"Warm-up: coremark, {args.warmup_time}s per binary",
            flush=True,
        )
        warm_test = run_tests.TESTS[0]
        for version, command in (
            ("baseline", baseline_command),
            ("candidate", candidate_command),
        ):
            run_once(
                version=version,
                command=command,
                test=warm_test,
                time_target=args.warmup_time,
                block=-1,
                slot=0,
            )

    results: dict[str, Any] = {}
    runs_by_test: dict[str, list[dict[str, Any]]] = {}
    initial_by_test: dict[str, dict[str, dict[str, Any]]] = {}
    confirmation_by_test: dict[str, dict[str, dict[str, Any]]] = {}
    third_by_test: dict[str, dict[str, dict[str, Any]]] = {}

    try:
        def add_block(
            test: dict[str, Any],
            runs: list[dict[str, Any]],
            block: int,
        ) -> None:
            schedule = (
                ["baseline", "candidate", "candidate", "baseline"]
                if block % 2 == 0
                else ["candidate", "baseline", "baseline", "candidate"]
            )
            for slot, version in enumerate(schedule):
                command = (
                    baseline_command
                    if version == "baseline"
                    else candidate_command
                )
                runs.append(
                    run_once(
                        version=version,
                        command=command,
                        test=test,
                        time_target=args.time,
                        block=block,
                        slot=slot,
                    )
                )

        # Stage 1: every selected benchmark gets the initial ABBA blocks.
        for index, test in enumerate(selected_tests, 1):
            name = test["name"]
            print(
                f"[initial {index}/{len(selected_tests)}] {name}",
                flush=True,
            )
            runs: list[dict[str, Any]] = []
            for block in range(args.blocks):
                add_block(test, runs, block)
            runs_by_test[name] = runs
            initial_by_test[name] = analyze_phase(
                test_name=name,
                runs=runs,
                blocks=range(args.blocks),
            )

        confirmation_targets = [
            test
            for test in selected_tests
            if confirmation_candidates(
                initial_by_test[test["name"]],
                args.regression_threshold,
                args.improvement_threshold,
            )
        ]

        # Stage 2: rerun benchmarks containing an initial regression or
        # improvement. Other metrics measured as a side effect are ignored.
        for index, test in enumerate(confirmation_targets, 1):
            name = test["name"]
            selected_metrics = confirmation_candidates(
                initial_by_test[name],
                args.regression_threshold,
                args.improvement_threshold,
            )
            print(
                (
                    f"[second {index}/{len(confirmation_targets)}] "
                    f"{name}: {', '.join(sorted(selected_metrics))}"
                ),
                flush=True,
            )
            runs = runs_by_test[name]
            confirmation_range = range(
                args.blocks,
                args.blocks + args.confirmation_blocks,
            )
            for block in confirmation_range:
                add_block(test, runs, block)
            all_confirmation_metrics = analyze_phase(
                test_name=name,
                runs=runs,
                blocks=confirmation_range,
            )
            confirmation_by_test[name] = {
                metric_name: all_confirmation_metrics[metric_name]
                for metric_name in selected_metrics
            }

        third_targets = [
            test
            for test in confirmation_targets
            if third_round_candidates(
                initial_by_test[test["name"]],
                confirmation_by_test[test["name"]],
                args.regression_threshold,
            )
        ]

        # Stage 3: only initial regressions that remained below threshold in
        # stage 2 can reach this round. Improvements never reach stage 3.
        for index, test in enumerate(third_targets, 1):
            name = test["name"]
            selected_metrics = third_round_candidates(
                initial_by_test[name],
                confirmation_by_test[name],
                args.regression_threshold,
            )
            print(
                (
                    f"[third {index}/{len(third_targets)}] "
                    f"{name}: {', '.join(sorted(selected_metrics))}"
                ),
                flush=True,
            )
            runs = runs_by_test[name]
            third_range = range(
                args.blocks + args.confirmation_blocks,
                args.blocks + 2 * args.confirmation_blocks,
            )
            for block in third_range:
                add_block(test, runs, block)
            all_third_metrics = analyze_phase(
                test_name=name,
                runs=runs,
                blocks=third_range,
            )
            third_by_test[name] = {
                metric_name: all_third_metrics[metric_name]
                for metric_name in selected_metrics
            }

        for test in selected_tests:
            name = test["name"]
            second_metrics = confirmation_candidates(
                initial_by_test[name],
                args.regression_threshold,
                args.improvement_threshold,
            )
            third_metrics = third_round_candidates(
                initial_by_test[name],
                confirmation_by_test.get(name, {}),
                args.regression_threshold,
            )
            metrics = classify_metrics(
                initial=initial_by_test[name],
                confirmation=confirmation_by_test.get(name, {}),
                third_round=third_by_test.get(name, {}),
                regression_threshold=args.regression_threshold,
                improvement_threshold=args.improvement_threshold,
            )
            results[name] = {
                "runs": runs_by_test[name],
                "initial_blocks": args.blocks,
                "confirmation_blocks": (
                    args.confirmation_blocks if second_metrics else 0
                ),
                "third_blocks": (
                    args.confirmation_blocks if third_metrics else 0
                ),
                "confirmed_metrics": sorted(second_metrics),
                "third_round_metrics": sorted(third_metrics),
                "metrics": metrics,
            }
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    document = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform,
        "engine": args.engine,
        "baseline_sha": args.baseline_sha,
        "candidate_sha": args.candidate_sha,
        "target_seconds": args.time,
        "initial_blocks": args.blocks,
        "confirmation_blocks": args.confirmation_blocks,
        "regression_threshold_percent": args.regression_threshold,
        "improvement_threshold_percent": args.improvement_threshold,
        "excluded_selectors": sorted(exclusions),
        "tests": results,
    }
    with (args.out_dir / "comparison.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        output.write(json.dumps(document, indent=2) + "\n")

    candidate_entries = []
    for test in results.values():
        for name, metric in test["metrics"].items():
            candidate_entries.append({
                "name": name,
                "unit": metric["unit"],
                "value": metric["candidate_geomean"],
                "range": max(metric["candidate_samples"])
                - min(metric["candidate_samples"]),
            })
    write_json(args.out_dir / "candidate.json", candidate_entries)

    summary = render_summary(
        platform=args.platform,
        engine=args.engine,
        baseline_sha=args.baseline_sha,
        candidate_sha=args.candidate_sha,
        regression_threshold=args.regression_threshold,
        improvement_threshold=args.improvement_threshold,
        results=results,
    )
    if exclusions:
        summary += (
            "\n> Excluded on this platform: "
            + ", ".join(f"`{name}`" for name in sorted(exclusions))
            + "\n"
        )
    (args.out_dir / "summary.md").write_text(
        summary, encoding="utf-8", newline="\n"
    )
    print()
    print(summary)

    statuses = [
        metric["status"]
        for test in results.values()
        for metric in test["metrics"].values()
    ]
    return 1 if "REGRESSION" in statuses else 0


if __name__ == "__main__":
    sys.exit(main())
