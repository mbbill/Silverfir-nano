#!/usr/bin/env python3
"""Compare two sf-nano-cli builds with per-benchmark ABBA measurements.

Both binaries run on the same machine, against this checkout's benchmark
corpus. Each block is ordered baseline/candidate/candidate/baseline (the next
block reverses that order), which cancels a roughly linear frequency or thermal
drift. Raw samples are always retained; thresholds are optional because they
must come from same-binary calibration runs rather than guesses.

Threshold file schema:

  {
    "platforms": {
      "arm64-linux": {
        "jit": {
          "coremark": 2.0,
          "sha256": 2.5
        }
      }
    }
  }

Values are allowed regression percentages. Metrics without a calibrated value
are report-only.
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
from typing import Any

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


def threshold_for(
    config: dict[str, Any],
    platform: str,
    engine: str,
    metric: str,
) -> float | None:
    value = (
        config.get("platforms", {})
        .get(platform, {})
        .get(engine, {})
        .get(metric)
    )
    if value is None:
        return None
    threshold = float(value)
    if threshold <= 0:
        raise ValueError(
            f"threshold for {platform}/{engine}/{metric} must be positive"
        )
    return threshold


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


def analyze_test(
    *,
    test_name: str,
    runs: list[dict[str, Any]],
    blocks: int,
    thresholds: dict[str, Any],
    platform: str,
    engine: str,
) -> dict[str, Any]:
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

        for block in range(blocks):
            block_runs = [run for run in runs if run["block"] == block]
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
        threshold = threshold_for(thresholds, platform, engine, metric_name)

        status = "REPORT"
        if threshold is not None:
            below = [
                delta <= -threshold
                for delta in pair_deltas
            ]
            if delta_percent <= -threshold and all(below):
                status = "REGRESSION"
            elif delta_percent <= -threshold or any(below):
                status = "UNSTABLE"
            else:
                status = "PASS"

        metrics[metric_name] = {
            "unit": unit,
            "direction": direction,
            "baseline_samples": baseline_values,
            "candidate_samples": candidate_values,
            "baseline_geomean": baseline_mean,
            "candidate_geomean": candidate_mean,
            "pair_deltas_percent": pair_deltas,
            "delta_percent": delta_percent,
            "threshold_percent": threshold,
            "status": status,
        }

    return {"runs": runs, "metrics": metrics}


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
    results: dict[str, Any],
) -> str:
    lines = [
        f"## Performance: {platform} / {engine}",
        "",
        f"`{baseline_sha[:12]}` -> `{candidate_sha[:12]}`",
        "",
        "| Metric | Baseline | Candidate | Delta | Paired deltas | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for test in results.values():
        for name, metric in test["metrics"].items():
            pairs = ", ".join(
                f"{delta:+.2f}%"
                for delta in metric["pair_deltas_percent"]
            )
            gate = metric["status"]
            if metric["threshold_percent"] is not None:
                gate += f" ({metric['threshold_percent']:.2f}%)"
            lines.append(
                "| "
                f"{name} | "
                f"{format_number(metric['baseline_geomean'])} | "
                f"{format_number(metric['candidate_geomean'])} | "
                f"{metric['delta_percent']:+.2f}% | "
                f"{pairs} | {gate} |"
            )
    lines.append("")
    if all(
        metric["threshold_percent"] is None
        for test in results.values()
        for metric in test["metrics"].values()
    ):
        lines.append(
            "> Report-only: no calibrated threshold exists for this "
            "platform/engine yet."
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
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=3,
        help="Add blocks up to this limit when calibrated pairs disagree",
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
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--fail-on-unstable", action="store_true")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.time <= 0:
        parser.error("--time must be positive")
    if args.warmup_time < 0:
        parser.error("--warmup-time cannot be negative")
    if args.blocks <= 0:
        parser.error("--blocks must be positive")
    if args.max_blocks < args.blocks:
        parser.error("--max-blocks cannot be less than --blocks")

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
        thresholds = {}
        if args.thresholds:
            thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
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
    try:
        for index, test in enumerate(selected_tests, 1):
            name = test["name"]
            print(f"[{index}/{len(selected_tests)}] {name}", flush=True)
            runs: list[dict[str, Any]] = []
            block_count = 0

            def add_block(block: int) -> None:
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

            while block_count < args.blocks:
                add_block(block_count)
                block_count += 1

            while True:
                analysis = analyze_test(
                    test_name=name,
                    runs=runs,
                    blocks=block_count,
                    thresholds=thresholds,
                    platform=args.platform,
                    engine=args.engine,
                )
                unstable = any(
                    metric["status"] == "UNSTABLE"
                    for metric in analysis["metrics"].values()
                )
                if not unstable or block_count >= args.max_blocks:
                    break
                print(
                    f"  ambiguous calibrated result; adding ABBA block "
                    f"{block_count + 1}/{args.max_blocks}",
                    flush=True,
                )
                add_block(block_count)
                block_count += 1
            results[name] = analysis
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform,
        "engine": args.engine,
        "baseline_sha": args.baseline_sha,
        "candidate_sha": args.candidate_sha,
        "target_seconds": args.time,
        "initial_blocks": args.blocks,
        "max_blocks": args.max_blocks,
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
    if args.fail_on_regression and "REGRESSION" in statuses:
        return 1
    if args.fail_on_unstable and "UNSTABLE" in statuses:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
