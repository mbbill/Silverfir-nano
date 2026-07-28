#!/usr/bin/env python3
"""Compare two sf-nano-cli builds with paired, probability-based sampling.

Both binaries run on the same machine against this checkout's benchmark
corpus. Alternating blocks contain two adjacent A/B pairs and reverse order
between blocks:

    baseline, candidate, baseline, candidate
    candidate, baseline, candidate, baseline

The initial blocks run every selected benchmark. For each metric, paired
candidate/baseline ratios are converted to log deltas. Their mean, sample
volatility, and Student-t posterior determine how many total pairs would be
needed to reach the requested regression or improvement probability.

Only benchmarks whose initial directional probability crosses the pilot screen
are rerun. The initial sample is a pilot and is not reused by the final gate:
probability is calculated from a new, independent confirmation sample.
Confirmation starts at the minimum pair count. If it has not crossed the gate
but its measured effect and variance can still resolve within the budget, more
pairs are added and the requirement is estimated again. This sample split
avoids selection bias from choosing a direction on the same data used for the
decision, while re-estimation avoids trusting one noisy pilot variance estimate.
The final one-sided probability thresholds use a Bonferroni correction across
all metrics, parallel performance jobs, and possible adaptive confirmation
looks so the requested probability controls the whole benchmark family rather
than being reapplied independently at every opportunity.

Metrics that were not selected by the initial screen cannot become regressions
or improvements merely because their benchmark was rerun for another metric.
"""

from __future__ import annotations

import argparse
import hashlib
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

import run_tests  # noqa: E402
from ci.bench_metrics import (  # noqa: E402
    METRIC_EXTRACTORS,
    extract_metrics,
    write_json,
)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive samples")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by regularized incomplete beta."""
    max_iterations = 200
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for iteration in range(1, max_iterations + 1):
        even = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x
            / ((qam + even) * (a + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration) * (qab + iteration) * x
            / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result

    raise ArithmeticError("incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError("beta parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - (
        factor * _beta_continued_fraction(b, a, 1.0 - x) / b
    )


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    """CDF of Student's t without a scipy dependency."""
    if degrees_of_freedom <= 0:
        raise ValueError("degrees of freedom must be positive")
    if value == 0.0:
        return 0.5
    if math.isinf(value):
        return 1.0 if value > 0.0 else 0.0

    df = float(degrees_of_freedom)
    x = df / (df + value * value)
    tail = 0.5 * regularized_incomplete_beta(x, df / 2.0, 0.5)
    return 1.0 - tail if value > 0.0 else tail


def probability_summary(pair_ratios: list[float]) -> dict[str, Any]:
    if len(pair_ratios) < 2:
        raise ValueError("probability analysis requires at least two pairs")
    if any(ratio <= 0.0 for ratio in pair_ratios):
        raise ValueError("paired ratios must be positive")

    log_deltas = [math.log(ratio) for ratio in pair_ratios]
    pair_count = len(log_deltas)
    mean_log = math.fsum(log_deltas) / pair_count
    squared = math.fsum(
        (value - mean_log) ** 2 for value in log_deltas
    )
    stddev_log = math.sqrt(squared / (pair_count - 1))
    standard_error = stddev_log / math.sqrt(pair_count)

    if standard_error == 0.0:
        if mean_log < 0.0:
            probability_regression = 1.0
        elif mean_log > 0.0:
            probability_regression = 0.0
        else:
            probability_regression = 0.5
    else:
        zero_t = -mean_log / standard_error
        probability_regression = student_t_cdf(
            zero_t, pair_count - 1
        )

    return {
        "pair_count": pair_count,
        "log_deltas": log_deltas,
        "mean_log_ratio": mean_log,
        "log_ratio_stddev": stddev_log,
        "standard_error": standard_error,
        "probability_regression": probability_regression,
        "probability_improvement": 1.0 - probability_regression,
        "delta_percent": math.expm1(mean_log) * 100.0,
        "volatility_percent": math.expm1(stddev_log) * 100.0,
        "pair_deltas_percent": [
            math.expm1(value) * 100.0 for value in log_deltas
        ],
    }


def direction_probability_for_pairs(
    *,
    mean_log_ratio: float,
    log_ratio_stddev: float,
    pair_count: int,
) -> float:
    if pair_count < 2:
        raise ValueError("pair count must be at least two")
    magnitude = abs(mean_log_ratio)
    if magnitude == 0.0:
        return 0.5
    if log_ratio_stddev == 0.0:
        return 1.0
    t_value = magnitude / (log_ratio_stddev / math.sqrt(pair_count))
    return student_t_cdf(t_value, pair_count - 1)


def family_adjusted_probability(
    probability: float,
    *,
    metric_count: int,
    job_count: int,
    maximum_looks: int,
) -> float:
    """Convert a family-wide probability into a per-look threshold."""
    if not 0.5 < probability < 1.0:
        raise ValueError("probability must be between 0.5 and 1")
    if metric_count <= 0:
        raise ValueError("metric count must be positive")
    if job_count <= 0:
        raise ValueError("job count must be positive")
    if maximum_looks <= 0:
        raise ValueError("maximum looks must be positive")
    hypotheses = metric_count * job_count * maximum_looks
    return 1.0 - (1.0 - probability) / hypotheses


def required_pairs(
    metric: dict[str, Any],
    *,
    probability: float,
    minimum_pairs: int,
    maximum_pairs: int,
) -> int | None:
    """Predict a fixed final pair count from the initial effect and variance."""
    if not 0.5 < probability < 1.0:
        raise ValueError("probability must be between 0.5 and 1")
    current_pairs = int(metric["pair_count"])
    start = max(current_pairs, minimum_pairs)
    if start % 2:
        start += 1

    for pair_count in range(start, maximum_pairs + 1, 2):
        observed_probability = direction_probability_for_pairs(
            mean_log_ratio=float(metric["mean_log_ratio"]),
            log_ratio_stddev=float(metric["log_ratio_stddev"]),
            pair_count=pair_count,
        )
        if observed_probability >= probability:
            return pair_count
    return None


def required_pairs_for_direction(
    metric: dict[str, Any],
    *,
    direction: str,
    probability: float,
    minimum_pairs: int,
    maximum_pairs: int,
) -> int | None:
    """Predict pairs for a direction frozen before this sample was taken."""
    mean_log = float(metric["mean_log_ratio"])
    if direction == "regression":
        directed_mean = -mean_log
    elif direction == "improvement":
        directed_mean = mean_log
    else:
        raise ValueError(f"unknown candidate direction: {direction}")

    if directed_mean <= 0.0:
        return None
    return required_pairs(
        metric,
        probability=probability,
        minimum_pairs=minimum_pairs,
        maximum_pairs=maximum_pairs,
    )


def metric_plans(
    metrics: dict[str, dict[str, Any]],
    *,
    regression_probability: float,
    improvement_probability: float,
    minimum_pairs: int,
    maximum_pairs: int,
    pilot_probability: float = 0.8,
) -> dict[str, dict[str, Any]]:
    if not 0.5 < pilot_probability < 1.0:
        raise ValueError("pilot probability must be between 0.5 and 1")
    plans: dict[str, dict[str, Any]] = {}
    for name, metric in metrics.items():
        mean_log = float(metric["mean_log_ratio"])
        if mean_log < 0.0:
            direction = "regression"
            target_probability = regression_probability
            pilot_direction_probability = float(
                metric["probability_regression"]
            )
        elif mean_log > 0.0:
            direction = "improvement"
            target_probability = improvement_probability
            pilot_direction_probability = float(
                metric["probability_improvement"]
            )
        else:
            direction = None
            target_probability = None
            pilot_direction_probability = 0.5

        target_pairs = None
        if direction is not None:
            target_pairs = required_pairs(
                metric,
                probability=float(target_probability),
                minimum_pairs=minimum_pairs,
                maximum_pairs=maximum_pairs,
            )
        plans[name] = {
            "direction": direction,
            "target_probability": target_probability,
            "target_pairs": target_pairs,
            "pilot_direction_probability": pilot_direction_probability,
            "selected": (
                direction is not None
                and pilot_direction_probability >= pilot_probability
            ),
        }
    return plans


def classify_metrics(
    *,
    initial: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
    plans: dict[str, dict[str, Any]],
    regression_probability: float,
    improvement_probability: float,
    identical_binaries: bool = False,
) -> dict[str, dict[str, Any]]:
    """Classify only directions frozen by the initial screening stage."""
    metrics: dict[str, dict[str, Any]] = {}
    for name, initial_metric in initial.items():
        plan = plans[name]
        selected = bool(plan["selected"])
        measured = final.get(name) if selected else initial_metric
        if measured is None:
            raise ValueError(f"{name}: selected metric has no final sample")

        direction = plan["direction"]
        if direction == "regression" and selected:
            status = (
                "REGRESSION"
                if measured["probability_regression"]
                >= regression_probability
                else "RECOVERED"
            )
        elif direction == "improvement" and selected:
            status = (
                "IMPROVEMENT"
                if measured["probability_improvement"]
                >= improvement_probability
                else "PASS"
            )
        else:
            status = "PASS"

        # Byte-identical executables cannot contain a source performance
        # change. Keep an apparent regression visible as runner drift, but
        # never let it fail the gate or claim a source improvement.
        if identical_binaries:
            if status == "REGRESSION":
                status = "UNSTABLE"
            elif status == "IMPROVEMENT":
                status = "PASS"

        metrics[name] = {
            **measured,
            "initial": initial_metric,
            "candidate_direction": direction,
            "pilot_direction_probability": plan[
                "pilot_direction_probability"
            ],
            "selected": selected,
            "target_pairs": plan["target_pairs"],
            "status": status,
        }
    return metrics


def load_build_metadata(
    path: Path | None,
    *,
    platform: str,
    engine: str,
    baseline_sha: str,
    candidate_sha: str,
    baseline_exec: Path,
    candidate_exec: Path,
) -> bool:
    """Validate build provenance and return whether the binaries are identical."""
    if path is None:
        return False

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("document is not an object")
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported schema_version")
        if metadata.get("platform") != platform:
            raise ValueError("platform does not match")
        if metadata.get("engine") != engine:
            raise ValueError("engine does not match")
        builds = metadata.get("builds")
        if not isinstance(builds, dict):
            raise ValueError("builds is not an object")

        observed_hashes = {}
        for label, revision, executable in (
            ("baseline", baseline_sha, baseline_exec),
            ("candidate", candidate_sha, candidate_exec),
        ):
            build = builds.get(label)
            if not isinstance(build, dict):
                raise ValueError(f"{label} build is not an object")
            if build.get("revision") != revision:
                raise ValueError(f"{label} revision does not match")
            executable = executable.resolve()
            size = executable.stat().st_size
            if build.get("size") != size:
                raise ValueError(f"{label} executable size does not match")
            with executable.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if build.get("sha256") != digest:
                raise ValueError(f"{label} executable sha256 does not match")
            observed_hashes[label] = digest

        identical_binaries = metadata.get("identical_binaries")
        if not isinstance(identical_binaries, bool):
            raise ValueError("identical_binaries is not boolean")
        observed_identical = (
            observed_hashes["baseline"] == observed_hashes["candidate"]
        )
        if identical_binaries != observed_identical:
            raise ValueError("identical_binaries does not match executables")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid build metadata: {exc}") from exc

    return identical_binaries


def command_for(
    binary: Path,
    runner_prefix: str,
    engine: str,
) -> tuple[str, list[str]]:
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


def run_correctness_suite(
    *,
    baseline_command: tuple[str, list[str]],
    candidate_command: tuple[str, list[str]],
    selected_tests: list[dict[str, Any]],
    time_target: float,
    platform: str,
    engine: str,
    baseline_sha: str,
    candidate_sha: str,
    out_dir: Path,
) -> int:
    """Run every benchmark once on A and B and validate its fixed oracle."""
    results: dict[str, Any] = {}
    failed = False
    commands = {
        "baseline": baseline_command,
        "candidate": candidate_command,
    }

    for index, test in enumerate(selected_tests, 1):
        name = test["name"]
        print(f"[correctness {index}/{len(selected_tests)}] {name}", flush=True)
        test_results: dict[str, Any] = {}
        for version in ("baseline", "candidate"):
            cli, cli_extra = commands[version]
            try:
                _, status, metric, elapsed = run_tests.run_test(
                    cli,
                    test,
                    cli_extra,
                    time_target=time_target,
                    correctness_only=True,
                )
            except Exception as exc:
                status = "FAIL"
                metric = str(exc)
                elapsed = None
            if status != "PASS":
                failed = True
            test_results[version] = {
                "status": status,
                "metric": metric,
                "elapsed_seconds": elapsed,
            }
            print(
                f"  {version}: {status} "
                f"({elapsed:.3f}s) {metric}"
                if elapsed is not None
                else f"  {version}: {status} {metric}",
                flush=True,
            )
        results[name] = test_results

    document = {
        "schema_version": 1,
        "mode": "correctness",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform,
        "engine": engine,
        "baseline_sha": baseline_sha,
        "candidate_sha": candidate_sha,
        "target_seconds": time_target,
        "tests": results,
    }
    (out_dir / "correctness.json").write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        f"## Benchmark correctness: {platform} / {engine}",
        "",
        f"`{baseline_sha or 'baseline'}` -> "
        f"`{candidate_sha or 'candidate'}`",
        "",
        "| Benchmark | Baseline | Candidate |",
        "| --- | --- | --- |",
    ]
    for name, test_results in results.items():
        lines.append(
            f"| {name} | "
            f"{test_results['baseline']['status']} | "
            f"{test_results['candidate']['status']} |"
        )
    lines.extend([
        "",
        "> QEMU jobs validate benchmark outputs only. Performance deltas are "
        "intentionally not calculated or gated.",
        "",
    ])
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(
        summary, encoding="utf-8", newline="\n"
    )
    print()
    print(summary)
    return 1 if failed else 0


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


def pair_block(
    runs: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    versions = [run["version"] for run in runs]
    if versions == ["baseline", "candidate", "baseline", "candidate"]:
        return [(runs[0], runs[1]), (runs[2], runs[3])]
    if versions == ["candidate", "baseline", "candidate", "baseline"]:
        return [(runs[1], runs[0]), (runs[3], runs[2])]
    raise ValueError(f"unexpected alternating schedule: {versions}")


def analyze_phase(
    *,
    test_name: str,
    runs: list[dict[str, Any]],
    blocks: Iterable[int],
) -> dict[str, dict[str, Any]]:
    block_indices = list(blocks)
    if not block_indices:
        raise ValueError(f"{test_name}: phase has no alternating blocks")

    expected_names = [
        label
        for label, _regex, _unit, _direction
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
                    raise ValueError(
                        f"unit changed for {test_name}/{metric_name}"
                    )
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

        statistics = probability_summary(pair_ratios)
        metrics[metric_name] = {
            "unit": unit,
            "direction": direction,
            "baseline_samples": baseline_values,
            "candidate_samples": candidate_values,
            "baseline_geomean": geometric_mean(baseline_values),
            "candidate_geomean": geometric_mean(candidate_values),
            **statistics,
        }

    return metrics


def format_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def format_probability(value: float) -> str:
    if value >= 0.999999995:
        return ">99.999999%"
    if value <= 0.000000005:
        return "<0.000001%"
    return f"{value * 100:.6f}%"


def render_summary(
    *,
    platform: str,
    engine: str,
    baseline_sha: str,
    candidate_sha: str,
    initial_pairs: int,
    minimum_pairs: int,
    maximum_pairs: int,
    pilot_probability: float,
    regression_probability: float,
    improvement_probability: float,
    effective_regression_probability: float,
    effective_improvement_probability: float,
    family_metric_count: int,
    family_job_count: int,
    maximum_looks: int,
    identical_binaries: bool,
    results: dict[str, Any],
) -> str:
    lines = [
        f"## Performance: {platform} / {engine}",
        "",
        f"`{baseline_sha[:12]}` -> `{candidate_sha[:12]}`",
        "",
    ]
    if identical_binaries:
        lines.extend([
            (
                "> **Byte-identical executables:** measured differences are "
                "runner drift. They remain visible, but cannot fail the gate "
                "or claim an improvement."
            ),
            "",
        ])
    lines.extend([
        (
            "> Timing: the requested duration applies to every benchmark. "
            "CoreMark uses its explicit non-standard regression mode here; "
            "a bare CoreMark invocation retains the official EEMBC "
            "10-second-minimum run."
        ),
        "",
        (
            f"> Probability gate: start with `{initial_pairs}` paired samples. "
            f"A direction with at least "
            f"`{pilot_probability * 100:.1f}%` pilot probability enters an "
            f"independent confirmation. Confirmation starts at "
            f"`{minimum_pairs}` pairs and adaptively grows to at most "
            f"`{maximum_pairs}` while the frozen direction can still "
            f"converge. Requested family-wide confidence is "
            f"`{regression_probability * 100:.3f}%` for regressions and "
            f"`{improvement_probability * 100:.3f}%` for improvements across "
            f"`{family_metric_count}` metrics, `{family_job_count}` performance "
            f"jobs, and at most `{maximum_looks}` confirmation looks. "
            f"Bonferroni-adjusted per-look gates are "
            f"`P(regression) >= "
            f"{effective_regression_probability * 100:.6f}%` and "
            f"`P(improvement) >= "
            f"{effective_improvement_probability * 100:.6f}%`."
        ),
        "",
        (
            "| Metric | Baseline | Candidate | Delta (pair range) | "
            "Pair volatility | P(reg) | P(imp) | Pairs | Pilot P | "
            "Initial target | Gate |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for test in results.values():
        for name, metric in test["metrics"].items():
            pair_deltas = metric["pair_deltas_percent"]
            pair_range = (
                f"{min(pair_deltas):+.2f}%..{max(pair_deltas):+.2f}%"
            )
            pilot_probability_text = format_probability(
                metric["pilot_direction_probability"]
            )
            initial_target = (
                str(metric["target_pairs"])
                if metric["selected"]
                and metric["target_pairs"] is not None
                else (
                    f"{minimum_pairs}+ adaptive"
                    if metric["selected"]
                    else "-"
                )
            )
            gate = metric["status"]
            if gate in {"REGRESSION", "IMPROVEMENT", "UNSTABLE"}:
                gate = f"**{gate}**"
            pairs = (
                f"{initial_pairs}+{metric['pair_count']}"
                if metric["selected"]
                else str(metric["pair_count"])
            )
            lines.append(
                "| "
                f"{name} | "
                f"{format_number(metric['baseline_geomean'])} | "
                f"{format_number(metric['candidate_geomean'])} | "
                f"{metric['delta_percent']:+.2f}% ({pair_range}) | "
                f"{metric['volatility_percent']:.2f}% | "
                f"{format_probability(metric['probability_regression'])} | "
                f"{format_probability(metric['probability_improvement'])} | "
                f"{pairs} | {pilot_probability_text} | "
                f"{initial_target} | {gate} |"
            )
    lines.extend([
        "",
        "> Only directions selected from the initial full-suite sample may "
        "change the gate. Metrics observed incidentally during a targeted "
        "rerun are ignored.",
        "",
    ])
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
    parser.add_argument(
        "--build-metadata",
        type=Path,
        help=(
            "Metadata from ci.performance_build; byte-identical executables "
            "are treated as drift calibration and cannot fail"
        ),
    )
    parser.add_argument("--time", type=float, default=run_tests.DEFAULT_TARGET)
    parser.add_argument("--warmup-time", type=float, default=0.5)
    parser.add_argument(
        "--blocks",
        type=int,
        default=2,
        help="Initial alternating blocks; each block contains two pairs",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=6,
        help="Minimum total pairs for an initially selected metric",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=24,
        help="Maximum total pairs for an initially selected metric",
    )
    parser.add_argument(
        "--regression-probability",
        type=float,
        default=99.99,
        metavar="PERCENT",
        help="Probability required to fail the regression gate",
    )
    parser.add_argument(
        "--improvement-probability",
        type=float,
        default=99.99,
        metavar="PERCENT",
        help="Probability required to report an improvement",
    )
    parser.add_argument(
        "--pilot-probability",
        type=float,
        default=80.0,
        metavar="PERCENT",
        help="Directional pilot probability required to start confirmation",
    )
    parser.add_argument(
        "--family-jobs",
        type=int,
        default=1,
        help=(
            "Parallel performance jobs covered by the requested family-wide "
            "probability (default: 1 for a local run)"
        ),
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
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run every selected benchmark once on A and B without deltas",
    )
    args = parser.parse_args()

    if args.time <= 0:
        parser.error("--time must be positive")
    if args.warmup_time < 0:
        parser.error("--warmup-time cannot be negative")
    if args.blocks <= 0:
        parser.error("--blocks must be positive")
    initial_pairs = args.blocks * 2
    if args.min_pairs < max(2, initial_pairs):
        parser.error("--min-pairs cannot be below the initial pair count")
    if args.max_pairs < args.min_pairs:
        parser.error("--max-pairs cannot be below --min-pairs")
    if args.min_pairs % 2 or args.max_pairs % 2:
        parser.error("--min-pairs and --max-pairs must be even")
    if args.family_jobs <= 0:
        parser.error("--family-jobs must be positive")
    for option, value in (
        ("--pilot-probability", args.pilot_probability),
        ("--regression-probability", args.regression_probability),
        ("--improvement-probability", args.improvement_probability),
    ):
        if not 50.0 < value < 100.0:
            parser.error(f"{option} must be between 50 and 100")

    pilot_probability = args.pilot_probability / 100.0
    regression_probability = args.regression_probability / 100.0
    improvement_probability = args.improvement_probability / 100.0

    try:
        identical_binaries = load_build_metadata(
            args.build_metadata,
            platform=args.platform,
            engine=args.engine,
            baseline_sha=args.baseline_sha,
            candidate_sha=args.candidate_sha,
            baseline_exec=args.baseline_exec,
            candidate_exec=args.candidate_exec,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

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

    if args.correctness_only:
        return run_correctness_suite(
            baseline_command=baseline_command,
            candidate_command=candidate_command,
            selected_tests=selected_tests,
            time_target=args.time,
            platform=args.platform,
            engine=args.engine,
            baseline_sha=args.baseline_sha,
            candidate_sha=args.candidate_sha,
            out_dir=args.out_dir,
        )

    if args.warmup_time:
        warm_test = next(
            test
            for test in run_tests.TESTS
            if not test.get("target_arg")
        )
        print(
            f"Warm-up: {warm_test['name']}, "
            f"{args.warmup_time}s per binary",
            flush=True,
        )
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
    plans_by_test: dict[str, dict[str, dict[str, Any]]] = {}
    final_by_test: dict[str, dict[str, dict[str, Any]]] = {}
    confirmation_looks_by_test: dict[str, list[int]] = {}

    try:
        def add_block(
            test: dict[str, Any],
            runs: list[dict[str, Any]],
            block: int,
        ) -> None:
            schedule = (
                ["baseline", "candidate", "baseline", "candidate"]
                if block % 2 == 0
                else ["candidate", "baseline", "candidate", "baseline"]
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
            initial = analyze_phase(
                test_name=name,
                runs=runs,
                blocks=range(args.blocks),
            )
            initial_by_test[name] = initial

        family_metric_count = sum(
            len(metrics) for metrics in initial_by_test.values()
        )
        maximum_looks = 1 + (args.max_pairs - args.min_pairs) // 2
        effective_regression_probability = family_adjusted_probability(
            regression_probability,
            metric_count=family_metric_count,
            job_count=args.family_jobs,
            maximum_looks=maximum_looks,
        )
        effective_improvement_probability = family_adjusted_probability(
            improvement_probability,
            metric_count=family_metric_count,
            job_count=args.family_jobs,
            maximum_looks=maximum_looks,
        )
        for name, initial in initial_by_test.items():
            plans_by_test[name] = metric_plans(
                initial,
                regression_probability=effective_regression_probability,
                improvement_probability=effective_improvement_probability,
                minimum_pairs=args.min_pairs,
                maximum_pairs=args.max_pairs,
                pilot_probability=pilot_probability,
            )

        target_tests = [
            test
            for test in selected_tests
            if any(
                plan["selected"]
                for plan in plans_by_test[test["name"]].values()
            )
        ]

        for index, test in enumerate(target_tests, 1):
            name = test["name"]
            plans = plans_by_test[name]
            selected_plans = {
                metric_name: plan
                for metric_name, plan in plans.items()
                if plan["selected"]
            }
            detail = ", ".join(
                f"{metric_name}:{plan['direction']}->{plan['target_pairs']}"
                for metric_name, plan in sorted(selected_plans.items())
            )
            print(
                f"[confirm {index}/{len(target_tests)}] {name}: "
                f"pilot {detail}",
                flush=True,
            )
            runs = runs_by_test[name]
            confirmation_start = args.blocks
            confirmation_pairs = 0
            next_pairs: int | None = args.min_pairs
            confirmation_looks: list[int] = []

            while next_pairs is not None:
                previous_blocks = confirmation_pairs // 2
                target_blocks = next_pairs // 2
                for relative_block in range(previous_blocks, target_blocks):
                    add_block(
                        test,
                        runs,
                        confirmation_start + relative_block,
                    )
                confirmation_pairs = next_pairs
                confirmation_looks.append(confirmation_pairs)

                confirmation_stop = (
                    confirmation_start + confirmation_pairs // 2
                )
                all_final = analyze_phase(
                    test_name=name,
                    runs=runs,
                    blocks=range(confirmation_start, confirmation_stop),
                )
                final_selected = {
                    metric_name: all_final[metric_name]
                    for metric_name in selected_plans
                }
                final_by_test[name] = final_selected

                unresolved_targets: list[int] = []
                look_detail = []
                for metric_name, plan in sorted(selected_plans.items()):
                    metric = final_selected[metric_name]
                    direction = str(plan["direction"])
                    probability = (
                        float(metric["probability_regression"])
                        if direction == "regression"
                        else float(metric["probability_improvement"])
                    )
                    target_probability = float(plan["target_probability"])
                    if probability >= target_probability:
                        state = "crossed"
                    else:
                        projected = required_pairs_for_direction(
                            metric,
                            direction=direction,
                            probability=target_probability,
                            minimum_pairs=confirmation_pairs + 2,
                            maximum_pairs=args.max_pairs,
                        )
                        if projected is None:
                            state = "futile"
                        else:
                            state = f"next->{projected}"
                            unresolved_targets.append(projected)
                    look_detail.append(
                        f"{metric_name}:{format_probability(probability)} "
                        f"{state}"
                    )
                print(
                    f"  {confirmation_pairs} confirmation pairs: "
                    + ", ".join(look_detail),
                    flush=True,
                )
                next_pairs = (
                    min(unresolved_targets)
                    if unresolved_targets
                    else None
                )

            confirmation_looks_by_test[name] = confirmation_looks

        for test in selected_tests:
            name = test["name"]
            plans = plans_by_test[name]
            metrics = classify_metrics(
                initial=initial_by_test[name],
                final=final_by_test.get(name, {}),
                plans=plans,
                regression_probability=effective_regression_probability,
                improvement_probability=effective_improvement_probability,
                identical_binaries=identical_binaries,
            )
            selected_plans = {
                metric_name: plan
                for metric_name, plan in plans.items()
                if plan["selected"]
            }
            confirmation_pairs = max(
                (
                    int(metric["pair_count"])
                    for metric_name, metric in metrics.items()
                    if metric_name in selected_plans
                ),
                default=0,
            )
            results[name] = {
                "runs": runs_by_test[name],
                "initial_blocks": args.blocks,
                "confirmation_blocks": confirmation_pairs // 2,
                "confirmation_looks": confirmation_looks_by_test.get(
                    name, []
                ),
                "candidate_metrics": sorted(selected_plans),
                "metrics": metrics,
            }
    except (ArithmeticError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    document = {
        "schema_version": 9,
        "model": "paired-log-student-t-family-corrected-adaptive-confirmation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform,
        "engine": args.engine,
        "baseline_sha": args.baseline_sha,
        "candidate_sha": args.candidate_sha,
        "target_seconds": args.time,
        "custom_target_argument_tests": [
            test["name"]
            for test in selected_tests
            if test.get("target_arg")
        ],
        "initial_pairs": initial_pairs,
        "minimum_pairs": args.min_pairs,
        "maximum_pairs": args.max_pairs,
        "pilot_probability": pilot_probability,
        "regression_probability": regression_probability,
        "improvement_probability": improvement_probability,
        "effective_regression_probability": (
            effective_regression_probability
        ),
        "effective_improvement_probability": (
            effective_improvement_probability
        ),
        "family_metric_count": family_metric_count,
        "family_job_count": args.family_jobs,
        "maximum_confirmation_looks": maximum_looks,
        "identical_binaries": identical_binaries,
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
        initial_pairs=initial_pairs,
        minimum_pairs=args.min_pairs,
        maximum_pairs=args.max_pairs,
        pilot_probability=pilot_probability,
        regression_probability=regression_probability,
        improvement_probability=improvement_probability,
        effective_regression_probability=(
            effective_regression_probability
        ),
        effective_improvement_probability=(
            effective_improvement_probability
        ),
        family_metric_count=family_metric_count,
        family_job_count=args.family_jobs,
        maximum_looks=maximum_looks,
        identical_binaries=identical_binaries,
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
