#!/usr/bin/env python3
"""Compare two Silverfir-nano revisions with pinned wasmi-benchmarks.

The upstream Criterion suite provides ten samples per benchmark.  A single
adjacent A/B process pair is therefore the full-suite pilot.  Only benchmarks
selected by that pilot are measured again, with the process order reversed.
Further confirmation runs remain limited to the initially selected benchmark
and alternate process order.

The probability model and family correction are shared with ci.performance.
The dedicated CoreMark score runner is intentionally absent; startup/coremark
remains one of the seven Criterion startup workloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ci.performance_stats import (
    classify_metrics,
    family_adjusted_probability,
    geometric_mean,
    metric_plans,
    probability_summary,
    required_pairs_for_direction,
)


WASMI_BENCHMARKS_REPOSITORY = (
    "https://github.com/wasmi-labs/wasmi-benchmarks"
)
SILVERFIR_REPOSITORY = "https://github.com/mbbill/Silverfir-nano"
WASMI_BENCHMARKS_REVISION = (
    "16a3d7c8fdb05506c116a9451175732d1ac77099"
)
CARGO_CRITERION_VERSION = "1.1.0"
CRITERION_SAMPLE_COUNT = 10

BENCHMARK_GROUPS = (
    "execute/counter-local",
    "execute/counter-param",
    "execute/counter-global",
    "execute/fibonacci-rec",
    "execute/fibonacci-iter",
    "execute/fibonacci-tail",
    "execute/sort",
    "execute/prime_sieve",
    "execute/matrix_mul",
    "execute/nbody",
    "execute/argon2",
    "execute/tiny_keccak",
    "execute/mandelbrot",
    "execute/spectralnorm",
    "execute/compression",
    "execute/word_count",
    "execute/json_parse",
    "execute/reverse_complement",
    "execute/regex_redux",
    "execute/bulk-ops",
    "startup/bz2",
    "startup/pulldown-cmark",
    "startup/spidermonkey",
    "startup/ffmpeg",
    "startup/coremark",
    "startup/argon2",
    "startup/erc20",
)

ENGINE_FEATURE = {
    "jit": "silverfir-nano-jit",
    "interp": "silverfir-nano-interp",
}
ENGINE_RUNTIME_ID = {
    "jit": "silverfir-nano.jit",
    "interp": "silverfir-nano.interpreter",
}


@dataclass(frozen=True)
class CargoContext:
    version: str
    source: Path
    suite: Path
    config: Path
    target: Path
    cargo: str
    toolchain: str
    feature: str
    runtime_id: str


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def slug(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def source_fingerprint(
    source: Path,
    package_manifests: Iterable[str],
) -> str:
    """Hash every local package resolved from the measured checkout."""
    source = source.resolve()
    roots: list[Path] = []
    for manifest in package_manifests:
        root = Path(manifest).resolve().parent
        try:
            root.relative_to(source)
        except ValueError as exc:
            raise ValueError(
                f"local package is outside measured checkout: {root}"
            ) from exc
        roots.append(root)
    roots = sorted(set(roots))
    if not roots:
        raise ValueError(f"no local packages resolved below {source}")

    digest = hashlib.sha256()
    for root in roots:
        package_relative = root.relative_to(source).as_posix()
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "target" not in path.relative_to(root).parts
            and "__pycache__" not in path.relative_to(root).parts
        )
        for path in files:
            relative = (
                f"{package_relative}/{path.relative_to(root).as_posix()}"
            ).encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def write_patch_config(path: Path, source: Path) -> None:
    core = (source / "sf-nano-core").resolve()
    if not (core / "Cargo.toml").is_file():
        raise ValueError(f"sf-nano-core checkout not found: {core}")
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted_path = json.dumps(core.as_posix())
    path.write_text(
        (
            f'[patch."{SILVERFIR_REPOSITORY}"]\n'
            f"sf-nano-core = {{ path = {quoted_path} }}\n"
        ),
        encoding="utf-8",
        newline="\n",
    )


def cargo_environment(context: CargoContext) -> dict[str, str]:
    env = dict(os.environ)
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_TARGET_DIR"] = str(context.target)
    env["CARGO_TERM_COLOR"] = "never"
    env["CRITERION_HOME"] = str(
        (context.target / "criterion-home").resolve()
    )
    env["RUST_BACKTRACE"] = "1"

    encoded = env.pop("CARGO_ENCODED_RUSTFLAGS", "")
    flags = [flag for flag in encoded.split("\x1f") if flag]
    rustflags = env.pop("RUSTFLAGS", "")
    if rustflags:
        # CI does not set whitespace-bearing path flags. Preserve ordinary
        # flags without trying to reinterpret shell quoting.
        flags.extend(rustflags.split())
    flags.append(
        "--remap-path-prefix="
        f"{context.source.resolve().as_posix()}=/workspace/sf-nano"
    )
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    return env


def cargo_prefix(context: CargoContext, subcommand: str) -> list[str]:
    return [
        context.cargo,
        f"+{context.toolchain}",
        subcommand,
    ]


def criterion_command(
    context: CargoContext,
    *arguments: str,
) -> list[str]:
    return [
        *cargo_prefix(context, "criterion"),
        "--manifest-path",
        str(context.suite / "Cargo.toml"),
        "--locked",
        "--bench",
        "criterion",
        "--no-default-features",
        "--features",
        context.feature,
        "--plotting-backend",
        "disabled",
        *arguments,
    ]


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    capture: bool,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {command_text(command)}", flush=True)
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
        check=False,
    )
    if check:
        raise_for_process_failure(result, command)
    return result


def raise_for_process_failure(
    result: subprocess.CompletedProcess[str],
    command: Sequence[str],
) -> None:
    if not result.returncode:
        return
    detail = (result.stderr or result.stdout or "").strip()
    if len(detail) > 4000:
        detail = detail[-4000:]
    raise RuntimeError(
        f"command failed with exit code {result.returncode}: "
        f"{command_text(command)}\n{detail}"
    )


def git_revision(path: Path) -> str:
    result = run_process(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        cwd=path,
        env=dict(os.environ),
        capture=True,
    )
    return result.stdout.strip()


def verify_source_revision(
    source: Path,
    expected: str,
    *,
    label: str,
) -> str:
    if len(expected) < 7:
        raise ValueError(
            f"{label}: expected revision must contain at least 7 characters"
        )
    observed = git_revision(source)
    if not observed.startswith(expected):
        raise ValueError(
            f"{label}: checkout revision is {observed}, expected {expected}"
        )
    return observed


def verify_cargo_criterion(
    cargo: str,
    toolchain: str,
) -> str:
    result = run_process(
        [cargo, f"+{toolchain}", "criterion", "--version"],
        cwd=Path.cwd(),
        env=dict(os.environ),
        capture=True,
    )
    observed = result.stdout.strip()
    expected = f"cargo-criterion {CARGO_CRITERION_VERSION}"
    if observed != expected:
        raise ValueError(
            f"expected {expected!r}, observed {observed!r}"
        )
    return observed


def prepare_suite_copy(source: Path, destination: Path) -> None:
    """Create an isolated mutable lockfile context from the pinned suite."""
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {".git", "target"}}

    shutil.copytree(source, destination, ignore=ignore)
    if not (destination / "Cargo.lock").is_file():
        raise ValueError(
            f"isolated wasmi-benchmarks copy has no Cargo.lock: {destination}"
        )


def reachable_packages(
    metadata: dict[str, Any],
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Return only packages reachable with the selected Cargo features."""
    resolve = metadata.get("resolve") or {}
    root_id = resolve.get("root")
    return dependency_closure_packages(
        metadata,
        root_id,
        label=label,
    )


def dependency_closure_packages(
    metadata: dict[str, Any],
    root_id: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Return a package and only its transitive dependencies."""
    packages = {
        package["id"]: package for package in metadata.get("packages", [])
    }
    resolve = metadata.get("resolve") or {}
    nodes = {
        node["id"]: node for node in resolve.get("nodes", [])
    }
    if root_id not in nodes:
        raise ValueError(
            f"{label}: Cargo metadata has no node for root {root_id}"
        )

    resolved_ids: set[str] = set()
    pending = [str(root_id)]
    while pending:
        package_id = pending.pop()
        if package_id in resolved_ids:
            continue
        node = nodes.get(package_id)
        if node is None:
            raise ValueError(
                f"{label}: Cargo metadata omits node {package_id}"
            )
        resolved_ids.add(package_id)
        pending.extend(
            str(dependency["pkg"])
            for dependency in node.get("deps", [])
        )
    return [
        packages[package_id]
        for package_id in resolved_ids
        if package_id in packages
    ]


def verify_resolution(context: CargoContext) -> dict[str, Any]:
    command = [
        *cargo_prefix(context, "metadata"),
        "--manifest-path",
        str(context.suite / "Cargo.toml"),
        "--format-version",
        "1",
        "--no-default-features",
        "--features",
        context.feature,
    ]
    result = run_process(
        command,
        cwd=context.suite,
        env=cargo_environment(context),
        capture=True,
    )
    metadata = json.loads(result.stdout)
    resolved = reachable_packages(metadata, label=context.version)

    nano = [
        package for package in resolved if package.get("name") == "sf-nano-core"
    ]
    if len(nano) != 1:
        raise ValueError(
            f"{context.version}: expected one resolved sf-nano-core, "
            f"found {len(nano)}"
        )
    expected_manifest = (
        context.source / "sf-nano-core" / "Cargo.toml"
    ).resolve()
    observed_manifest = Path(nano[0]["manifest_path"]).resolve()
    if observed_manifest != expected_manifest:
        raise ValueError(
            f"{context.version}: sf-nano-core resolved to "
            f"{observed_manifest}, expected {expected_manifest}"
        )

    runtime_adapters = sorted(
        {
            str(package["name"])
            for package in resolved
            if str(package.get("name", "")).startswith("rt-")
        }
    )
    if runtime_adapters != ["rt-silverfir-nano"]:
        raise ValueError(
            f"{context.version}: unexpected runtime adapters resolved: "
            + ", ".join(runtime_adapters)
        )

    runtime_dependencies = dependency_closure_packages(
        metadata,
        nano[0]["id"],
        label=f"{context.version} sf-nano-core",
    )
    source_root = context.source.resolve()
    local_manifests = []
    for package in runtime_dependencies:
        manifest = Path(str(package["manifest_path"])).resolve()
        try:
            manifest.relative_to(source_root)
        except ValueError:
            continue
        local_manifests.append(str(manifest))
    local_manifests.sort()
    if str(expected_manifest) not in local_manifests:
        raise ValueError(
            f"{context.version}: resolved local package set omits sf-nano-core"
        )

    # The first metadata call updates only this version's private lockfile for
    # the local [patch]. Every build and measurement after this is locked.
    locked_command = [*command]
    locked_command.insert(
        locked_command.index("--no-default-features"),
        "--locked",
    )
    run_process(
        locked_command,
        cwd=context.suite,
        env=cargo_environment(context),
        capture=True,
    )
    return {
        "sf_nano_manifest": str(observed_manifest),
        "runtime_adapters": runtime_adapters,
        "feature": context.feature,
        "local_manifests": local_manifests,
    }


def build_context(context: CargoContext) -> dict[str, Any]:
    context.target.mkdir(parents=True, exist_ok=True)
    criterion_home = context.target / "criterion-home"
    if criterion_home.exists():
        shutil.rmtree(criterion_home)
    resolution = verify_resolution(context)
    started = time.monotonic()
    run_process(
        criterion_command(context, "--no-run"),
        cwd=context.suite,
        env=cargo_environment(context),
        capture=False,
    )
    return {
        **resolution,
        "target": str(context.target.resolve()),
        "elapsed_seconds": time.monotonic() - started,
        "source_fingerprint": source_fingerprint(
            context.source,
            resolution["local_manifests"],
        ),
    }


def parse_criterion_json(
    output: str,
    *,
    group: str,
    runtime_id: str,
) -> dict[str, Any]:
    messages = []
    for line_number, line in enumerate(output.splitlines(), 1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{group}: non-JSON cargo-criterion output on line "
                f"{line_number}: {line[:160]!r}"
            ) from exc
        if isinstance(message, dict):
            messages.append(message)

    completed = [
        message
        for message in messages
        if message.get("reason") == "benchmark-complete"
    ]
    if len(completed) != 1:
        raise ValueError(
            f"{group}: expected one benchmark-complete message, "
            f"found {len(completed)}"
        )
    message = completed[0]
    benchmark_id = str(message.get("id", ""))
    expected_prefix = f"{group}/{runtime_id}"
    if (
        benchmark_id != expected_prefix
        and not benchmark_id.startswith(expected_prefix + "/")
    ):
        raise ValueError(
            f"{group}: unexpected benchmark id {benchmark_id!r}; "
            f"expected {expected_prefix!r}"
        )

    iterations = message.get("iteration_count")
    measured = message.get("measured_values")
    if not isinstance(iterations, list) or not isinstance(measured, list):
        raise ValueError(f"{group}: cargo-criterion samples are missing")
    if len(iterations) != CRITERION_SAMPLE_COUNT:
        raise ValueError(
            f"{group}: expected {CRITERION_SAMPLE_COUNT} iteration samples, "
            f"found {len(iterations)}"
        )
    if len(measured) != len(iterations):
        raise ValueError(f"{group}: sample arrays have different lengths")
    unit = str(message.get("unit", ""))
    if unit != "ns":
        raise ValueError(
            f"{group}: expected nanosecond samples, found unit {unit!r}"
        )

    normalized = []
    for index, (count, value) in enumerate(zip(iterations, measured)):
        if (
            not isinstance(count, (int, float))
            or not isinstance(value, (int, float))
            or count <= 0
            or value <= 0
        ):
            raise ValueError(f"{group}: invalid sample at index {index}")
        normalized.append(float(value) / float(count))

    return {
        "benchmark_id": benchmark_id,
        "iteration_count": [float(value) for value in iterations],
        "measured_values": [float(value) for value in measured],
        "normalized_ns": normalized,
        "unit": unit,
        "typical": message.get("typical"),
    }


def measure_case(
    context: CargoContext,
    *,
    group: str,
    phase: str,
    run_index: int,
    raw_root: Path,
) -> dict[str, Any]:
    run_dir = (
        raw_root / phase / f"run-{run_index + 1}" / slug(group) / context.version
    )
    archived_criterion = run_dir / "criterion"
    criterion_home = context.target / "criterion-home"
    if criterion_home.exists():
        shutil.rmtree(criterion_home)
    if archived_criterion.exists():
        shutil.rmtree(archived_criterion)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = cargo_environment(context)
    command = criterion_command(
        context,
        "--message-format=json",
        group,
    )
    print(
        f"[{phase} {run_index + 1}] {group}: {context.version}",
        flush=True,
    )
    started = time.monotonic()
    result = run_process(
        command,
        cwd=context.suite,
        env=env,
        capture=True,
        check=False,
    )
    elapsed = time.monotonic() - started

    (run_dir / "stdout.jsonl").write_text(
        result.stdout, encoding="utf-8", newline="\n"
    )
    (run_dir / "stderr.log").write_text(
        result.stderr, encoding="utf-8", newline="\n"
    )
    (run_dir / "command.txt").write_text(
        command_text(command) + "\n", encoding="utf-8", newline="\n"
    )
    if criterion_home.exists():
        shutil.copytree(criterion_home, archived_criterion)
        shutil.rmtree(criterion_home)
    raise_for_process_failure(result, command)
    sample = parse_criterion_json(
        result.stdout,
        group=group,
        runtime_id=context.runtime_id,
    )
    print(
        f"  {context.version}: {elapsed:.2f}s, "
        f"{len(sample['normalized_ns'])} Criterion samples",
        flush=True,
    )
    return {
        "version": context.version,
        "elapsed_seconds": elapsed,
        **sample,
    }


def measure_pair(
    contexts: dict[str, CargoContext],
    *,
    group: str,
    phase: str,
    run_index: int,
    baseline_first: bool,
    raw_root: Path,
) -> dict[str, Any]:
    order = (
        ("baseline", "candidate")
        if baseline_first
        else ("candidate", "baseline")
    )
    measured = {
        version: measure_case(
            contexts[version],
            group=group,
            phase=phase,
            run_index=run_index,
            raw_root=raw_root,
        )
        for version in order
    }
    return {
        "order": list(order),
        "baseline": measured["baseline"],
        "candidate": measured["candidate"],
    }


def summarize_pairs(pairs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    ratios: list[float] = []
    pair_count = 0
    for pair in pairs:
        baseline = list(pair["baseline"]["normalized_ns"])
        candidate = list(pair["candidate"]["normalized_ns"])
        if len(baseline) != len(candidate):
            raise ValueError("baseline and candidate sample counts differ")
        baseline_samples.extend(baseline)
        candidate_samples.extend(candidate)
        ratios.extend(
            baseline_value / candidate_value
            for baseline_value, candidate_value in zip(baseline, candidate)
        )
        pair_count += 1
    if not ratios:
        raise ValueError("no Criterion pairs to summarize")
    return {
        "unit": "ns/iteration",
        "direction": "lower",
        "process_pairs": pair_count,
        "baseline_samples": baseline_samples,
        "candidate_samples": candidate_samples,
        "baseline_geomean": geometric_mean(baseline_samples),
        "candidate_geomean": geometric_mean(candidate_samples),
        **probability_summary(ratios),
    }


def format_number(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}s"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.3f}ms"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.3f}us"
    return f"{value:.2f}ns"


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
    suite_sha: str,
    pilot_probability: float,
    regression_probability: float,
    improvement_probability: float,
    effective_regression_probability: float,
    effective_improvement_probability: float,
    family_jobs: int,
    maximum_looks: int,
    identical_sources: bool,
    metrics: dict[str, dict[str, Any]],
) -> str:
    schedule = (
        "- schedule: one 10-sample adjacent A/B pilot; resolved local "
        "runtime source is identical, so confirmation is skipped"
        if identical_sources
        else (
            "- schedule: one 10-sample adjacent A/B pilot; selected "
            f"benchmarks receive up to `{maximum_looks}` independent "
            "reverse/alternating confirmation pairs"
        )
    )
    lines = [
        f"## wasmi-benchmarks: {platform} / {engine}",
        "",
        f"`{baseline_sha[:12]}` -> `{candidate_sha[:12]}`",
        "",
        f"- wasmi-benchmarks: `{suite_sha}`",
        (
            f"- corpus: `{len(metrics)}` Criterion benchmarks; dedicated "
            "CoreMark score excluded, `startup/coremark` retained"
        ),
        schedule,
        (
            f"- requested family confidence: regression "
            f"`{regression_probability * 100:.3f}%`, improvement "
            f"`{improvement_probability * 100:.3f}%`; pilot "
            f"`{pilot_probability * 100:.1f}%`"
        ),
        (
            f"- family correction: `{len(metrics)}` metrics x "
            f"`{family_jobs}` jobs x `{maximum_looks}` looks; effective "
            f"P(reg) `{effective_regression_probability * 100:.6f}%`, "
            f"P(imp) `{effective_improvement_probability * 100:.6f}%`"
        ),
        "",
    ]
    if identical_sources:
        lines.extend([
            (
                "> **Identical resolved local runtime source:** apparent "
                "regressions "
                "are runner drift and are reported as UNSTABLE, not failures."
            ),
            "",
        ])
    lines.extend([
        (
            "| Benchmark | Baseline | Candidate | Delta (sample range) | "
            "Volatility | P(reg) | P(imp) | Samples | Runs | Pilot P | Gate |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for name in BENCHMARK_GROUPS:
        if name not in metrics:
            continue
        metric = metrics[name]
        deltas = metric["pair_deltas_percent"]
        delta_range = f"{min(deltas):+.2f}%..{max(deltas):+.2f}%"
        gate = metric["status"]
        if gate in {"REGRESSION", "IMPROVEMENT", "UNSTABLE"}:
            gate = f"**{gate}**"
        lines.append(
            "| "
            f"{name} | {format_number(metric['baseline_geomean'])} | "
            f"{format_number(metric['candidate_geomean'])} | "
            f"{metric['delta_percent']:+.2f}% ({delta_range}) | "
            f"{metric['volatility_percent']:.2f}% | "
            f"{format_probability(metric['probability_regression'])} | "
            f"{format_probability(metric['probability_improvement'])} | "
            f"{metric['pair_count']} | {metric['process_pairs']} | "
            f"{format_probability(metric['pilot_direction_probability'])} | "
            f"{gate} |"
        )
    lines.extend([
        "",
        (
            "> Only directions selected by the full-suite pilot can affect "
            "the gate. Later changes in other benchmarks are ignored."
        ),
        "",
    ])
    return "\n".join(lines)


def render_failure_summary(
    *,
    platform: str,
    engine: str,
    baseline_sha: str,
    candidate_sha: str,
    error: str,
) -> str:
    return "\n".join([
        f"## wasmi-benchmarks: {platform} / {engine}",
        "",
        f"`{baseline_sha[:12]}` -> `{candidate_sha[:12]}`",
        "",
        "> [!CAUTION]",
        "> **SETUP OR MEASUREMENT ERROR.** This job did not produce a",
        "> performance verdict. Inspect the job log and uploaded raw data.",
        "",
        "```text",
        error.replace("```", "'''")[-4000:],
        "```",
        "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--baseline-source", required=True, type=Path)
    parser.add_argument("--candidate-source", required=True, type=Path)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--engine", choices=("jit", "interp"), required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument(
        "--work-root",
        type=Path,
        help=(
            "Ephemeral isolated suite copies; defaults beside --target-root"
        ),
    )
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--toolchain", default="1.97.0")
    parser.add_argument(
        "--suite-sha",
        default=WASMI_BENCHMARKS_REVISION,
    )
    parser.add_argument(
        "--pilot-probability",
        type=float,
        default=80.0,
        metavar="PERCENT",
    )
    parser.add_argument(
        "--regression-probability",
        type=float,
        default=99.99,
        metavar="PERCENT",
    )
    parser.add_argument(
        "--improvement-probability",
        type=float,
        default=99.99,
        metavar="PERCENT",
    )
    parser.add_argument("--family-jobs", type=int, default=4)
    parser.add_argument(
        "--max-confirmation-runs",
        type=int,
        default=2,
        help=(
            "Maximum independent A/B process pairs after the pilot; each "
            f"pair contributes {CRITERION_SAMPLE_COUNT} samples"
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="GROUP",
        help="Run only an exact benchmark group; may be repeated",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    for option, value in (
        ("--pilot-probability", args.pilot_probability),
        ("--regression-probability", args.regression_probability),
        ("--improvement-probability", args.improvement_probability),
    ):
        if not 50.0 < value < 100.0:
            parser.error(f"{option} must be between 50 and 100")
    if args.family_jobs <= 0:
        parser.error("--family-jobs must be positive")
    if args.max_confirmation_runs <= 0:
        parser.error("--max-confirmation-runs must be positive")
    if (
        not args.platform
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in args.platform
        )
    ):
        parser.error(
            "--platform may contain only letters, digits, dot, dash, "
            "and underscore"
        )

    selected_groups = list(BENCHMARK_GROUPS)
    if args.only:
        unknown = set(args.only) - set(BENCHMARK_GROUPS)
        if unknown:
            parser.error(
                "--only contains unknown groups: "
                + ", ".join(sorted(unknown))
            )
        selected_groups = [
            group for group in BENCHMARK_GROUPS if group in set(args.only)
        ]

    args.suite = args.suite.resolve()
    args.baseline_source = args.baseline_source.resolve()
    args.candidate_source = args.candidate_source.resolve()
    args.out_dir = args.out_dir.resolve()
    args.target_root = args.target_root.resolve()
    args.work_root = (
        args.work_root.resolve()
        if args.work_root is not None
        else (args.target_root.parent / "wasmi-work").resolve()
    )
    if not (args.suite / "Cargo.toml").is_file():
        parser.error(f"wasmi-benchmarks checkout not found: {args.suite}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = args.out_dir / "cargo-config"
    raw_root = args.out_dir / "raw"

    pilot_probability = args.pilot_probability / 100.0
    regression_probability = args.regression_probability / 100.0
    improvement_probability = args.improvement_probability / 100.0
    maximum_looks = args.max_confirmation_runs
    effective_regression_probability = family_adjusted_probability(
        regression_probability,
        metric_count=len(selected_groups),
        job_count=args.family_jobs,
        maximum_looks=maximum_looks,
    )
    effective_improvement_probability = family_adjusted_probability(
        improvement_probability,
        metric_count=len(selected_groups),
        job_count=args.family_jobs,
        maximum_looks=maximum_looks,
    )

    try:
        cargo_criterion_version = verify_cargo_criterion(
            args.cargo,
            args.toolchain,
        )
        suite_revision = git_revision(args.suite)
        if suite_revision != args.suite_sha:
            raise ValueError(
                f"wasmi-benchmarks revision is {suite_revision}, "
                f"expected {args.suite_sha}"
            )

        source_revisions = {
            "baseline": verify_source_revision(
                args.baseline_source,
                args.baseline_sha,
                label="baseline",
            ),
            "candidate": verify_source_revision(
                args.candidate_source,
                args.candidate_sha,
                label="candidate",
            ),
        }
        contexts: dict[str, CargoContext] = {}
        feature = ENGINE_FEATURE[args.engine]
        runtime_id = ENGINE_RUNTIME_ID[args.engine]
        for version, source in (
            ("baseline", args.baseline_source),
            ("candidate", args.candidate_source),
        ):
            suite_copy = (
                args.work_root
                / args.platform
                / args.engine
                / version
                / "suite"
            )
            prepare_suite_copy(args.suite, suite_copy)
            config = suite_copy / ".cargo" / "config.toml"
            write_patch_config(config, source)
            configs.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, configs / f"{version}.toml")
            contexts[version] = CargoContext(
                version=version,
                source=source,
                suite=suite_copy,
                config=config,
                target=(
                    args.target_root
                    / args.platform
                    / args.engine
                    / version
                ),
                cargo=args.cargo,
                toolchain=args.toolchain,
                feature=feature,
                runtime_id=runtime_id,
            )

        builds = {
            version: build_context(context)
            for version, context in contexts.items()
        }
        for version, revision in source_revisions.items():
            builds[version]["source_revision"] = revision
        identical_sources = (
            builds["baseline"]["source_fingerprint"]
            == builds["candidate"]["source_fingerprint"]
        )

        pilot_pairs: dict[str, dict[str, Any]] = {}
        initial: dict[str, dict[str, Any]] = {}
        for index, group in enumerate(selected_groups):
            pair = measure_pair(
                contexts,
                group=group,
                phase="pilot",
                run_index=0,
                baseline_first=index % 2 == 0,
                raw_root=raw_root,
            )
            pilot_pairs[group] = pair
            initial[group] = summarize_pairs([pair])

        max_pairs = CRITERION_SAMPLE_COUNT * args.max_confirmation_runs
        plans = metric_plans(
            initial,
            regression_probability=effective_regression_probability,
            improvement_probability=effective_improvement_probability,
            minimum_pairs=CRITERION_SAMPLE_COUNT,
            maximum_pairs=max_pairs,
            pilot_probability=pilot_probability,
        )
        confirmation_pairs: dict[str, list[dict[str, Any]]] = {
            group: []
            for group, plan in plans.items()
            if plan["selected"]
        }
        pending = set(confirmation_pairs)

        if identical_sources:
            final = {
                group: initial[group]
                for group in confirmation_pairs
            }
        else:
            for run_index in range(args.max_confirmation_runs):
                if not pending:
                    break
                for index, group in enumerate(selected_groups):
                    if group not in pending:
                        continue
                    pilot_baseline_first = index % 2 == 0
                    pair = measure_pair(
                        contexts,
                        group=group,
                        phase="confirmation",
                        run_index=run_index,
                        baseline_first=(
                            pilot_baseline_first
                            if run_index % 2
                            else not pilot_baseline_first
                        ),
                        raw_root=raw_root,
                    )
                    confirmation_pairs[group].append(pair)
                    metric = summarize_pairs(confirmation_pairs[group])
                    direction = str(plans[group]["direction"])
                    target = (
                        effective_regression_probability
                        if direction == "regression"
                        else effective_improvement_probability
                    )
                    probability = float(
                        metric[
                            "probability_regression"
                            if direction == "regression"
                            else "probability_improvement"
                        ]
                    )
                    if probability >= target:
                        pending.remove(group)
                        continue

                    current_pairs = int(metric["pair_count"])
                    if current_pairs >= max_pairs:
                        pending.remove(group)
                        continue
                    projected = required_pairs_for_direction(
                        metric,
                        direction=direction,
                        probability=target,
                        minimum_pairs=current_pairs + 2,
                        maximum_pairs=max_pairs,
                    )
                    if projected is None:
                        pending.remove(group)

            final = {
                group: summarize_pairs(pairs)
                for group, pairs in confirmation_pairs.items()
            }
        metrics = classify_metrics(
            initial=initial,
            final=final,
            plans=plans,
            regression_probability=effective_regression_probability,
            improvement_probability=effective_improvement_probability,
            identical_binaries=identical_sources,
        )
    except (
        ArithmeticError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        error = str(exc)
        print(f"ERROR: {error}", file=sys.stderr)
        failure_summary = render_failure_summary(
            platform=args.platform,
            engine=args.engine,
            baseline_sha=args.baseline_sha,
            candidate_sha=args.candidate_sha,
            error=error,
        )
        (args.out_dir / "summary.md").write_text(
            failure_summary,
            encoding="utf-8",
            newline="\n",
        )
        (args.out_dir / "failure.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ERROR",
                    "platform": args.platform,
                    "engine": args.engine,
                    "baseline_sha": args.baseline_sha,
                    "candidate_sha": args.candidate_sha,
                    "error": error,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return 2

    for group, metric in metrics.items():
        metric["process_pairs"] = (
            1
            if identical_sources or not plans[group]["selected"]
            else len(confirmation_pairs[group])
        )

    document = {
        "schema_version": 1,
        "model": (
            "criterion-samples-paired-log-student-t-"
            "family-corrected-adaptive-confirmation"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform,
        "engine": args.engine,
        "baseline_sha": args.baseline_sha,
        "candidate_sha": args.candidate_sha,
        "wasmi_benchmarks_repository": WASMI_BENCHMARKS_REPOSITORY,
        "wasmi_benchmarks_revision": suite_revision,
        "cargo_criterion_version": cargo_criterion_version,
        "criterion_samples_per_run": CRITERION_SAMPLE_COUNT,
        "pilot_probability": pilot_probability,
        "regression_probability": regression_probability,
        "improvement_probability": improvement_probability,
        "effective_regression_probability": (
            effective_regression_probability
        ),
        "effective_improvement_probability": (
            effective_improvement_probability
        ),
        "family_metric_count": len(selected_groups),
        "family_job_count": args.family_jobs,
        "maximum_confirmation_looks": maximum_looks,
        "identical_sources": identical_sources,
        "builds": builds,
        "pilot_pairs": pilot_pairs,
        "confirmation_pairs": confirmation_pairs,
        "metrics": metrics,
    }
    (args.out_dir / "comparison.json").write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    summary = render_summary(
        platform=args.platform,
        engine=args.engine,
        baseline_sha=args.baseline_sha,
        candidate_sha=args.candidate_sha,
        suite_sha=suite_revision,
        pilot_probability=pilot_probability,
        regression_probability=regression_probability,
        improvement_probability=improvement_probability,
        effective_regression_probability=(
            effective_regression_probability
        ),
        effective_improvement_probability=(
            effective_improvement_probability
        ),
        family_jobs=args.family_jobs,
        maximum_looks=maximum_looks,
        identical_sources=identical_sources,
        metrics=metrics,
    )
    (args.out_dir / "summary.md").write_text(
        summary, encoding="utf-8", newline="\n"
    )
    print()
    print(summary)
    return (
        1
        if any(
            metric["status"] == "REGRESSION"
            for metric in metrics.values()
        )
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
