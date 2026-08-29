#!/usr/bin/env python3
"""Measure and rank the complete wasmi-benchmarks interpreter field.

This is intentionally separate from ``ci.wasmi_performance``.  The regression
driver compares two Silverfir-nano revisions while enabling exactly one runtime
adapter.  This driver builds the pinned suite's complete ``interpreters``
feature once, measures all seven startup workloads on one runner, and answers a
different question: is Silverfir-nano faster to start than every non-lazy
interpreter in that same run?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from ci import wasmi_performance
from ci.performance_stats import geometric_mean


WASMI_BENCHMARKS_REVISION = wasmi_performance.WASMI_BENCHMARKS_REVISION
CARGO_CRITERION_VERSION = wasmi_performance.CARGO_CRITERION_VERSION
INTERPRETER_FEATURE = "interpreters"
NANO_RUNTIME_ID = "silverfir-nano.interpreter"
STARTUP_GROUPS = tuple(
    group
    for group in wasmi_performance.BENCHMARK_GROUPS
    if group.startswith("startup/")
)

# The exact runtime field registered by the pinned suite's ``interpreters``
# feature.  Keeping this explicit turns an upstream feature or registration
# drift into a visible harness failure instead of silently changing the race.
EXPECTED_INTERPRETERS = frozenset(
    {
        "dlr-wasm-interpreter",
        "fizzy",
        NANO_RUNTIME_ID,
        "spacewasm",
        "stitch",
        "submilli-wasm",
        "tinywasm",
        "toywasm",
        "wamr",
        "wasm3.eager",
        "wasm3.lazy",
        "wasmedge",
        "wasmi-v0.31",
        "wasmi-v0.32",
        "wasmi-v1.eager.checked",
        "wasmi-v1.lazy-translation.checked",
        "wasmi-v1.lazy.checked",
        "wasmi-v1.lazy.unchecked",
        "wasmi-v2.eager.checked",
        "wasmi-v2.lazy-translation.checked",
        "wasmi-v2.lazy.checked",
        "wasmi-v2.lazy.unchecked",
        "wasmtime.pulley",
    }
)

EXPECTED_RUNTIME_ADAPTERS = frozenset(
    {
        "rt-dlr-wasm-interpreter",
        "rt-fizzy",
        "rt-silverfir-nano",
        "rt-spacewasm",
        "rt-stitch",
        "rt-submilli-wasm",
        "rt-tinywasm",
        "rt-toywasm",
        "rt-wamr",
        "rt-wasm3",
        "rt-wasmedge",
        "rt-wasmi-v0-31",
        "rt-wasmi-v0-32",
        "rt-wasmi-v1",
        "rt-wasmi-v2",
        "rt-wasmtime",
    }
)

# The pinned adapter deliberately skips FFmpeg for Pulley because translating
# it through Cranelift takes too long.  Every other interpreter must produce all
# seven rows.  Pulley remains in the non-lazy gate, compared over the six cases
# it and Nano both run, and the report labels that exception explicitly.
ALLOWED_MISSING = {
    "wasmtime.pulley": frozenset({"startup/ffmpeg"}),
}


def cargo_environment(target_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_TARGET_DIR"] = str(target_root.resolve())
    env["CARGO_TERM_COLOR"] = "never"
    env["CRITERION_HOME"] = str(
        (target_root / "criterion-home").resolve()
    )
    env["RUST_BACKTRACE"] = "1"
    return env


def criterion_command(
    suite: Path,
    *arguments: str,
    cargo: str = "cargo",
    toolchain: str = "stable",
) -> list[str]:
    return [
        cargo,
        f"+{toolchain}",
        "criterion",
        "--manifest-path",
        str((suite / "Cargo.toml").resolve()),
        "--locked",
        "--bench",
        "criterion",
        "--no-default-features",
        "--features",
        INTERPRETER_FEATURE,
        "--plotting-backend",
        "disabled",
        *arguments,
    ]


def metadata_command(
    suite: Path,
    *,
    locked: bool,
    cargo: str = "cargo",
    toolchain: str = "stable",
) -> list[str]:
    command = [
        cargo,
        f"+{toolchain}",
        "metadata",
        "--manifest-path",
        str((suite / "Cargo.toml").resolve()),
        "--format-version",
        "1",
    ]
    if locked:
        command.append("--locked")
    command.extend(
        [
            "--no-default-features",
            "--features",
            INTERPRETER_FEATURE,
        ]
    )
    return command


def prepare(
    *,
    suite: Path,
    source: Path,
    target_root: Path,
    metadata_out: Path,
    cargo: str,
    toolchain: str,
) -> None:
    suite = suite.resolve()
    source = source.resolve()
    target_root = target_root.resolve()
    if wasmi_performance.git_revision(suite) != WASMI_BENCHMARKS_REVISION:
        raise ValueError(
            "wasmi-benchmarks checkout does not match the pinned revision "
            f"{WASMI_BENCHMARKS_REVISION}"
        )
    wasmi_performance.verify_cargo_criterion(cargo, toolchain)
    wasmi_performance.write_patch_config(
        suite / ".cargo" / "config.toml",
        source,
    )

    env = cargo_environment(target_root)
    # The first metadata pass updates only this checked-out suite's lockfile for
    # the candidate path patch.  Every command after it is locked.
    result = wasmi_performance.run_process(
        metadata_command(
            suite,
            locked=False,
            cargo=cargo,
            toolchain=toolchain,
        ),
        cwd=suite,
        env=env,
        capture=True,
    )
    metadata = json.loads(result.stdout)
    reachable = wasmi_performance.reachable_packages(
        metadata,
        label="startup ranking",
    )
    nano_packages = [
        package
        for package in reachable
        if package.get("name") == "sf-nano-core"
    ]
    expected_manifest = (source / "sf-nano-core" / "Cargo.toml").resolve()
    if len(nano_packages) != 1:
        raise ValueError(
            "expected one reachable sf-nano-core package, found "
            f"{len(nano_packages)}"
        )
    observed_manifest = Path(nano_packages[0]["manifest_path"]).resolve()
    if observed_manifest != expected_manifest:
        raise ValueError(
            f"sf-nano-core resolved to {observed_manifest}, "
            f"expected candidate {expected_manifest}"
        )

    runtime_adapters = frozenset(
        str(package["name"])
        for package in reachable
        if str(package.get("name", "")).startswith("rt-")
    )
    if runtime_adapters != EXPECTED_RUNTIME_ADAPTERS:
        missing = sorted(EXPECTED_RUNTIME_ADAPTERS - runtime_adapters)
        extra = sorted(runtime_adapters - EXPECTED_RUNTIME_ADAPTERS)
        raise ValueError(
            "pinned interpreter adapter field changed; "
            f"missing={missing}, extra={extra}"
        )

    wasmi_performance.run_process(
        metadata_command(
            suite,
            locked=True,
            cargo=cargo,
            toolchain=toolchain,
        ),
        cwd=suite,
        env=env,
        capture=True,
    )
    wasmi_performance.run_process(
        criterion_command(
            suite,
            "--no-run",
            cargo=cargo,
            toolchain=toolchain,
        ),
        cwd=suite,
        env=env,
        capture=False,
    )

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(
        json.dumps(
            {
                "candidate_revision": wasmi_performance.git_revision(source),
                "cargo_criterion": CARGO_CRITERION_VERSION,
                "feature": INTERPRETER_FEATURE,
                "runtime_adapters": sorted(runtime_adapters),
                "sf_nano_manifest": str(observed_manifest),
                "wasmi_benchmarks_revision": WASMI_BENCHMARKS_REVISION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def measure(
    *,
    suite: Path,
    target_root: Path,
    raw_dir: Path,
    cargo: str,
    toolchain: str,
) -> None:
    suite = suite.resolve()
    target_root = target_root.resolve()
    raw_dir = raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    criterion_home = target_root / "criterion-home"
    failures: list[str] = []
    for group in STARTUP_GROUPS:
        if criterion_home.exists():
            shutil.rmtree(criterion_home)
        slug = group.replace("/", "__")
        stdout_path = raw_dir / f"{slug}.jsonl"
        stderr_path = raw_dir / f"{slug}.stderr.log"
        command_path = raw_dir / f"{slug}.command.txt"
        command = criterion_command(
            suite,
            "--message-format=json",
            group,
            cargo=cargo,
            toolchain=toolchain,
        )
        print(f"measuring {group}", flush=True)
        result = wasmi_performance.run_process(
            command,
            cwd=suite,
            env=cargo_environment(target_root),
            capture=True,
            check=False,
        )
        stdout_path.write_text(
            result.stdout,
            encoding="utf-8",
            newline="\n",
        )
        stderr_path.write_text(
            result.stderr,
            encoding="utf-8",
            newline="\n",
        )
        command_path.write_text(
            wasmi_performance.command_text(command) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if result.returncode:
            failures.append(f"{group}: exit {result.returncode}")
            print(f"{failures[-1]} (continuing)", file=sys.stderr)
    if failures:
        raise RuntimeError("one or more startup groups failed: " + "; ".join(failures))


def decode_json_stream(text: str, *, label: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    index = 0
    records: list[dict[str, Any]] = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}: invalid JSON stream at byte {index}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def collect_measurements(raw_dir: Path) -> dict[str, dict[str, float]]:
    measurements: dict[str, dict[str, float]] = {}
    for group in STARTUP_GROUPS:
        path = raw_dir / f"{group.replace('/', '__')}.jsonl"
        if not path.is_file():
            raise ValueError(f"missing raw result for {group}: {path}")
        for record in decode_json_stream(
            path.read_text(encoding="utf-8"),
            label=group,
        ):
            if record.get("reason") != "benchmark-complete":
                continue
            benchmark_id = str(record.get("id", ""))
            prefix = group + "/"
            if not benchmark_id.startswith(prefix):
                continue
            engine = benchmark_id[len(prefix) :]
            if not engine or "/" in engine:
                raise ValueError(f"unexpected startup benchmark id {benchmark_id!r}")
            estimate = record.get("typical") or record.get("mean")
            if not isinstance(estimate, dict):
                raise ValueError(f"{benchmark_id}: missing timing estimate")
            nanoseconds = estimate.get("estimate")
            if not isinstance(nanoseconds, (int, float)) or nanoseconds <= 0:
                raise ValueError(f"{benchmark_id}: invalid timing estimate")
            engine_results = measurements.setdefault(engine, {})
            if group in engine_results:
                raise ValueError(f"duplicate result for {benchmark_id}")
            engine_results[group] = float(nanoseconds)

    observed = frozenset(measurements)
    if observed != EXPECTED_INTERPRETERS:
        missing = sorted(EXPECTED_INTERPRETERS - observed)
        extra = sorted(observed - EXPECTED_INTERPRETERS)
        raise ValueError(
            "pinned interpreter result field changed; "
            f"missing={missing}, extra={extra}"
        )
    expected_groups = frozenset(STARTUP_GROUPS)
    for engine, results in measurements.items():
        missing = expected_groups - frozenset(results)
        allowed = ALLOWED_MISSING.get(engine, frozenset())
        if missing != allowed:
            raise ValueError(
                f"{engine}: missing startup groups {sorted(missing)}, "
                f"expected only {sorted(allowed)}"
            )
    return measurements


SEMANTICALLY_LAZY_RUNTIMES = frozenset({
    # Stitch validates eagerly but compiles each function on its first call.
    # Its public ID omits the word "lazy", so token-only filtering is wrong.
    "stitch",
})


def is_lazy_runtime(engine: str) -> bool:
    tokens = re.split(r"[^a-z0-9]+", engine.lower())
    return "lazy" in tokens or engine.lower() in SEMANTICALLY_LAZY_RUNTIMES


def build_document(
    measurements: dict[str, dict[str, float]],
    *,
    platform: str,
    candidate_sha: str,
) -> dict[str, Any]:
    nano = measurements[NANO_RUNTIME_ID]
    rows = []
    for engine, results in measurements.items():
        groups = [group for group in STARTUP_GROUPS if group in results]
        complete = len(groups) == len(STARTUP_GROUPS)
        seven_case_geomean_ns = (
            geometric_mean([results[group] for group in STARTUP_GROUPS])
            if complete
            else None
        )
        lazy = is_lazy_runtime(engine)
        comparison = None
        if engine != NANO_RUNTIME_ID and not lazy:
            common = [
                group
                for group in STARTUP_GROUPS
                if group in results and group in nano
            ]
            nano_geomean = geometric_mean([nano[group] for group in common])
            peer_geomean = geometric_mean([results[group] for group in common])
            comparison = {
                "common_workloads": common,
                "nano_faster": nano_geomean < peer_geomean,
                "peer_over_nano": peer_geomean / nano_geomean,
            }
        rows.append(
            {
                "engine": engine,
                "coverage": len(groups),
                "lazy_excluded": lazy,
                "measurements_ns": {
                    group: results[group]
                    for group in groups
                },
                "nano_comparison": comparison,
                "seven_case_geomean_ns": seven_case_geomean_ns,
            }
        )

    rows.sort(
        key=lambda row: (
            row["seven_case_geomean_ns"] is None,
            row["seven_case_geomean_ns"] or float("inf"),
            row["engine"],
        )
    )
    peers = [
        row
        for row in rows
        if row["engine"] != NANO_RUNTIME_ID and not row["lazy_excluded"]
    ]
    beaten = [
        row
        for row in peers
        if row["nano_comparison"]["nano_faster"]
    ]
    return {
        "all_non_lazy_beaten": len(beaten) == len(peers),
        "candidate_sha": candidate_sha,
        "excluded_lazy_engines": sorted(
            row["engine"] for row in rows if row["lazy_excluded"]
        ),
        "nano_beaten_peer_count": len(beaten),
        "non_lazy_peer_count": len(peers),
        "platform": platform,
        "rows": rows,
        "startup_groups": list(STARTUP_GROUPS),
        "wasmi_benchmarks_revision": WASMI_BENCHMARKS_REVISION,
    }


def format_duration(nanoseconds: float | None) -> str:
    if nanoseconds is None:
        return "N/A"
    if nanoseconds < 1_000:
        return f"{nanoseconds:.0f} ns"
    if nanoseconds < 1_000_000:
        return f"{nanoseconds / 1_000:.3f} us"
    return f"{nanoseconds / 1_000_000:.3f} ms"


def comparison_text(row: dict[str, Any]) -> str:
    comparison = row["nano_comparison"]
    if comparison is None:
        return "Nano (reference)"
    count = len(comparison["common_workloads"])
    ratio = comparison["peer_over_nano"]
    suffix = "" if count == len(STARTUP_GROUPS) else f" ({count} common cases)"
    if comparison["nano_faster"]:
        return f"YES, Nano is {ratio:.2f}x faster{suffix}"
    return f"NO, Nano takes {1.0 / ratio:.2f}x the time{suffix}"


def render_summary(document: dict[str, Any]) -> str:
    rows = document["rows"]
    non_lazy = [row for row in rows if not row["lazy_excluded"]]
    lazy = [row for row in rows if row["lazy_excluded"]]
    lines = [
        f"## Interpreter startup ranking: {document['platform']}",
        "",
        f"- candidate: `{document['candidate_sha']}`",
        f"- wasmi-benchmarks: `{document['wasmi_benchmarks_revision']}`",
        f"- field: `{len(rows)}` interpreter configurations on one runner",
        f"- workloads: `{len(STARTUP_GROUPS)}` startup cases",
        "- gate: lazy-token configurations and audited first-call compilers are excluded",
        "",
        "| Rank | Non-lazy interpreter | Coverage | 7-case geomean | Does Nano win? |",
        "|---:|---|---:|---:|---|",
    ]
    rank = 0
    for row in non_lazy:
        if row["seven_case_geomean_ns"] is not None:
            rank += 1
            rank_text = str(rank)
        else:
            rank_text = "N/A"
        lines.append(
            f"| {rank_text} | `{row['engine']}` | "
            f"{row['coverage']}/7 | "
            f"{format_duration(row['seven_case_geomean_ns'])} | "
            f"{comparison_text(row)} |"
        )

    lines.extend(
        [
            "",
            "### Excluded lazy configurations",
            "",
            "| Engine | Coverage | 7-case geomean |",
            "|---|---:|---:|",
        ]
    )
    for row in lazy:
        lines.append(
            f"| `{row['engine']}` | {row['coverage']}/7 | "
            f"{format_duration(row['seven_case_geomean_ns'])} |"
        )

    if document["all_non_lazy_beaten"]:
        verdict = (
            "PASS: Nano is faster than all "
            f"{document['non_lazy_peer_count']} non-lazy peers."
        )
    else:
        losses = [
            row["engine"]
            for row in non_lazy
            if row["engine"] != NANO_RUNTIME_ID
            and not row["nano_comparison"]["nano_faster"]
        ]
        verdict = (
            "FAIL: Nano beats "
            f"{document['nano_beaten_peer_count']}/"
            f"{document['non_lazy_peer_count']} non-lazy peers; "
            "not ahead of "
            + ", ".join(f"`{engine}`" for engine in losses)
            + "."
        )
    lines.extend(
        [
            "",
            f"**{verdict}**",
            "",
            "> `wasmtime.pulley` intentionally omits `startup/ffmpeg` in the pinned suite. "
            "Its Nano comparison uses the six common workloads and it receives no "
            "7-case geomean or rank.",
            "",
        ]
    )
    return "\n".join(lines)


def report(
    *,
    raw_dir: Path,
    platform: str,
    candidate_sha: str,
    json_out: Path,
    summary_out: Path,
) -> dict[str, Any]:
    measurements = collect_measurements(raw_dir)
    document = build_document(
        measurements,
        platform=platform,
        candidate_sha=candidate_sha,
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_out.write_text(render_summary(document), encoding="utf-8")
    return document


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--suite", type=Path, required=True)
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--target-root", type=Path, required=True)
    prepare_parser.add_argument("--metadata-out", type=Path, required=True)
    prepare_parser.add_argument("--cargo", default="cargo")
    prepare_parser.add_argument("--toolchain", default="stable")

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--suite", type=Path, required=True)
    measure_parser.add_argument("--target-root", type=Path, required=True)
    measure_parser.add_argument("--raw-dir", type=Path, required=True)
    measure_parser.add_argument("--cargo", default="cargo")
    measure_parser.add_argument("--toolchain", default="stable")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--raw-dir", type=Path, required=True)
    report_parser.add_argument("--platform", required=True)
    report_parser.add_argument("--candidate-sha", required=True)
    report_parser.add_argument("--json-out", type=Path, required=True)
    report_parser.add_argument("--summary-out", type=Path, required=True)
    report_parser.add_argument("--require-nano-fastest", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(
                suite=args.suite,
                source=args.source,
                target_root=args.target_root,
                metadata_out=args.metadata_out,
                cargo=args.cargo,
                toolchain=args.toolchain,
            )
            return 0
        if args.command == "measure":
            measure(
                suite=args.suite,
                target_root=args.target_root,
                raw_dir=args.raw_dir,
                cargo=args.cargo,
                toolchain=args.toolchain,
            )
            return 0
        document = report(
            raw_dir=args.raw_dir,
            platform=args.platform,
            candidate_sha=args.candidate_sha,
            json_out=args.json_out,
            summary_out=args.summary_out,
        )
        if args.require_nano_fastest and not document["all_non_lazy_beaten"]:
            return 1
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"startup ranking error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
