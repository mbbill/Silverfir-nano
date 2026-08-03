"""Cross-engine WASI-suite standings for the x64 tuning branch.

Branch-only tooling for .github/workflows/x64-standings.yml: drives
benchmarks/wasi for Silverfir-nano JIT (via run_tests.py), a prebuilt
wasmtime (Cranelift), and V8 (via run_v8.mjs under Node), then renders
one within-run comparison table. The suite reports rates, so higher is
better and the gap column is engine/nano. No statistics and no verdict —
a snapshot instrument, deleted together with its workflow before merge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_DIR = REPO_ROOT / "benchmarks" / "wasi"

FLOAT = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
# run_tests.py appends compile stats to the metric when the runtime
# reports them: "  [x86_64] (func:N, ssa:N, ...)". Not part of the rate.
COMPILE_INFO = re.compile(r"\s+\[\w+\] \(func:.*$")


def load_run_tests():
    spec = importlib.util.spec_from_file_location(
        "wasi_run_tests", SUITE_DIR / "run_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def short_name(name: str) -> str:
    head, _, tail = name.partition("/")
    return head if tail.endswith(".wasm") else name


def parse_metric(bench: str, metric: str) -> dict[str, tuple[float, str]]:
    """Split one PASS metric string into {metric_key: (rate, unit)}."""
    metric = COMPILE_INFO.sub("", metric).strip()
    parts = re.split(r"[;,] ", metric)
    out: dict[str, tuple[float, str]] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        label, colon, value_text = part.partition(":")
        if not colon:
            label, value_text = "", part
        match = FLOAT.search(value_text)
        if not match:
            continue
        unit = value_text[match.end():].strip()
        label = label.strip()
        prefix = bench.split("/")[-1] + " "
        if label.lower().startswith(prefix.lower()):
            label = label[len(prefix):]
        key = f"{bench}/{label}" if label else bench
        out[key] = (float(match.group(0)), unit)
    return out


def run_native(run_tests, cli: str, cli_extra: tuple[str, ...], seconds: float):
    metrics: dict[str, tuple[float, str]] = {}
    failures: list[str] = []
    for test in run_tests.TESTS:
        name, status, metric, _ = run_tests.run_test(
            cli, test, cli_extra=cli_extra, time_target=seconds
        )
        bench = short_name(name)
        print(f"  {name}: {status}  {metric}", flush=True)
        if status != "PASS":
            failures.append(f"{bench}: {status} ({metric})")
            continue
        metrics.update(parse_metric(bench, metric))
    return metrics, failures


def run_v8(node: str, seconds: float):
    proc = subprocess.run(
        [node, "run_v8.mjs", "--time", str(seconds)],
        cwd=SUITE_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    sys.stdout.write(proc.stdout)
    metrics: dict[str, tuple[float, str]] = {}
    failures: list[str] = []
    in_table = False
    for line in proc.stdout.splitlines():
        if line.startswith("----"):
            in_table = True
            continue
        if not in_table or not line.strip():
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        name, status, metric = parts
        bench = short_name(name)
        if status != "PASS":
            failures.append(f"{bench}: {status} ({metric})")
            continue
        metrics.update(parse_metric(bench, metric))
    return metrics, failures


def geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def fmt_rate(value: float, unit: str) -> str:
    text = f"{value:,.0f}" if value >= 1000 else f"{value:g}"
    return f"{text} {unit}".strip()


def render(
    runtimes: dict[str, dict[str, tuple[float, str]]],
    nano: str,
    notes: list[str],
    failures: list[str],
) -> str:
    others = [name for name in runtimes if name != nano]
    metrics = sorted({key for rates in runtimes.values() for key in rates})

    def gap(key: str, other: str) -> float | None:
        if key in runtimes[nano] and key in runtimes[other]:
            return runtimes[other][key][0] / runtimes[nano][key][0]
        return None

    def worst(key: str) -> float:
        return max(
            (gap(key, other) or 0.0 for other in others), default=0.0
        )

    lines = ["# x64 WASI-suite standings", ""]
    lines += [f"- {note}" for note in notes]
    lines += [
        "",
        "Rates — higher is better. The gap column `<engine>/nano` above "
        "1.00 means that engine is faster than nano on that row. Rows "
        "are sorted widest gap first. Engines ran suite-by-suite, not "
        "interleaved.",
        "",
    ]
    header = ["metric", nano] + others + [f"{name}/nano" for name in others]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    ratios: dict[str, list[float]] = {name: [] for name in others}
    for key in sorted(metrics, key=worst, reverse=True):
        cells = [key]
        for name in [nano] + others:
            if key in runtimes[name]:
                cells.append(fmt_rate(*runtimes[name][key]))
            else:
                cells.append("—")
        for name in others:
            ratio = gap(key, name)
            if ratio is None:
                cells.append("—")
            else:
                ratios[name].append(ratio)
                cells.append(f"{ratio:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    geo_cells = ["**geomean**", ""] + [""] * len(others)
    for name in others:
        mean = geomean(ratios[name])
        geo_cells.append(f"**{mean:.2f}**" if mean is not None else "—")
    lines.append("| " + " | ".join(geo_cells) + " |")
    lines.append("")
    if failures:
        lines.append("Rows missing above:")
        lines += [f"- {failure}" for failure in failures]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nano", required=True)
    parser.add_argument("--wasmtime", required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--time", type=float, default=2.0)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args(argv)

    run_tests = load_run_tests()
    runtimes: dict[str, dict[str, tuple[float, str]]] = {}
    failures: list[str] = []

    # run_test spawns each benchmark with the benchmark's directory as
    # cwd, so the runtime paths must survive that change.
    nano = str(Path(args.nano).resolve())
    wasmtime = str(Path(args.wasmtime).resolve())

    print("== silverfir-nano.jit ==", flush=True)
    runtimes["silverfir-nano.jit"], failed = run_native(
        run_tests, nano, (), args.time
    )
    failures += [f"silverfir-nano.jit {entry}" for entry in failed]

    print("== wasmtime.cranelift ==", flush=True)
    runtimes["wasmtime.cranelift"], failed = run_native(
        run_tests, wasmtime, ("run", "--dir", "."), args.time
    )
    failures += [f"wasmtime.cranelift {entry}" for entry in failed]

    print("== v8 (node) ==", flush=True)
    runtimes["v8"], failed = run_v8(args.node, args.time)
    failures += [f"v8 {entry}" for entry in failed]

    empty = [name for name, rates in runtimes.items() if not rates]
    if empty:
        raise SystemExit(f"no metrics from: {', '.join(empty)}")

    report = render(runtimes, "silverfir-nano.jit", args.note, failures)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report, encoding="utf-8")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "notes": args.note,
                "failures": failures,
                "rates": {
                    name: {key: value[0] for key, value in rates.items()}
                    for name, rates in runtimes.items()
                },
            },
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
