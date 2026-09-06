"""Nano-only paired execution experiment; ordinary regression CI remains the gate."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from ci.performance_stats import probability_summary, geometric_mean


def score_from_output(output: str) -> float:
    scores = re.findall(r"result=\[F32\(([^)]+)\)\]", output)
    if len(scores) != 1:
        raise ValueError(f"expected exactly one CoreMark result, got {len(scores)}")
    score = float(scores[0])
    if not math.isfinite(score) or score <= 0:
        raise ValueError(f"invalid CoreMark score: {score}")
    return score


def throughput_from_output(output: str) -> float:
    measurements = re.findall(r"^measurement runs=(\d+) elapsed_ns=(\d+)$", output, re.MULTILINE)
    if len(measurements) != 1:
        raise ValueError(f"expected exactly one execution measurement, got {len(measurements)}")
    runs, elapsed_ns = map(int, measurements[0])
    if runs <= 0 or elapsed_ns <= 0:
        raise ValueError("execution requires positive work and duration")
    return runs * 1e9 / elapsed_ns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--wasm", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--mode", choices=["coremark", "setup"], default="coremark")
    parser.add_argument("--input", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--expected-output-f64", type=float)
    args = parser.parse_args()
    variants = dict(v.split("=", 1) for v in args.variant)
    if len(variants) != len(args.variant) or len(variants) < 2 or args.rounds < 2:
        parser.error("need distinct variants and at least two rounds")
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error("execution duration must be positive and at most 60 seconds")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    scores = {name: [] for name in variants}
    records = []
    for round_index in range(args.rounds):
        order = list(variants)
        if round_index % 2:
            order.reverse()
        for name in order:
            command = [variants[name], str(args.wasm.resolve()), args.mode]
            if args.mode == "setup":
                command += [str(args.input), str(args.seconds)]
                if args.expected_output_f64 is not None:
                    command += [str(args.expected_output_f64)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            output = result.stdout + result.stderr
            (out / f"round-{round_index}-{name}.log").write_text(output)
            if result.returncode:
                raise RuntimeError(f"{name} exited {result.returncode}: {output}")
            score = (score_from_output(output) if args.mode == "coremark"
                     else throughput_from_output(output))
            scores[name].append(score)
            records.append({"round": round_index, "variant": name, "score": score})
            print(f"round={round_index} variant={name} score={score}", flush=True)
    baseline = next(iter(variants))
    comparisons = {}
    for name in list(variants)[1:]:
        comparisons[f"{name}/{baseline}"] = probability_summary([
            candidate/base for candidate, base in zip(scores[name], scores[baseline])])
    for left, right in zip(list(variants)[1:], list(variants)[2:]):
        comparisons[f"{right}/{left}"] = probability_summary([
            b/a for a, b in zip(scores[left], scores[right])])
    doc = {"mode": args.mode, "input": args.input, "seconds": args.seconds,
           "expected_output_f64": args.expected_output_f64,
           "scores": scores, "schedule": records, "comparisons": comparisons}
    (out / "comparison.json").write_text(json.dumps(doc, indent=2) + "\n")
    title = "CoreMark" if args.mode == "coremark" else args.wasm.parent.name
    unit = "score" if args.mode == "coremark" else "runs/second"
    lines = [f"## Nano-only wasmi {title} experiment", "",
             f"{args.rounds} process pairs per comparison, alternating variant order; score ratios.", "",
             f"| Variant | Geomean {unit} |", "|---|---:|"]
    for name, values in scores.items():
        lines.append(f"| {name} | {geometric_mean(values):.2f} |")
    lines += ["", "| Comparison | Throughput change | P(improvement) | P(regression) |",
              "|---|---:|---:|---:|"]
    for name, row in comparisons.items():
        delta = math.expm1(row["mean_log_ratio"]) * 100
        lines.append(f'| {name} | {delta:+.2f}% | {row["probability_improvement"]:.6%} | {row["probability_regression"]:.6%} |')
    lines += ["", "These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.", ""]
    summary = "\n".join(lines)
    (out / "summary.md").write_text(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
