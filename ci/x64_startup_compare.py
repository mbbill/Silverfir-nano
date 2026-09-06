"""Temporary Nano-only startup isolation; full dev CI remains the gate."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
from ci.performance_stats import probability_summary, geometric_mean
from ci.x64_coremark_compare import throughput_from_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', action='append', required=True)
    parser.add_argument('--wasm', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=3)
    args = parser.parse_args()
    variants = dict(item.split('=', 1) for item in args.variant)
    if len(variants) != len(args.variant) or len(variants) < 2 or args.rounds < 2:
        parser.error('need distinct variants and at least two rounds')
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error('duration must be positive and at most 60 seconds')
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    scores = {name: [] for name in variants}
    records = []
    for round_index in range(args.rounds):
        order = list(variants)
        if round_index % 2:
            order.reverse()
        for name in order:
            command = [variants[name], str(args.wasm.resolve()), 'startup', '0', str(args.seconds)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            output = result.stdout + result.stderr
            (out / f'round-{round_index}-{name}.log').write_text(output)
            if result.returncode:
                raise RuntimeError(f'{name} exited {result.returncode}: {output}')
            throughput = throughput_from_output(output)
            scores[name].append(throughput)
            records.append({'round': round_index, 'variant': name, 'command': command,
                            'instantiations_per_second': throughput})
            print(records[-1], flush=True)
    baseline = next(iter(variants))
    pairs = [(baseline, name) for name in list(variants)[1:]]
    pairs += list(zip(list(variants)[1:], list(variants)[2:]))
    # The ablations compare directly with current, independent of list order.
    if 'current' in variants:
        pairs += [('current', name) for name in ['nocache', 'nodse'] if name in variants]
    comparisons = {f'{right}/{left}': probability_summary([
        b/a for a, b in zip(scores[left], scores[right])]) for left, right in pairs}
    (out / 'comparison.json').write_text(json.dumps({
        'mode': 'startup', 'seconds': args.seconds, 'scores': scores,
        'schedule': records, 'comparisons': comparisons}, indent=2) + '\n')
    lines = [f'## Nano-only startup: {args.wasm.stem}', '',
             f'{args.rounds} rounds, alternating process order. Instantiation and destruction are timed.', '',
             '| Variant | Instantiations/s |', '|---|---:|']
    lines += [f'| {name} | {geometric_mean(values):.5f} |' for name, values in scores.items()]
    lines += ['', '| Comparison | Throughput change | P(improvement) | P(regression) |', '|---|---:|---:|---:|']
    for name, row in comparisons.items():
        delta = math.expm1(row['mean_log_ratio']) * 100
        lines.append(f'| {name} | {delta:+.2f}% | {row["probability_improvement"]:.6%} | {row["probability_regression"]:.6%} |')
    lines += ['', 'Diagnostic estimates; full dev startup CI remains the gate.', '']
    summary = '\n'.join(lines)
    (out / 'summary.md').write_text(summary)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(summary)
    print(summary)


if __name__ == '__main__':
    main()
