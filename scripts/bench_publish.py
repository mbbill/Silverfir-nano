#!/usr/bin/env python3
"""Run WASI benchmarks and emit benchmark-action JSON files.

Imports benchmarks/wasi/run_tests.py and runs each test exactly as a local
run does -- the benchmarks are self-timing, so there is no CI-only reduced
mode -- then converts each test's metric string into benchmark-action's
customBigger / customSmaller JSON format. Two files are produced in
--out-dir:

  higher.json -- metrics where bigger is better (Iter/s, MB/s, Score)
  lower.json  -- metrics where smaller is better (seconds)

Usage:
  python3 scripts/bench_publish.py \\
      --exec "target/release/sf-nano-cli" \\
      --out-dir bench-output/
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "wasi"))

import run_tests  # noqa: E402


# Per-test metric extractors. Each entry maps a TESTS[*]["name"] to a list of
# (label, regex, unit, direction) tuples. The regex is applied to the metric
# string returned by run_tests.run_test; group(1) is parsed as float.
#
# Test names not listed here are simply skipped.
METRIC_EXTRACTORS = {
    "coremark/coremark.wasm": [
        ("coremark", r"^([\d.]+)", "Iter/s", "higher"),
    ],
    "sha256/sha256.wasm": [
        ("sha256", r"^([\d.]+)\s*MB/s", "MB/s", "higher"),
    ],
    "bzip2/bzip2.wasm": [
        ("bzip2", r"^([\d.]+)\s*MB/s", "MB/s", "higher"),
    ],
    "lz4/lz4.wasm": [
        ("lz4-compress", r"lz4 compress: throughput = ([\d.]+)", "MB/s", "higher"),
        ("lz4-decompress", r"lz4 decompress: throughput = ([\d.]+)", "MB/s", "higher"),
    ],
    "lua/fib": [
        ("lua-fib", r"^([\d.]+)\s*fib20/s", "fib20/s", "higher"),
    ],
    "lua/sunfish": [
        ("lua-sunfish", r"^([\d.]+)", "Score", "higher"),
    ],
    "lua/json_bench": [
        ("lua-json", r"^([\d.]+)", "Score", "higher"),
    ],
    "mandelbrot/mandel.wasm": [
        ("mandelbrot", r"^([\d.]+)\s*Kpixel/s", "Kpixel/s", "higher"),
    ],
    "c-ray/c-ray.wasm": [
        ("c-ray", r"^([\d.]+)\s*Kpixel/s", "Kpixel/s", "higher"),
    ],
    "stream/stream.wasm": [
        ("stream-Copy", r"Copy:\s*([\d.]+)", "MB/s", "higher"),
        ("stream-Scale", r"Scale:\s*([\d.]+)", "MB/s", "higher"),
        ("stream-Add", r"Add:\s*([\d.]+)", "MB/s", "higher"),
        ("stream-Triad", r"Triad:\s*([\d.]+)", "MB/s", "higher"),
    ],
    "sqlite/speedtest1.wasm": [
        ("sqlite", r"^([\d.]+)\s*size/s", "size/s", "higher"),
    ],
}


def write_json(path: Path, entries: list[dict]) -> None:
    """Write one result file with LF endings on every host.

    `write_text` uses universal newlines, so on Windows it turns the
    trailing "\\n" into "\\r\\n". That made the same empty result read as
    `[]` on Linux and `[]\\r` through the publish workflow's
    `"$(cat ...)" != "[]"` guard -- command substitution strips the LF but
    keeps the CR -- so the Windows row alone got past the guard and handed
    an empty array to the benchmark action, which fails on it.
    """
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entries, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exec", required=True, help="Runtime command (may contain spaces, e.g. 'qemu-arm-static -cpu cortex-a15 path/to/sf-nano-cli')")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exec_parts = args.exec.split()
    # run_tests.run_test() invokes the subprocess with cwd=<benchmark dir>, so
    # relative binary paths break. If the first token names a real file, lock
    # it to an absolute path; otherwise (e.g. "qemu-arm-static <abs-path>"),
    # assume PATH lookup is intended and only absolutize the trailing binary.
    if Path(exec_parts[0]).exists():
        exec_parts[0] = str(Path(exec_parts[0]).resolve())
    else:
        for i in range(1, len(exec_parts)):
            if Path(exec_parts[i]).exists():
                exec_parts[i] = str(Path(exec_parts[i]).resolve())
                break
    cli = exec_parts[0]
    cli_extra = exec_parts[1:]

    higher: list[dict] = []
    lower: list[dict] = []
    failed: list[str] = []

    tests = run_tests.TESTS
    for i, test in enumerate(tests, 1):
        name = test["name"]
        print(f"[{i}/{len(tests)}] {name} ...", end=" ", flush=True)
        _, status, metric_str, _ = run_tests.run_test(cli, test, cli_extra)
        print(f"{status}  {metric_str}")

        if status == "FAIL":
            failed.append(f"{name}: {metric_str}")
            continue
        if status != "PASS":
            continue
        if name not in METRIC_EXTRACTORS:
            continue

        for label, regex, unit, direction in METRIC_EXTRACTORS[name]:
            m = re.search(regex, metric_str)
            if not m:
                print(f"  warn: pattern {regex!r} did not match {metric_str!r}")
                continue
            try:
                value = float(m.group(1))
            except ValueError:
                print(f"  warn: non-numeric value {m.group(1)!r}")
                continue
            entry = {"name": label, "unit": unit, "value": value}
            (higher if direction == "higher" else lower).append(entry)

    write_json(args.out_dir / "higher.json", higher)
    write_json(args.out_dir / "lower.json", lower)
    print()
    print(f"Wrote {len(higher)} higher-is-better and {len(lower)} lower-is-better metrics to {args.out_dir}/")

    if failed:
        print()
        print("Failed tests (results from these are missing):")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
