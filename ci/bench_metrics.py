"""Normalize the metric strings emitted by the in-tree WASI benchmarks."""

import json
import re
from pathlib import Path

# Each entry maps a TESTS[*]["name"] to a list of
# (label, regex, unit, direction) tuples. The regex is applied to the metric
# string returned by run_tests.run_test; group(1) is parsed as float.
#
# Every active performance test must have an entry. ci.test_performance keeps
# this mapping in lockstep with run_tests.TESTS.
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
    "funcref/funcref.wasm": [
        (
            "funcref-exported-table",
            r"funcref exported-table: rate = ([\d.]+)",
            "calls/s",
            "higher",
        ),
        (
            "funcref-direct",
            r"funcref direct: rate = ([\d.]+)",
            "calls/s",
            "higher",
        ),
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
    "sqlite/sqlite_bench.wasm": [
        (
            "sqlite",
            r"^([\d.]+)\s*iteration/s",
            "iteration/s",
            "higher",
        ),
    ],
}


def extract_metrics(test_name: str, metric_str: str) -> list[dict]:
    """Parse one run_tests metric string into normalized metric entries."""
    entries: list[dict] = []
    for label, regex, unit, direction in METRIC_EXTRACTORS.get(test_name, []):
        match = re.search(regex, metric_str)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        entries.append({
            "name": label,
            "unit": unit,
            "direction": direction,
            "value": value,
        })
    return entries


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
