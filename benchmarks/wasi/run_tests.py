#!/usr/bin/env python3
"""Run WASI benchmark tests with sf-nano-cli and collect results.

Usage:
    python3 run_tests.py              # run full benchmark suite
    python3 run_tests.py --fast       # run with reduced workloads for CI/QEMU
    python3 run_tests.py --exec PATH  # run with a different runtime
    python3 run_tests.py --interp     # run with the interpreter engine
"""

import argparse
import hashlib
import os
import re
import subprocess
import shutil
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_CLI = os.path.join(REPO_ROOT, "target", "release", "sf-nano-cli")

# Each test has a "full" configuration and optional "fast" overrides.
# When --fast is given, the "fast" dict is merged into the test dict,
# replacing args, stdout_md5, stdout_contains, etc.
TESTS = [
    # --- Integer / Control Flow ---
    {
        "name": "coremark/coremark.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "coremark"),
        "args": ["coremark.wasm"],
        "pattern": r"Iterations/Sec\s*:\s*(\S+)",
        "source": "stdout",
        "stdout_contains": [
            "seedcrc          : 0xe9f5",
            "[0]crclist       : 0xe714",
            "[0]crcmatrix     : 0x1fd7",
            "[0]crcstate      : 0x8e3a",
        ],
    },
    {
        "name": "sha256/sha256.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "sha256"),
        "args": ["sha256.wasm"],
        "pattern": r"sha256: throughput = (\S+ MB/s)",
        "source": "stdout",
        "stdout_contains": "hash = d824bde2acff5ab2",
    },
    {
        "name": "bzip2/bzip2.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "bzip2"),
        "args": ["bzip2.wasm"],
        "pattern": r"bzip2: throughput = (\S+ MB/s)",
        "source": "stdout",
    },
    {
        "name": "lz4/lz4.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "lz4"),
        "args": ["lz4.wasm"],
        "pattern": r"(lz4 (?:compress|decompress): throughput = \S+ MB/s)",
        "source": "stdout",
        "multi": True,
    },
    # --- Lua ---
    {
        "name": "lua/fib",
        "cwd": os.path.join(SCRIPT_DIR, "lua"),
        "args": ["lua.wasm", "fib_38.lua"],
        "pattern": None,
        "source": None,
        "stdout_contains": "fib(38) = 39088169",
        "fast": {
            "args": ["lua.wasm", "fib_34.lua"],
            "stdout_contains": "fib(34) = 5702887",
        },
    },
    {
        "name": "lua/sunfish",
        "cwd": os.path.join(SCRIPT_DIR, "lua"),
        "args": ["lua.wasm", "sunfish.lua"],
        "pattern": r"Score:\s+(\S+)",
        "source": "stdout",
        "fast": {
            "args": ["lua.wasm", "sunfish.lua", "10"],
        },
    },
    {
        "name": "lua/json_bench",
        "cwd": os.path.join(SCRIPT_DIR, "lua"),
        "args": ["lua.wasm", "json_bench.lua"],
        "pattern": r"Score:\s+(\S+)",
        "source": "stdout",
        "fast": {
            "args": ["lua.wasm", "json_bench.lua", "5", "50"],
        },
    },
    # --- Floating Point ---
    {
        "name": "mandelbrot/mandel.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "mandelbrot"),
        "args": ["mandel.wasm", "128", "6e6"],
        "pattern": r"Elapsed time:\s*(.+)",
        "source": "stderr",
        "stdout_md5": "a254de76a7cc668adb5464ab66517c5c",
        "fast": {
            "args": ["mandel.wasm", "768", "2000"],
            "stdout_md5": "7912f0d441822be136ce9c8bafa9641b",
        },
    },
    {
        "name": "c-ray/c-ray.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "c-ray"),
        "args": ["c-ray.wasm", "-s", "4000x4000"],
        "stdin": os.path.join(SCRIPT_DIR, "c-ray", "scene"),
        "pattern": r"Rendering took:\s*(.+\))",
        "source": "stderr",
        "stdout_md5": "d02c3bc410e3ba820688599794aa3e9d",
        "fast": {
            "args": ["c-ray.wasm", "-s", "1600x1200"],
            "stdout_md5": "01e7d76f49aaa72ac00c276e7f3eb031",
        },
    },
    # --- Memory Bound ---
    {
        "name": "stream/stream.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "stream"),
        "args": ["stream.wasm"],
        "pattern": r"(Copy|Scale|Add|Triad):\s+(\S+)",
        "source": "stdout",
        "multi": True,
        "stdout_contains": "Solution Validates",
    },
    # --- Database ---
    {
        "name": "sqlite/speedtest1.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "sqlite"),
        "cli_args": ["--dir", "."],
        "args": ["speedtest1.wasm", "--memdb", "--nosync", "--journal", "off",
                 "--size", "1000", "--testset", "main"],
        "pattern": r"^\s*(TOTAL\.+\s+\S+s)",
        "source": "stdout",
        "fast": {
            "args": ["speedtest1.wasm", "--memdb", "--nosync", "--journal", "off",
                     "--size", "10", "--testset", "main"],
        },
    },
]


def resolve_test(test, fast):
    """Return a merged test dict when --fast is given."""
    if not fast:
        return test
    overrides = test.get("fast")
    if not overrides:
        return test
    merged = dict(test)
    merged.update(overrides)
    del merged["fast"]
    return merged


def run_test(cli, test, cli_extra=()):
    name = test["name"]
    cwd = test["cwd"]
    cmd = [cli] + cli_extra + test.get("cli_args", []) + test["args"]
    stdin_file = test.get("stdin")
    pattern = test.get("pattern")
    source = test.get("source")
    multi = test.get("multi", False)
    expected = test.get("expected")

    wasm_path = os.path.join(cwd, test["args"][0])
    if not os.path.exists(wasm_path):
        return name, "SKIP", "wasm file not found", None

    if stdin_file and not os.path.exists(stdin_file):
        return name, "SKIP", f"missing input: {os.path.basename(stdin_file)}", None

    stdin_data = None
    if stdin_file:
        with open(stdin_file, "rb") as f:
            stdin_data = f.read()
        repeat = test.get("stdin_repeat", 1)
        if repeat > 1:
            stdin_data = stdin_data * repeat

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            input=stdin_data,
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return name, "FAIL", "timeout (300s)", None
    except Exception as e:
        return name, "FAIL", str(e), None
    elapsed = time.monotonic() - t0

    exit_code = proc.returncode
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")

    # Any nonzero exit is a failure, even if the process printed benchmark lines.
    combined = stdout + stderr
    if exit_code != 0:
        for line in combined.splitlines():
            if "Error" in line:
                return name, "FAIL", line.strip(), elapsed
        return name, "FAIL", f"exit code {exit_code}", elapsed

    # Verify deterministic output if configured.
    stdout_contains = test.get("stdout_contains")
    if stdout_contains:
        if isinstance(stdout_contains, str):
            stdout_contains = [stdout_contains]
        for needle in stdout_contains:
            if needle not in stdout:
                return name, "FAIL", f"expected stdout to contain '{needle}'", elapsed
    stdout_md5 = test.get("stdout_md5")
    if stdout_md5:
        actual_md5 = hashlib.md5(proc.stdout).hexdigest()
        if actual_md5 != stdout_md5:
            return name, "FAIL", f"stdout md5 {actual_md5} != {stdout_md5}", elapsed

    # Extract compile stats if present: [arch] (func:N, ssa:N, mir:N, code:N)
    native_info = ""
    for line in stderr.splitlines():
        stripped = line.strip()
        if "(func:" in stripped and "ssa:" in stripped:
            native_info = "  " + stripped
            break

    # Extract metric
    if pattern is None:
        return name, "PASS", f"{elapsed:.3f}s (wall clock){native_info}", elapsed

    text = stderr if source == "stderr" else stdout
    if multi:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            sep = test.get("separator", "; ")
            if isinstance(matches[0], tuple):
                metric = ", ".join(f"{m[0]}: {m[1]} MB/s" for m in matches)
            else:
                metric = sep.join(m.strip() for m in matches)
            return name, "PASS", f"{metric}{native_info}", elapsed
    else:
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            metric = m.group(1).strip()
            if expected and metric != expected:
                return name, "FAIL", f"expected {expected}, got {metric}", elapsed
            return name, "PASS", f"{metric}{native_info}", elapsed

    if exit_code == 0:
        return name, "PASS", f"{elapsed:.3f}s (no metric found){native_info}", elapsed
    return name, "FAIL", f"exit code {exit_code}", elapsed


def main():
    parser = argparse.ArgumentParser(description="Run WASI benchmark tests")
    parser.add_argument("--exec", dest="cli", default=DEFAULT_CLI,
                        help="Path to the WASM runtime executable")
    parser.add_argument("--cli-args", default="",
                        help="Extra args for the runtime (e.g. '--dir .' for wasmtime)")
    parser.add_argument("--fast", action="store_true",
                        help="Use reduced workloads for CI/QEMU iteration")
    parser.add_argument("--interp", action="store_true",
                        help="Run with the interpreter (passes --interp to sf-nano-cli)")
    args = parser.parse_args()

    cli_parts = args.cli.split()
    cli_parts[0] = os.path.expanduser(cli_parts[0])
    cli = cli_parts[0]
    cli_extra = cli_parts[1:] + (args.cli_args.split() if args.cli_args else [])
    if args.interp:
        cli_extra = ["--interp"] + cli_extra
    if not os.path.exists(cli) and not shutil.which(cli):
        print(f"ERROR: runtime not found: {cli}", file=sys.stderr)
        if cli == DEFAULT_CLI:
            print("Run: cargo build --release --bin sf-nano-cli", file=sys.stderr)
        sys.exit(1)

    mode = "fast" if args.fast else "full"
    if args.interp:
        mode += ", interp"
    print(f"Runtime: {' '.join(cli_parts)} ({mode} mode)")
    print()

    tests = [resolve_test(t, args.fast) for t in TESTS]
    results = []
    total = len(tests)
    for i, test in enumerate(tests, 1):
        name = test["name"]
        print(f"[{i}/{total}] {name} ...", end=" ", flush=True)
        result = run_test(cli, test, cli_extra)
        results.append(result)
        _, status, metric, elapsed = result
        elapsed_str = f"{elapsed:.1f}s" if elapsed else ""
        print(f"{status}  {elapsed_str}  {metric}")

    # Summary
    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    failed = sum(1 for _, s, _, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _, _ in results if s == "SKIP")

    print()
    print("=" * 72)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped"
          f" / {total} total")
    print("=" * 72)
    print()
    print(f"{'Test':<35} {'Status':<6} {'Metric'}")
    print("-" * 72)
    for name, status, metric, _ in results:
        print(f"{name:<35} {status:<6} {metric}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
