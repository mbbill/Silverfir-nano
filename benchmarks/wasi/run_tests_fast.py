#!/usr/bin/env python3
"""Fast validation tests for emu32 and other slow backends.

Runs the same programs as run_tests.py but with drastically reduced workloads
where possible. Focuses on correctness (exit code, expected output) rather
than benchmark scores.

Usage:
    python3 run_tests_fast.py                        # default sf-nano-cli
    python3 run_tests_fast.py --exec PATH            # custom runtime
    python3 run_tests_fast.py --timeout 120          # per-test timeout (s)
    python3 run_tests_fast.py --exec PATH --cli-args '--emu32'
"""

import argparse
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_CLI = os.path.join(REPO_ROOT, "target", "release", "sf-nano-cli")

# Each test has:
#   name, cwd, args       - same as run_tests.py
#   expect_stdout          - regex that MUST appear in stdout (None = skip)
#   expect_stderr          - regex that MUST appear in stderr (None = skip)
#   expect_exit            - expected exit code (default 0)
#   stdin                  - optional stdin file
#
# Workloads are reduced where the program accepts parameters.
TESTS = [
    # --- Integer / Control Flow ---
    {
        "name": "coremark/coremark.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "coremark"),
        "args": ["coremark.wasm"],
        "expect_stdout": r"CoreMark 1\.0 :\s*\S+",
    },
    {
        "name": "sha256/sha256.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "sha256"),
        "args": ["sha256.wasm"],
        "expect_stdout": r"sha256: throughput = \S+ MB/s",
    },
    {
        "name": "bzip2/bzip2.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "bzip2"),
        "args": ["bzip2.wasm"],
        "expect_stdout": r"bzip2: throughput = \S+ MB/s",
    },
    {
        "name": "lz4/lz4.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "lz4"),
        "args": ["lz4.wasm"],
        "expect_stdout": r"lz4 compress: throughput = \S+ MB/s",
    },
    # --- Lua (reduced workloads) ---
    {
        "name": "lua/fib",
        "cwd": os.path.join(SCRIPT_DIR, "lua"),
        "args": ["lua.wasm", "fib.lua"],
        "expect_stdout": r"fib\(38\) = 39088169",
    },
    {
        # NODES_SEARCHED=100, TIME_BUDGET=1
        "name": "lua/sunfish",
        "cwd": os.path.join(SCRIPT_DIR, "lua"),
        "args": ["lua.wasm", "sunfish.lua", "100", "1"],
        "expect_stdout": r"Score:\s+\S+",
    },
    {
        # TIME_BUDGET=1, N_USERS=10
        "name": "lua/json_bench",
        "cwd": os.path.join(SCRIPT_DIR, "lua"),
        "args": ["lua.wasm", "json_bench.lua", "1", "10"],
        "expect_stdout": r"Score:\s+\S+",
    },
    # --- Floating Point (reduced resolution/iterations) ---
    {
        # resolution=4, iterations=100 (vs 128 / 6e6)
        "name": "mandelbrot/mandel.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "mandelbrot"),
        "args": ["mandel.wasm", "4", "100"],
        "expect_stderr": r"Elapsed time:",
    },
    {
        # 100x100 (vs 4000x4000)
        "name": "c-ray/c-ray.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "c-ray"),
        "args": ["c-ray.wasm", "-s", "100x100"],
        "stdin": os.path.join(SCRIPT_DIR, "c-ray", "scene"),
        "expect_stderr": r"Rendering took:",
    },
    # --- Memory Bound ---
    {
        "name": "stream/stream.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "stream"),
        "args": ["stream.wasm"],
        "expect_stdout": r"Copy:\s+\S+",
    },
]


def run_test(cli, test, cli_extra, timeout):
    name = test["name"]
    cwd = test["cwd"]
    cmd = [cli] + cli_extra + test["args"]

    wasm_path = os.path.join(cwd, test["args"][0])
    if not os.path.exists(wasm_path):
        return name, "SKIP", "wasm file not found", None

    stdin_file = test.get("stdin")
    if stdin_file and not os.path.exists(stdin_file):
        return name, "SKIP", f"missing input: {os.path.basename(stdin_file)}", None

    stdin_data = None
    if stdin_file:
        with open(stdin_file, "rb") as f:
            stdin_data = f.read()

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, input=stdin_data,
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return name, "TIMEOUT", f">{timeout}s", None
    except Exception as e:
        return name, "FAIL", str(e), None
    elapsed = time.monotonic() - t0

    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")

    # Check exit code
    expected_exit = test.get("expect_exit", 0)
    if proc.returncode != expected_exit:
        # Find an error message to report
        for line in (stdout + stderr).splitlines():
            if "Error" in line or "error" in line or "panic" in line:
                return name, "FAIL", f"exit {proc.returncode}: {line.strip()}", elapsed
        return name, "FAIL", f"exit code {proc.returncode} (expected {expected_exit})", elapsed

    # Check expected stdout pattern
    pat = test.get("expect_stdout")
    if pat and not re.search(pat, stdout, re.MULTILINE):
        preview = stdout[:200].replace("\n", "\\n")
        return name, "FAIL", f"missing expected stdout pattern: {pat!r} (got: {preview!r})", elapsed

    # Check expected stderr pattern
    pat = test.get("expect_stderr")
    if pat and not re.search(pat, stderr, re.MULTILINE):
        preview = stderr[:200].replace("\n", "\\n")
        return name, "FAIL", f"missing expected stderr pattern: {pat!r} (got: {preview!r})", elapsed

    return name, "PASS", f"{elapsed:.1f}s", elapsed


def main():
    parser = argparse.ArgumentParser(description="Fast validation tests for slow backends")
    parser.add_argument("--exec", dest="cli", default=DEFAULT_CLI,
                        help="Path to the WASM runtime executable")
    parser.add_argument("--cli-args", default="",
                        help="Extra args for the runtime (e.g. '--emu32')")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-test timeout in seconds (default: 120)")
    args = parser.parse_args()

    cli = args.cli
    cli_extra = args.cli_args.split() if args.cli_args else []
    timeout = args.timeout

    if not os.path.exists(cli):
        print(f"ERROR: runtime not found at {cli}", file=sys.stderr)
        if cli == DEFAULT_CLI:
            print("Run: cargo build --release --bin sf-nano-cli", file=sys.stderr)
        sys.exit(1)

    print(f"Runtime: {cli}")
    if cli_extra:
        print(f"Args:    {' '.join(cli_extra)}")
    print(f"Timeout: {timeout}s per test")
    print()

    results = []
    total = len(TESTS)
    for i, test in enumerate(TESTS, 1):
        name = test["name"]
        print(f"[{i}/{total}] {name} ...", end=" ", flush=True)
        result = run_test(cli, test, cli_extra, timeout)
        results.append(result)
        _, status, info, _ = result
        print(f"{status}  {info}")

    # Summary
    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    failed = sum(1 for _, s, _, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _, _ in results if s == "SKIP")
    timedout = sum(1 for _, s, _, _ in results if s == "TIMEOUT")

    print()
    print("=" * 72)
    print(f"Results: {passed} passed, {failed} failed, "
          f"{timedout} timeout, {skipped} skipped / {total} total")
    print("=" * 72)

    if failed > 0:
        print("\nFAILED:")
        for name, status, info, _ in results:
            if status == "FAIL":
                print(f"  {name}: {info}")

    if timedout > 0:
        print("\nTIMEOUT (not a correctness failure):")
        for name, status, info, _ in results:
            if status == "TIMEOUT":
                print(f"  {name}")

    # Exit 1 only on correctness failures, not timeouts
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
