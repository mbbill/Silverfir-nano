#!/usr/bin/env python3
"""Run WASI benchmark tests with sf-nano-cli and collect results.

Usage:
    python3 run_tests.py              # run with sf-nano-cli
    python3 run_tests.py --exec PATH  # run with a different runtime
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

# Parameters are tuned so each test runs ~3-10 s on a typical machine.
# simple: fib(20) is hardcoded, can't increase (~0.5s).
TESTS = [
    # --- Integer / Control Flow ---
    {
        "name": "coremark/coremark.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "coremark"),
        "args": ["coremark.wasm"],
        "pattern": r"Iterations/Sec\s*:\s*(\S+)",
        "source": "stdout",
    },
    {
        "name": "sha256/sha256.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "sha256"),
        "args": ["sha256.wasm"],
        "pattern": r"sha256: throughput = (\S+ MB/s)",
        "source": "stdout",
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
        "args": ["lua.wasm", "fib.lua"],
        "pattern": None,
        "source": None,
    },
    {
        "name": "lua/sunfish",
        "cwd": os.path.join(SCRIPT_DIR, "lua"),
        "args": ["lua.wasm", "sunfish.lua"],
        "pattern": r"Score:\s+(\S+)",
        "source": "stdout",
    },
    {
        "name": "lua/json_bench",
        "cwd": os.path.join(SCRIPT_DIR, "lua"),
        "args": ["lua.wasm", "json_bench.lua"],
        "pattern": r"Score:\s+(\S+)",
        "source": "stdout",
    },
    # --- Floating Point ---
    {
        "name": "mandelbrot/mandel.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "mandelbrot"),
        "args": ["mandel.wasm", "128", "6e6"],
        "pattern": r"Elapsed time:\s*(.+)",
        "source": "stderr",
    },
    {
        "name": "c-ray/c-ray.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "c-ray"),
        "args": ["c-ray.wasm", "-s", "4000x4000"],
        "stdin": os.path.join(SCRIPT_DIR, "c-ray", "scene"),
        "pattern": r"Rendering took:\s*(.+\))",
        "source": "stderr",
    },
    # --- Memory Bound ---
    {
        "name": "stream/stream.wasm",
        "cwd": os.path.join(SCRIPT_DIR, "stream"),
        "args": ["stream.wasm"],
        "pattern": r"(Copy|Scale|Add|Triad):\s+(\S+)",
        "source": "stdout",
        "multi": True,
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
    },
]


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
    args = parser.parse_args()

    cli = args.cli
    cli_extra = args.cli_args.split() if args.cli_args else []
    if not os.path.exists(cli):
        print(f"ERROR: runtime not found at {cli}", file=sys.stderr)
        if cli == DEFAULT_CLI:
            print("Run: cargo build --release --bin sf-nano-cli", file=sys.stderr)
        sys.exit(1)

    print(f"Runtime: {cli}")
    print()

    results = []
    total = len(TESTS)
    for i, test in enumerate(TESTS, 1):
        name = test["name"]
        print(f"[{i}/{total}] {name} ...", end=" ", flush=True)
        result = run_test(cli, test, cli_extra)
        results.append(result)
        _, status, metric, _ = result
        print(f"{status}  {metric}")

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
