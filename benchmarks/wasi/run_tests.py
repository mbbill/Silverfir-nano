#!/usr/bin/env python3
"""Run the WASI benchmark suite with sf-nano-cli and collect results.

Usage:
    python3 run_tests.py                # run the suite (2s per benchmark)
    python3 run_tests.py --time 10      # 10s per benchmark
    python3 run_tests.py --interp       # run with the interpreter engine
    python3 run_tests.py --exec PATH    # run with a different runtime

Most benchmarks are self-timing: they take a wall-clock target in seconds as
their LAST argument and calibrate how many identical work units fit it. The
work unit, data size, algorithm, and working set stay fixed; only the repeat
count changes. Calibration is not a performance sample: a fresh batch is
timed after calibration and reports work/second. This keeps run time bounded
across native JITs, interpreters, and qemu without comparing different
problem sizes. See common/bench.h for the shared contract.

CoreMark's bare invocation still follows the upstream EEMBC benchmark,
including its 10-second-minimum calibration. This regression harness uses
CoreMark's explicit --target-seconds extension so its wall time is bounded
like the rest of the suite; such a run is not an official CoreMark result.

Validation is always a fixed, target-independent check (a checksum, a hash,
a CRC), never anything that scales with the workload.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_CLI = os.path.join(REPO_ROOT, "target", "release", "sf-nano-cli")

DEFAULT_TARGET = 2.0
TIMEOUT = 300


def bench(name, subdir, args, pattern, source="stdout", **kw):
    return dict(name=name, cwd=os.path.join(SCRIPT_DIR, subdir), args=args,
                pattern=pattern, source=source, **kw)


TESTS = [
    # --- Integer / control flow ---
    bench("coremark/coremark.wasm", "coremark", ["coremark.wasm"],
          r"Iterations/Sec\s*:\s*(\S+)",
          target_arg="--target-seconds={seconds}",
          correctness_args=["0", "0", "102", "1"],
          contains=["seedcrc          : 0xe9f5",
                    "[0]crclist       : 0xe714",
                    "[0]crcmatrix     : 0x1fd7",
                    "[0]crcstate      : 0x8e3a"]),
    bench("sha256/sha256.wasm", "sha256", ["sha256.wasm"],
          r"sha256: throughput = (\S+ MB/s)",
          contains="hash = 5eb4ca70d0ee472b"),
    bench("bzip2/bzip2.wasm", "bzip2", ["bzip2.wasm"],
          r"bzip2: throughput = (\S+ MB/s)",
          contains="32 KB input -> 3 KB compressed"),
    bench("lz4/lz4.wasm", "lz4", ["lz4.wasm"],
          r"(lz4 (?:compress|decompress): throughput = \S+ MB/s)",
          multi=True, contains="64 KB input -> 27 KB compressed"),
    # --- Lua ---
    bench("lua/fib", "lua", ["lua.wasm", "fib.lua"],
          r"fib: rate = (\S+ fib20/s)", contains="fib(20) = 6765"),
    bench("lua/sunfish", "lua", ["lua.wasm", "sunfish.lua"],
          r"Score:\s+(\S+)", contains="Result:        b1c3 / 0"),
    bench("lua/json_bench", "lua", ["lua.wasm", "json_bench.lua"],
          r"Score:\s+(\S+)", contains="JSON roundtrip validates"),
    # --- Floating point ---
    bench("mandelbrot/mandel.wasm", "mandelbrot", ["mandel.wasm"],
          r"mandel: rate = (\S+ Kpixel/s)",
          contains="mandel: checksum = 6a0fc6b0"),
    bench("c-ray/c-ray.wasm", "c-ray", ["c-ray.wasm"],
          r"c-ray: rate = (\S+ Kpixel/s)",
          stdin=os.path.join(SCRIPT_DIR, "c-ray", "scene"),
          contains="c-ray: checksum = 75700000"),
    # --- Memory bound ---
    bench("stream/stream.wasm", "stream", ["stream.wasm"],
          r"(Copy|Scale|Add|Triad):\s+(\S+)",
          multi=True, contains="Solution Validates"),
    # --- Database ---
    bench("sqlite/sqlite_bench.wasm", "sqlite", ["sqlite_bench.wasm"],
          r"sqlite: rate = (\S+ iteration/s)",
          contains="sqlite: checksum = 524800"),
]


def _invoke(cli, test, cli_extra, prog_args, stdin_data):
    cmd = list(cli_extra) + test.get("cli_args", []) + prog_args
    t0 = time.monotonic()
    proc = subprocess.run([cli] + cmd, cwd=test["cwd"], input=stdin_data,
                          capture_output=True, timeout=TIMEOUT)
    return proc, time.monotonic() - t0


def program_args(test, time_target, correctness_only=False):
    """Return argv for correctness or self-calibrated performance mode."""
    args = list(test["args"])
    if correctness_only:
        args.extend(test.get("correctness_args", ["--bench-correctness"]))
    elif test.get("target_arg"):
        args.append(test["target_arg"].format(seconds=time_target))
    else:
        args.append(str(time_target))
    return args


def run_test(
    cli, test, cli_extra=(), time_target=None, correctness_only=False
):
    name = test["name"]
    target = DEFAULT_TARGET if time_target is None else time_target

    wasm = os.path.join(test["cwd"], test["args"][0])
    if not os.path.exists(wasm):
        return name, "SKIP", "wasm file not found", None

    stdin_data = None
    stdin_file = test.get("stdin")
    if stdin_file:
        if not os.path.exists(stdin_file):
            return name, "SKIP", f"missing input: {os.path.basename(stdin_file)}", None
        with open(stdin_file, "rb") as f:
            stdin_data = f.read()

    prog_args = program_args(test, target, correctness_only)

    try:
        proc, elapsed = _invoke(cli, test, cli_extra, prog_args, stdin_data)
    except subprocess.TimeoutExpired:
        return name, "FAIL", f"timeout ({TIMEOUT}s)", None
    except Exception as e:
        return name, "FAIL", str(e), None

    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        for line in (stdout + stderr).splitlines():
            if "Error" in line:
                return name, "FAIL", line.strip(), elapsed
        return name, "FAIL", f"exit code {proc.returncode}", elapsed

    needles = test.get("contains") or []
    if isinstance(needles, str):
        needles = [needles]
    for needle in needles:
        if needle not in stdout:
            return name, "FAIL", f"expected stdout to contain '{needle}'", elapsed

    if correctness_only:
        return name, "PASS", "validated", elapsed

    # Compile stats, when the runtime reports them: [arch] (func:N, ssa:N, ...)
    info = ""
    for line in stderr.splitlines():
        if "(func:" in line and "ssa:" in line:
            info = "  " + line.strip()
            break

    text = stderr if test["source"] == "stderr" else stdout
    if test.get("multi"):
        matches = re.findall(test["pattern"], text, re.MULTILINE)
        if matches:
            if isinstance(matches[0], tuple):
                metric = ", ".join(f"{a}: {b} MB/s" for a, b in matches)
            else:
                metric = "; ".join(m.strip() for m in matches)
            return name, "PASS", f"{metric}{info}", elapsed
    else:
        m = re.search(test["pattern"], text, re.MULTILINE)
        if m:
            return name, "PASS", f"{m.group(1).strip()}{info}", elapsed

    return name, "PASS", f"{elapsed:.3f}s (no metric found){info}", elapsed


def main():
    p = argparse.ArgumentParser(description="Run WASI benchmark tests")
    p.add_argument("--exec", dest="cli", default=DEFAULT_CLI,
                   help="Path to the WASM runtime executable")
    p.add_argument("--cli-args", default="",
                   help="Extra args for the runtime (e.g. '--dir .' for wasmtime)")
    p.add_argument(
        "--time",
        type=float,
        default=DEFAULT_TARGET,
        help=(
            f"Seconds per benchmark (default {DEFAULT_TARGET}); CoreMark uses "
            "its non-standard regression mode"
        ),
    )
    p.add_argument("--interp", action="store_true",
                   help="Run with the interpreter (passes --interp to sf-nano-cli)")
    p.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run one fixed work unit per benchmark without calibration",
    )
    args = p.parse_args()

    target = args.time

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

    tests = TESTS
    mode = (
        "correctness only"
        if args.correctness_only
        else f"{target}s/self-calibrated benchmark"
    ) + (", interp" if args.interp else "")
    print(f"Runtime: {' '.join(cli_parts)} ({mode})")
    print()

    results = []
    for i, test in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {test['name']} ...", end=" ", flush=True)
        result = run_test(
            cli,
            test,
            cli_extra,
            time_target=target,
            correctness_only=args.correctness_only,
        )
        results.append(result)
        _, status, metric, elapsed = result
        print(f"{status}  {f'{elapsed:.1f}s' if elapsed else ''}  {metric}")

    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    failed = sum(1 for _, s, _, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _, _ in results if s == "SKIP")

    print()
    print("=" * 72)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped"
          f" / {len(tests)} total")
    print("=" * 72)
    print()
    print(f"{'Test':<35} {'Status':<6} {'Metric'}")
    print("-" * 72)
    for name, status, metric, _ in results:
        print(f"{name:<35} {status:<6} {metric}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
