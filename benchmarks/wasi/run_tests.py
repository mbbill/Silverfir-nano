#!/usr/bin/env python3
"""Run the WASI benchmark suite with sf-nano-cli and collect results.

Usage:
    python3 run_tests.py                # run the suite (2s per benchmark)
    python3 run_tests.py --time 10      # 10s per benchmark, for formal runs
    python3 run_tests.py --fast         # alias for --time 1, for CI/QEMU
    python3 run_tests.py --interp       # run with the interpreter engine
    python3 run_tests.py --exec PATH    # run with a different runtime

Every benchmark is self-timing: it takes a wall-clock target in seconds as
its LAST argument and sizes its own workload to hit it, so a run costs
about the same on a native JIT as on an interpreter under qemu. That is why
there is no second "reduced workload" variant to keep in sync -- the target
is the only knob. See common/bench.h for the ramp and its one rule.

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
          r"Score:\s+(\S+)"),
    bench("lua/json_bench", "lua", ["lua.wasm", "json_bench.lua"],
          r"Score:\s+(\S+)"),
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
    # speedtest1 is the one benchmark that cannot ramp a batch: it runs a
    # fixed one-shot sequence of sub-tests whose size is set by --size, so
    # it is regulated from here instead (a calibration run picks the size).
    # That costs one extra process launch, which on the JIT means compiling
    # its 1423 functions twice -- the reason this entry is the slowest to
    # start despite doing the least work.
    bench("sqlite/speedtest1.wasm", "sqlite",
          ["speedtest1.wasm", "--memdb", "--nosync", "--journal", "off",
           "--testset", "main"],
          r"^\s*(TOTAL\.+\s+\S+s)",
          size_arg=True),  # --memdb needs no filesystem, so no --dir
]


def resolve_test(test, fast=False):
    """Kept for API compatibility (scripts/bench_publish.py).

    Reduced "fast" workloads no longer exist: --fast is now just a smaller
    time target, which every benchmark honours on its own.
    """
    return test


def _invoke(cli, test, cli_extra, prog_args, stdin_data):
    cmd = list(cli_extra) + test.get("cli_args", []) + prog_args
    t0 = time.monotonic()
    proc = subprocess.run([cli] + cmd, cwd=test["cwd"], input=stdin_data,
                          capture_output=True, timeout=TIMEOUT)
    return proc, time.monotonic() - t0


TOTAL_RE = re.compile(r"^\s*TOTAL\.+\s+([\d.]+)s", re.MULTILINE)


def _sqlite_probe(cli, test, cli_extra, target):
    """Probe speedtest1 at a small size to pick the size for the real run.

    Returns (size, secs) of the probe, or None if it could not be timed.
    The probe is a real measurement -- it knows its own size and time -- so
    when it already covers enough of the target the caller reports it and
    skips the second run. That matters most on slow engines, where each
    launch re-JITs speedtest1's 1400+ functions and the compile, not the
    query work, is the dominant cost.
    """
    cal = 10
    args = test["args"] + ["--size", str(cal)]
    try:
        proc, _ = _invoke(cli, test, cli_extra, args, None)
    except Exception:
        return None
    m = TOTAL_RE.search(proc.stdout.decode("utf-8", errors="replace"))
    if not m or float(m.group(1)) <= 0:
        return None
    return cal, float(m.group(1))


def run_test(cli, test, cli_extra=(), time_target=None):
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

    prog_args = list(test["args"])
    size = None
    if test.get("size_arg"):
        probe = _sqlite_probe(cli, test, cli_extra, target)
        if probe and probe[1] >= target / 2.0:
            # Already covers enough of the target; a second, larger run would
            # only re-pay the compile for no extra confidence.
            psize, psecs = probe
            return name, "PASS", (f"{psize / psecs:.2f} size/s "
                                  f"(size={psize}, {psecs:.3f}s)"), psecs
        size = int(10 * target / probe[1]) if probe else 10
        size = max(1, min(size, 1000))
        prog_args += ["--size", str(size)]
    else:
        prog_args.append(str(target))  # every benchmark: target is the last arg

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

    # Compile stats, when the runtime reports them: [arch] (func:N, ssa:N, ...)
    info = ""
    for line in stderr.splitlines():
        if "(func:" in line and "ssa:" in line:
            info = "  " + line.strip()
            break

    if size is not None:
        m = TOTAL_RE.search(stdout)
        if m and float(m.group(1)) > 0:
            rate = size / float(m.group(1))
            return name, "PASS", (f"{rate:.2f} size/s "
                                  f"(size={size}, {m.group(1)}s){info}"), elapsed
        return name, "FAIL", "no TOTAL line in output", elapsed

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
    p.add_argument("--time", type=float, default=None,
                   help=f"Seconds per benchmark (default {DEFAULT_TARGET})")
    p.add_argument("--fast", action="store_true",
                   help="Alias for --time 1, for CI/QEMU")
    p.add_argument("--interp", action="store_true",
                   help="Run with the interpreter (passes --interp to sf-nano-cli)")
    args = p.parse_args()

    target = args.time if args.time is not None else (1.0 if args.fast else DEFAULT_TARGET)

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
    mode = f"{target}s/benchmark" + (", interp" if args.interp else "")
    print(f"Runtime: {' '.join(cli_parts)} ({mode})")
    print()

    results = []
    for i, test in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {test['name']} ...", end=" ", flush=True)
        result = run_test(cli, test, cli_extra, time_target=target)
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
