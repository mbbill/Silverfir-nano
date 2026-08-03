"""Capture an x64 profile of one benchmarks/wasi workload on a CI runner.

Branch-only tooling for .github/workflows/x64-profile.yml: runs one
suite workload under `perf record` with nano's jitdump exporter enabled
(core `jit-debug` feature), injects the JIT symbols, and writes text
reports plus the static native dump next to them. The artifact is read
locally; this script only captures. Deleted with the rest of the
dev/x64-** lane before merge.

GitHub runners are VMs without PMU passthrough: `perf record` falls back
to a software clock sampler and `perf stat` hardware counters may read
"<not supported>". Both outputs are still captured.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_DIR = REPO_ROOT / "benchmarks" / "wasi"

JIT_SYMBOL = re.compile(r"\bjit::\S+")


def load_run_tests():
    spec = importlib.util.spec_from_file_location(
        "wasi_run_tests", SUITE_DIR / "run_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_test(run_tests, workload: str) -> dict:
    for test in run_tests.TESTS:
        if test["name"] == workload or test["name"].startswith(workload):
            return test
    names = ", ".join(test["name"] for test in run_tests.TESTS)
    raise SystemExit(f"unknown workload {workload!r}; suite has: {names}")


def run(cmd, **kwargs):
    print("+", " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--engine", choices=["jit", "interp"], default="jit")
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--perf", default="perf")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--annotate-top", type=int, default=5)
    args = parser.parse_args(argv)

    run_tests = load_run_tests()
    test = find_test(run_tests, args.workload)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    cli = str(Path(args.cli).resolve())
    prog_args = run_tests.program_args(test, args.time)
    engine_args = ["--interp"] if args.engine == "interp" else []
    workload_cmd = [cli] + engine_args + prog_args
    cwd = test["cwd"]

    stdin_file = test.get("stdin")
    stdin = open(stdin_file, "rb") if stdin_file else subprocess.DEVNULL

    env = dict(os.environ)
    env["SF_JITDUMP"] = "1"
    env["SF_JITDUMP_DIR"] = str(out)
    env["SF_NATIVE_DUMP_DIR"] = str(out / "native-dump")

    # -k mono: jitdump timestamps must share the record clock or inject
    # drops every region.
    record = run(
        [args.perf, "record", "-k", "mono", "-o", str(out / "perf.data"),
         "--"] + workload_cmd,
        cwd=cwd, stdin=stdin, env=env,
    )
    if stdin_file:
        stdin.close()
    if record.returncode != 0:
        raise SystemExit(f"perf record failed: {record.returncode}")

    inject = run(
        [args.perf, "inject", "--jit", "-i", str(out / "perf.data"),
         "-o", str(out / "perf.jit.data")],
        cwd=cwd,
    )
    perf_data = "perf.jit.data" if inject.returncode == 0 else "perf.data"
    if inject.returncode != 0:
        print("perf inject --jit failed; reporting without JIT symbols",
              file=sys.stderr)

    with open(out / "report.txt", "w", encoding="utf-8") as sink:
        run(
            [args.perf, "report", "-i", str(out / perf_data), "--stdio",
             "--no-children", "--percent-limit", "0.3"],
            cwd=cwd, stdout=sink, stderr=subprocess.STDOUT,
        )

    report = (out / "report.txt").read_text(encoding="utf-8", errors="replace")
    symbols: list[str] = []
    for match in JIT_SYMBOL.finditer(report):
        symbol = match.group(0)
        if symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= args.annotate_top:
            break
    for index, symbol in enumerate(symbols):
        with open(out / f"annotate-{index}.txt", "w", encoding="utf-8") as sink:
            sink.write(f"symbol: {symbol}\n\n")
            sink.flush()
            run(
                [args.perf, "annotate", "-i", str(out / perf_data), "--stdio",
                 "--symbol", symbol],
                cwd=cwd, stdout=sink, stderr=subprocess.STDOUT,
            )

    # Unprofiled second run for wall-clock and best-effort counters.
    with open(out / "stat.txt", "w", encoding="utf-8") as sink:
        run(
            [args.perf, "stat", "-d", "--"] + workload_cmd,
            cwd=cwd,
            stdin=open(stdin_file, "rb") if stdin_file else subprocess.DEVNULL,
            env=dict(os.environ),
            stdout=sink, stderr=subprocess.STDOUT,
        )

    (out / "meta.json").write_text(
        json.dumps(
            {
                "workload": test["name"],
                "engine": args.engine,
                "time_target": args.time,
                "command": workload_cmd,
                "cwd": str(cwd),
                "platform": platform.platform(),
                "perf": args.perf,
                "jit_symbols": symbols,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"captured: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
