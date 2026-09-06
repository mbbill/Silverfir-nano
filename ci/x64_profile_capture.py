"""Temporary Nano-only profiling for the exact pinned wasmi fixtures."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command after -- is required")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SF_JITDUMP="1", SF_JITDUMP_DIR=str(out),
               SF_NATIVE_DUMP_DIR=str(out / "native-dump"))
    with (out / "workload.log").open("w") as log:
        subprocess.run(
            [args.perf, "record", "-e", "cpu-clock:u", "-F", "999", "-k", "mono",
             "-o", str(out / "perf.data"), "--", *command],
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180,
        )
    subprocess.run([args.perf, "inject", "--jit", "-i", str(out / "perf.data"),
                    "-o", str(out / "perf.jit.data")], check=True, timeout=120)
    with (out / "report.txt").open("w") as log:
        subprocess.run([args.perf, "report", "-i", str(out / "perf.jit.data"),
                        "--stdio", "--no-children", "--percent-limit", "0.3"],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    report = (out / "report.txt").read_text()
    symbols = list(dict.fromkeys(re.findall(r"\bjit::\S+", report)))[:8]
    if not symbols:
        raise RuntimeError("profile has no resolved JIT symbols")
    for index, symbol in enumerate(symbols):
        with (out / f"annotate-{index}.txt").open("w") as log:
            log.write(f"symbol: {symbol}\n\n")
            log.flush()
            subprocess.run([args.perf, "annotate", "-i", str(out / "perf.jit.data"),
                            "--stdio", "--symbol", symbol],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    (out / "meta.json").write_text(json.dumps({"command": command}, indent=2) + "\n")
    print((out / "workload.log").read_text())
    print(report)


if __name__ == "__main__":
    main()
