"""Audit saved experimental compiler logs with the correctness warning parser."""
from __future__ import annotations

import argparse
from pathlib import Path

from ci.runner import parse_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    args = parser.parse_args()
    failed = False
    for log in args.logs:
        # parse_log also serves subprocess failures without a saved log;
        # this standalone audit requires a readable, nonempty artifact.
        with log.open('rb') as stream:
            if not stream.read(1):
                parser.error(f'empty compiler log: {log}')
        errors, warnings, diagnostics = parse_log(log, 0)
        print(f'{log}: {errors} errors, {warnings} warnings')
        if errors or warnings:
            print(f'::error::ACTION REQUIRED: compiler diagnostics in {log}')
            for diagnostic in diagnostics:
                print(diagnostic)
            failed = True
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
