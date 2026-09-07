#!/usr/bin/env python3
"""Capture and check the embedding API, including memprof transparency.

Capturing writes a candidate for review, never the accepted snapshots. A
snapshot update is not evidence of human approval; see docs/PUBLIC_API_POLICY.md.
The Rust toolchain and extractor are pinned because rustdoc JSON and inferred
trait implementations are compiler-version dependent.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

TOOLCHAIN = "nightly-2026-09-06"
EXTRACTOR_VERSION = "0.52.0"
TARGET = "x86_64-unknown-linux-gnu"
PROFILES = {
    "default": None,
    "jit": "jit,guard-pages",
    "interp": "interp",
    "both": "jit,interp,guard-pages",
    "wasi": "jit,interp,guard-pages,wasi",
}
SUPPORT_SNAPSHOTS = (
    "tracked-alloc.txt", "tracked-alloc-memprof.txt", "tracked-alloc-features.json",
)
SNAPSHOTS = tuple(f"{profile}.txt" for profile in PROFILES) + (
    "features.json", *SUPPORT_SNAPSHOTS,
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
WARNING = re.compile(r"^warning(?:\[.*?\])?:", re.MULTILINE)


def api_diff(expected: str, actual: str, label: str) -> str:
    return "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"{label}/accepted",
            tofile=f"{label}/candidate",
        )
    )


def implementation_leaks(surface: str) -> list[str]:
    # An external alias can retain the same printed name while its concrete
    # type changes with a feature. Parity alone cannot detect that situation.
    forbidden = ("tracked_alloc::", "sf_nano_core::vm::", "sf_nano_core::utils::")
    return [line for line in surface.splitlines() if any(name in line for name in forbidden)]


def feature_contract(manifest: Path) -> str:
    document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    contract = {
        "edition": document.get("package", {}).get("edition", "2015"),
        "rust_version": document.get("package", {}).get("rust-version"),
        "features": {
            name: sorted(values)
            for name, values in document.get("features", {}).items()
        },
        "optional_dependencies": sorted(
            name
            for name, dependency in document.get("dependencies", {}).items()
            if isinstance(dependency, dict) and dependency.get("optional", False)
        ),
    }
    return json.dumps(contract, sort_keys=True, indent=2) + "\n"


def run(command: list[str], root: Path, log: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        command, cwd=root, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    diagnostics = ANSI.sub("", result.stderr)
    if result.returncode or WARNING.search(diagnostics):
        raise RuntimeError(f"command failed or emitted warnings; see {log}")
    return result.stdout


def capture(
    root: Path, output: Path, extractor: str, *, baseline: bool = False
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CARGO_TERM_COLOR="never")
    env["CARGO_TARGET_DIR"] = str(root / "target" / "public-api-build")
    version = run([extractor, "--version"], root, output / "extractor.log", env)
    if version.strip() != f"cargo-public-api {EXTRACTOR_VERSION}":
        raise RuntimeError(f"expected cargo-public-api {EXTRACTOR_VERSION}: {version}")
    problems = []
    for profile, features in PROFILES.items():
        surfaces = []
        # Historical code is evidence, not a retroactive parity requirement.
        # The candidate must always exercise both feature settings.
        for memprof in ((False,) if baseline else (False, True)):
            variant = profile + ("-memprof" if memprof else "")
            command = [
                "cargo", f"+{TOOLCHAIN}", "rustdoc", "--locked",
                "-p", "sf-nano-core", "--lib", "--target", TARGET,
            ]
            enabled = [] if features is None else features.split(",")
            if features is not None:
                command.append("--no-default-features")
            if memprof:
                enabled.append("memprof")
            if enabled:
                command.extend(["--features", ",".join(enabled)])
            command.extend(["--", "-Z", "unstable-options", "--output-format", "json"])
            run(command, root, output / f"{variant}.build.log", env)
            rustdoc = Path(env["CARGO_TARGET_DIR"]) / TARGET / "doc" / "sf_nano_core.json"
            surface = run(
                [extractor, "--rustdoc-json", str(rustdoc), "--color", "never",
                 "--omit", "blanket-impls"],
                root, output / f"{variant}.extract.log", env,
            )
            # Keep auto traits and derived implementations: they are contracts.
            (output / f"{variant}.txt").write_text(surface, encoding="utf-8")
            surfaces.append(surface)
            leaks = implementation_leaks(surface)
            if leaks and not baseline:
                problems.append(f"{variant}: {len(leaks)} internal-type API leaks")
        delta = "" if baseline else api_diff(
            surfaces[0], surfaces[1], f"{profile}/memprof-parity"
        )
        if delta:
            (output / f"{profile}.memprof.diff").write_text(delta, encoding="utf-8")
            problems.append(f"{profile}: memprof changes the public API")
        print(f"captured {profile}", flush=True)
    (output / "features.json").write_text(
        feature_contract(root / "sf-nano-core" / "Cargo.toml"), encoding="utf-8"
    )
    capture_support_api(root, output, extractor, env)
    return problems


def capture_support_api(
    root: Path, output: Path, extractor: str, env: dict[str, str]
) -> None:
    # This package ships alongside core. Its tool-facing API is reviewed in
    # both configurations, separately from core's embedding parity invariant.
    for memprof in (False, True):
        variant = "tracked-alloc" + ("-memprof" if memprof else "")
        command = [
            "cargo", f"+{TOOLCHAIN}", "rustdoc", "--locked",
            "-p", "sf-nano-tracked-alloc", "--lib", "--target", TARGET,
        ]
        if memprof:
            command.extend(["--features", "memprof"])
        command.extend(["--", "-Z", "unstable-options", "--output-format", "json"])
        run(command, root, output / f"{variant}.build.log", env)
        rustdoc = Path(env["CARGO_TARGET_DIR"]) / TARGET / "doc" / "tracked_alloc.json"
        surface = run(
            [extractor, "--rustdoc-json", str(rustdoc), "--color", "never",
             "--omit", "blanket-impls"],
            root, output / f"{variant}.extract.log", env,
        )
        (output / f"{variant}.txt").write_text(surface, encoding="utf-8")
        print(f"captured {variant}", flush=True)
    (output / "tracked-alloc-features.json").write_text(
        feature_contract(root / "tools" / "tracked-alloc" / "Cargo.toml"),
        encoding="utf-8",
    )


def compare_snapshots(accepted: Path, candidate: Path) -> list[str]:
    problems = []
    for name in SNAPSHOTS:
        expected = accepted / name
        if not expected.is_file():
            problems.append(f"missing reviewed API snapshot: {expected}")
            continue
        actual = (candidate / name).read_text(encoding="utf-8")
        delta = api_diff(expected.read_text(encoding="utf-8"), actual, name)
        if delta:
            (candidate / f"{name}.diff").write_text(delta, encoding="utf-8")
            problems.append(f"API changed: {name}; explicit API review required")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "check", "baseline"))
    parser.add_argument("--extractor", default="cargo-public-api")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "target" / "public-api-candidate").resolve()
    accepted = root / "ci" / "api"
    if output == accepted.resolve() or accepted.resolve() in output.parents:
        parser.error("capture cannot overwrite accepted API snapshots")
    try:
        problems = capture(root, output, args.extractor, baseline=args.mode == "baseline")
        if args.mode == "check":
            problems.extend(compare_snapshots(accepted, output))
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"API evidence: {output}")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
