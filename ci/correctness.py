"""Correctness gates, partitioned one logical platform per CI job.

Usage:
    python -m ci.correctness host x64-linux
    python -m ci.correctness host arm64-linux
    python -m ci.correctness host arm64-darwin
    python -m ci.correctness host x64-windows
    python -m ci.correctness cross armv7
    python -m ci.correctness cross riscv64
    python -m ci.correctness cross riscv32
    python -m ci.correctness bare thumbv8m
    python -m ci.correctness bare riscv32

Host jobs run only native work. Cross jobs all run on x64 Linux and each own
one QEMU-user target. Bare jobs compile and assemble target_os="none"
configurations; they do not pretend that qemu-user can execute bare-metal
binaries.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ci.runner import ROOT, TARGET, Result, Runner, require_tools, slug


TESTSUITE = TARGET / "webassembly-testsuite"
RV32_LINUX = "riscv32gc-unknown-linux-musl"

ENGINE_FEATURE_CONFIGS = (
    ("jit-only", "jit"),
    ("interp-only", "interp"),
    ("dual-engine", "jit,interp"),
)


@dataclass(frozen=True)
class Engine:
    name: str
    features: str
    runtime_args: tuple[str, ...]


JIT = Engine("jit", "jit", ("--backend", "native"))
INTERP = Engine("interp", "interp", ("--interp",))


def spectest_features(engine: Engine) -> str:
    """Features needed by the shared WAST harness for this runtime tier.

    The interpreter itself remains an interp build, but sf-nano-spectest's
    WAST driver currently lives behind its `jit` feature because both tiers
    share that entity-model harness.  Keep pure-interp compilation in the
    feature matrices; only the executable that actually drives WAST needs
    the combined harness.
    """

    return "jit,interp" if engine == INTERP else engine.features


@dataclass(frozen=True)
class CrossPlatform:
    name: str
    target: str
    qemu: str
    cpu: str
    jit_variants: tuple[tuple[str, str], ...]
    wasi_skip_args: tuple[str, ...] = ()


CROSS_PLATFORMS = {
    "armv7": CrossPlatform(
        name="armv7",
        target="armv7-unknown-linux-musleabihf",
        qemu="qemu-arm-static",
        cpu="cortex-a15",
        jit_variants=(("armv7a", ""), ("armv7m", "thumb2-test")),
    ),
    "riscv64": CrossPlatform(
        name="riscv64",
        target="riscv64gc-unknown-linux-musl",
        qemu="qemu-riscv64-static",
        cpu="rv64",
        jit_variants=(("riscv64", ""),),
    ),
    "riscv32": CrossPlatform(
        name="riscv32",
        target=RV32_LINUX,
        qemu="qemu-riscv32-static",
        cpu="rv32",
        jit_variants=(("riscv32", ""),),
        wasi_skip_args=("--skip-rv32-qemu-timestamp-tests",),
    ),
}

BARE_PLATFORMS = {
    "thumbv8m": "thumbv8m.main-none-eabihf",
    "riscv32": "riscv32imac-unknown-none-elf",
}

HOST_EXPECTATIONS = {
    "x64-linux": ("Linux", {"x86_64", "amd64"}),
    "arm64-linux": ("Linux", {"aarch64", "arm64"}),
    "arm64-darwin": ("Darwin", {"aarch64", "arm64"}),
    "x64-windows": ("Windows", {"amd64", "x86_64"}),
}


def cargo_prefix(subcommand: str, target: str | None = None) -> list[str]:
    if target == RV32_LINUX:
        return ["cargo", "+nightly", subcommand, "-Z", "build-std=std,panic_abort"]
    return ["cargo", subcommand]


def cargo_env(target: str | None = None, extra: dict[str, object] | None = None) -> dict[str, object]:
    env: dict[str, object] = dict(extra or {})
    if target == RV32_LINUX:
        env["ZIG_GLOBAL_CACHE_DIR"] = TARGET / "zig-cache"
    return env


def profile_args(profile: str) -> list[str]:
    return ["--release"] if profile == "release" else []


def profile_dir(profile: str) -> str:
    return "release" if profile == "release" else "debug"


def feature_args(features: str) -> list[str]:
    if features == "default":
        return []
    if features == "all":
        return ["--all-features"]
    return ["--no-default-features", "--features", features]


def binary_path(name: str, profile: str, *, target: str | None = None) -> Path:
    suffix = ".exe" if (target and "windows" in target) or (target is None and os.name == "nt") else ""
    base = TARGET / target if target else TARGET
    return base / profile_dir(profile) / f"{name}{suffix}"


def cargo(
    runner: Runner,
    name: str,
    subcommand: str,
    *,
    package: str | None = None,
    profile: str = "debug",
    target: str | None = None,
    features: str | None = None,
    extra: Sequence[str] = (),
    cwd: Path = ROOT,
    env: dict[str, object] | None = None,
) -> Result:
    argv = cargo_prefix(subcommand, target)
    argv.extend(profile_args(profile))
    if target:
        argv.extend(["--target", target])
    if package:
        argv.extend(["-p", package])
    if features is not None:
        argv.extend(feature_args(features))
    argv.extend(extra)
    return runner.run(name, argv, cwd=cwd, env=cargo_env(target, env))


def validate_host(runner: Runner, label: str) -> bool:
    expected_system, expected_machines = HOST_EXPECTATIONS[label]
    actual_system = platform.system()
    actual_machine = platform.machine().lower()
    if actual_system == expected_system and actual_machine in expected_machines:
        return True
    runner.run(
        "validate runner platform",
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"print('expected {expected_system}/{sorted(expected_machines)}; "
                f"got {actual_system}/{actual_machine}'); sys.exit(1)"
            ),
        ],
    )
    return False


def run_host_builds_and_tests(runner: Runner) -> None:
    # Build both profiles explicitly, but run the unit-test suite once. The
    # release binaries receive stronger end-to-end coverage from the spec and
    # WASI suites below; repeating every unit test under --release was the
    # single largest native-host command without adding a distinct boundary.
    cargo(
        runner,
        "cargo build workspace (debug)",
        "build",
        extra=("--workspace",),
    )
    cargo(
        runner,
        "cargo test workspace (debug)",
        "test",
        extra=("--workspace",),
    )
    cargo(
        runner,
        "cargo build workspace (release)",
        "build",
        profile="release",
        extra=("--workspace",),
    )


def run_host_feature_matrix(runner: Runner) -> None:
    # Feature coverage is about Cargo boundaries, not host architecture, so
    # run it once on x64 Linux instead of repeating it on every native runner.
    #
    # Default features are already compiled and tested by
    # run_host_builds_and_tests(). The three engine configurations below are
    # the independently shipped boundaries. Diagnostics such as ir-dump,
    # jitdump, call-trace, interp-count, and memprof are development aids:
    # compile them together once rather than manufacturing pairwise
    # combinations that users do not ship.
    for package in ("sf-nano-core", "sf-nano-cli"):
        for label, features in ENGINE_FEATURE_CONFIGS:
            cargo(
                runner,
                f"{package} features {label} (release)",
                "check",
                package=package,
                profile="release",
                features=features,
            )
    cargo(
        runner,
        "workspace diagnostic features (release)",
        "check",
        profile="release",
        features="all",
        extra=("--workspace",),
    )


def ensure_testsuite(runner: Runner) -> bool:
    if TESTSUITE.is_dir():
        return True
    runner.run(
        "validate WebAssembly testsuite",
        [
            sys.executable,
            "-c",
            f"import sys; print('missing testsuite: {TESTSUITE}'); sys.exit(1)",
        ],
    )
    return False


def run_native_spectest(runner: Runner) -> None:
    if not ensure_testsuite(runner):
        return
    for engine in (JIT, INTERP):
        build = cargo(
            runner,
            f"build native spectest / {engine.name} (release)",
            "build",
            package="sf-nano-spectest",
            profile="release",
            features=spectest_features(engine),
        )
        run_name = f"run native spectest / {engine.name} (release)"
        if not build.produced_output:
            runner.skip(run_name, f"build failed: {build.name}")
            continue
        runner.run(
            run_name,
            [binary_path("sf-nano-spectest", "release"), *engine.runtime_args],
            env={"TESTSUITE_DIR": TESTSUITE},
        )


def run_native_wasitest(runner: Runner) -> None:
    harness = cargo(
        runner,
        "build native WASI harness (release)",
        "build",
        package="sf-nano-wasitest",
        profile="release",
    )
    for engine in (JIT, INTERP):
        cli = cargo(
            runner,
            f"build native CLI / {engine.name} (release)",
            "build",
            package="sf-nano-cli",
            profile="release",
            features=engine.features,
        )
        run_name = f"run native WASI / {engine.name} (release)"
        if not harness.produced_output or not cli.produced_output:
            runner.skip(run_name, "WASI harness or CLI build failed")
            continue
        runner.run(
            run_name,
            [
                binary_path("sf-nano-wasitest", "release"),
                "--cli-path",
                binary_path("sf-nano-cli", "release"),
                *engine.runtime_args,
            ],
        )


def run_host(label: str) -> int:
    runner = Runner(f"host/{label}")
    if not require_tools(runner, ("cargo", "rustc")) or not validate_host(runner, label):
        return runner.finish()
    run_host_builds_and_tests(runner)
    if label == "x64-linux":
        run_host_feature_matrix(runner)
    run_native_spectest(runner)
    run_native_wasitest(runner)
    return runner.finish()


def variant_features(engine: Engine, extra_feature: str) -> str:
    features = [engine.features]
    if extra_feature:
        features.append(extra_feature)
    return ",".join(features)


def copy_built_binary(
    runner: Runner,
    *,
    name: str,
    profile: str,
    target: str,
    artifact_name: str,
) -> Path | None:
    source = binary_path(name, profile, target=target)
    destination = runner.tmp_dir / artifact_name
    if not source.is_file():
        runner.run(
            f"locate artifact {artifact_name}",
            [
                sys.executable,
                "-c",
                f"import sys; print('missing binary: {source}'); sys.exit(1)",
            ],
        )
        return None
    shutil.copy2(source, destination)
    return destination


def qemu_command(config: CrossPlatform, binary: Path, args: Iterable[object]) -> list[object]:
    return [config.qemu, "-cpu", config.cpu, binary, *args]


def run_cross_compile_coverage(runner: Runner, config: CrossPlatform) -> None:
    for engine in (JIT, INTERP):
        cargo(
            runner,
            f"build {config.name} core / {engine.name}",
            "build",
            package="sf-nano-core",
            target=config.target,
            features=engine.features,
        )
        cargo(
            runner,
            f"build {config.name} CLI / {engine.name}",
            "build",
            package="sf-nano-cli",
            target=config.target,
            features=engine.features,
        )
        cargo(
            runner,
            f"build {config.name} spectest / {engine.name}",
            "build",
            package="sf-nano-spectest",
            target=config.target,
            features=engine.features,
        )


def run_cross_spectest(runner: Runner, config: CrossPlatform) -> None:
    if not ensure_testsuite(runner):
        return

    # JIT validates both A32 and forced Thumb-2 on the Arm Linux target.
    for variant, extra_feature in config.jit_variants:
        features = variant_features(JIT, extra_feature)
        build = cargo(
            runner,
            f"build {variant} spectest / jit (release)",
            "build",
            package="sf-nano-spectest",
            profile="release",
            target=config.target,
            features=features,
        )
        run_name = f"run {variant} spectest / jit (release)"
        if not build.produced_output:
            runner.skip(run_name, f"build failed: {build.name}")
            continue
        binary = copy_built_binary(
            runner,
            name="sf-nano-spectest",
            profile="release",
            target=config.target,
            artifact_name=f"spectest-{variant}-jit-release",
        )
        if binary:
            runner.run(
                run_name,
                qemu_command(config, binary, JIT.runtime_args),
                env={"TESTSUITE_DIR": TESTSUITE},
            )

    # Interpreter execution does not use the JIT's Arm encoding switch, so
    # one interpreter run per target is meaningful. The shared WAST driver
    # itself currently requires both harness features; compile coverage above
    # still verifies the pure-interp configuration independently.
    build = cargo(
        runner,
        f"build {config.name} spectest / interp (release)",
        "build",
        package="sf-nano-spectest",
        profile="release",
        target=config.target,
        features=spectest_features(INTERP),
    )
    run_name = f"run {config.name} spectest / interp (release)"
    if not build.produced_output:
        runner.skip(run_name, f"build failed: {build.name}")
        return
    binary = copy_built_binary(
        runner,
        name="sf-nano-spectest",
        profile="release",
        target=config.target,
        artifact_name=f"spectest-{config.name}-interp-release",
    )
    if binary:
        runner.run(
            run_name,
            qemu_command(config, binary, INTERP.runtime_args),
            env={"TESTSUITE_DIR": TESTSUITE},
        )


def write_qemu_wrapper(runner: Runner, config: CrossPlatform, cli: Path, label: str) -> Path:
    wrapper = runner.tmp_dir / f"run-{slug(label)}.sh"
    command = " ".join(
        shlex.quote(str(part))
        for part in ("env", "-u", "PWD", "-u", "SHLVL", "-u", "OLDPWD", "-u", "_", config.qemu, "-cpu", config.cpu, cli)
    )
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"exec {command} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def run_cross_wasitest(runner: Runner, config: CrossPlatform) -> None:
    harness = cargo(
        runner,
        "build native WASI harness (release)",
        "build",
        package="sf-nano-wasitest",
        profile="release",
    )
    if not harness.produced_output:
        runner.skip(f"run {config.name} WASI", "native WASI harness build failed")
        return
    harness_binary = binary_path("sf-nano-wasitest", "release")

    for engine in (JIT, INTERP):
        variants = config.jit_variants if engine is JIT else ((config.name, ""),)
        for variant, extra_feature in variants:
            features = variant_features(engine, extra_feature)
            build = cargo(
                runner,
                f"build {variant} CLI / {engine.name} (release)",
                "build",
                package="sf-nano-cli",
                profile="release",
                target=config.target,
                features=features,
            )
            run_name = f"run {variant} WASI / {engine.name} (release)"
            if not build.produced_output:
                runner.skip(run_name, f"build failed: {build.name}")
                continue
            cli = copy_built_binary(
                runner,
                name="sf-nano-cli",
                profile="release",
                target=config.target,
                artifact_name=f"cli-{variant}-{engine.name}-release",
            )
            if not cli:
                continue
            wrapper = write_qemu_wrapper(runner, config, cli, f"{variant}-{engine.name}")
            runner.run(
                run_name,
                [
                    harness_binary,
                    "--cli-path",
                    wrapper,
                    *engine.runtime_args,
                    *config.wasi_skip_args,
                ],
                env={"TMPDIR": runner.tmp_dir},
            )


def run_cross(label: str) -> int:
    config = CROSS_PLATFORMS[label]
    runner = Runner(f"cross/{label}")
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        validate_host(runner, "x64-linux")
        return runner.finish()
    required = ["cargo", "rustc", "rustup", config.qemu]
    if config.target == RV32_LINUX:
        required.append("zig")
    if not require_tools(runner, required):
        return runner.finish()
    run_cross_compile_coverage(runner, config)
    run_cross_spectest(runner, config)
    run_cross_wasitest(runner, config)
    return runner.finish()


def run_bare(label: str) -> int:
    target = BARE_PLATFORMS[label]
    runner = Runner(f"bare/{label}")
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        validate_host(runner, "x64-linux")
        return runner.finish()
    if not require_tools(runner, ("cargo", "rustc", "rustup")):
        return runner.finish()

    cargo(
        runner,
        f"build bare smoke / {label}",
        "build",
        target=target,
        cwd=ROOT / "sf-nano-bare-smoke",
        env={"CARGO_TARGET_DIR": TARGET / "ci-bare" / label},
    )
    for features in ("jit", "interp", "jit,interp"):
        cargo(
            runner,
            f"assemble bare core / {label} / {features}",
            "build",
            package="sf-nano-core",
            target=target,
            features=features,
        )
    return runner.finish()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="suite", required=True)

    host = subparsers.add_parser("host", help="native build/test/feature/spec/WASI coverage")
    host.add_argument("platform", choices=tuple(HOST_EXPECTATIONS))

    cross = subparsers.add_parser("cross", help="one Linux QEMU-user target")
    cross.add_argument("platform", choices=tuple(CROSS_PLATFORMS))

    bare = subparsers.add_parser("bare", help="one compile-only bare-metal target")
    bare.add_argument("platform", choices=tuple(BARE_PLATFORMS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.suite == "host":
        return run_host(args.platform)
    if args.suite == "cross":
        return run_cross(args.platform)
    return run_bare(args.platform)


if __name__ == "__main__":
    raise SystemExit(main())
