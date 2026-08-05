"""Build reproducible baseline/candidate CLI binaries for performance CI.

The two revisions live in different checkout directories. Rust may otherwise
let those absolute paths affect diagnostics, metadata, and ultimately code
layout. Each source root is remapped to the same virtual path before copying
the resulting executable and recording its SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Sequence


RV32_LINUX = "riscv32gc-unknown-linux-musl"
VIRTUAL_SOURCE_ROOT = "/workspace"
IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE = 0x0040


def cargo_command(*, engine: str, target: str) -> list[str]:
    if target == RV32_LINUX:
        command = [
            "cargo",
            "+nightly",
            "build",
            "-Z",
            "build-std=std,panic_abort",
        ]
    else:
        command = ["cargo", "build"]
    command.extend(
        [
            "--release",
            "-p",
            "sf-nano-cli",
            "--no-default-features",
            "--features",
            engine,
        ]
    )
    if target:
        command.extend(["--target", target])
    return command


def cargo_config_rustflags(source: Path, target: str) -> list[str]:
    config_path = source / ".cargo" / "config.toml"
    if not config_path.is_file():
        return []
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    configured: object = config.get("build", {}).get("rustflags", [])
    if target:
        target_flags = config.get("target", {}).get(target, {}).get("rustflags")
        if target_flags is not None:
            configured = target_flags
    if isinstance(configured, str):
        raise TypeError(
            f"{config_path}: string rustflags cannot be merged safely; use a TOML array"
        )
    if not isinstance(configured, list) or not all(
        isinstance(flag, str) for flag in configured
    ):
        raise TypeError(f"{config_path}: rustflags must be an array of strings")
    return list(configured)


def remapped_environment(
    source: Path,
    target: str,
    environ: dict[str, str],
) -> dict[str, str]:
    env = dict(environ)
    remap = f"--remap-path-prefix={source.resolve()}={VIRTUAL_SOURCE_ROOT}"
    encoded = env.get("CARGO_ENCODED_RUSTFLAGS")
    if encoded is not None:
        flags = encoded.split("\x1f") if encoded else []
    else:
        rustflags = env.get("RUSTFLAGS", "").strip()
        if rustflags:
            raise ValueError(
                "performance CI requires CARGO_ENCODED_RUSTFLAGS when external "
                "flags are present; whitespace-split RUSTFLAGS cannot be merged safely"
            )
        flags = cargo_config_rustflags(source, target)
    env.pop("RUSTFLAGS", None)
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join([*flags, remap])
    return env


def cargo_target_dir(source: Path, environ: dict[str, str]) -> Path:
    configured = environ.get("CARGO_TARGET_DIR")
    if not configured:
        return source / "target"
    target_dir = Path(configured)
    return target_dir if target_dir.is_absolute() else source / target_dir


def isolated_build_environment(
    source: Path,
    *,
    label: str,
    target: str,
    environ: dict[str, str],
) -> dict[str, str]:
    env = remapped_environment(source, target, environ)
    env["CARGO_TARGET_DIR"] = str(
        (cargo_target_dir(source, env) / label).resolve()
    )
    # Baseline and candidate must compile with the same toolchain or the
    # measurement confounds compiler differences with code differences. A
    # baseline checkout may still carry a pinned rust-toolchain.toml, which
    # rustup would otherwise honor per working directory. An explicit
    # `cargo +nightly` (the rv32 build-std path) outranks this variable.
    env.setdefault("RUSTUP_TOOLCHAIN", "stable")
    # Measurement binaries get a fixed image base on Windows, where ASLR
    # is a link-time property: a benchmark score must be a property of
    # the image, not of the launch's address draw. Linux and Darwin pin
    # at spawn time instead (setarch -R and ci/noaslr.c). Applies only
    # to these perf builds; shipped artifacts keep dynamic bases.
    if sys.platform == "win32":
        flags = env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        flags.extend(["-C", "link-arg=/DYNAMICBASE:NO"])
        env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    return env


def built_executable(
    source: Path,
    *,
    target: str,
    exe_suffix: str,
    environ: dict[str, str],
) -> Path:
    target_prefix = Path(target) if target else Path()
    return cargo_target_dir(source, environ) / target_prefix / "release" / f"sf-nano-cli{exe_suffix}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pe_image_metadata(path: Path) -> dict[str, object]:
    """Read the image base and ASLR bit from a PE32/PE32+ executable."""
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise ValueError(f"{path}: missing DOS header")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    optional = pe_offset + 24
    if optional + 72 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError(f"{path}: missing PE optional header")
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic == 0x20B:
        image_base = struct.unpack_from("<Q", data, optional + 24)[0]
    elif magic == 0x10B:
        image_base = struct.unpack_from("<I", data, optional + 28)[0]
    else:
        raise ValueError(f"{path}: unsupported PE optional-header magic 0x{magic:04x}")
    characteristics = struct.unpack_from("<H", data, optional + 70)[0]
    return {
        "image_base": f"0x{image_base:016x}",
        "dynamic_base": bool(
            characteristics & IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE
        ),
    }


def build_one(
    *,
    label: str,
    source: Path,
    output: Path,
    engine: str,
    target: str,
    exe_suffix: str,
    environ: dict[str, str],
) -> dict[str, object]:
    source = source.resolve()
    env = isolated_build_environment(
        source,
        label=label,
        target=target,
        environ=environ,
    )
    command = cargo_command(engine=engine, target=target)
    print(f"[build {label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=source, env=env, check=True)

    artifact = built_executable(
        source,
        target=target,
        exe_suffix=exe_suffix,
        environ=env,
    )
    if not artifact.is_file():
        raise FileNotFoundError(f"cargo succeeded but executable is missing: {artifact}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, output)
    digest = sha256_file(output)
    size = output.stat().st_size
    image = pe_image_metadata(output) if sys.platform == "win32" else {}
    if image.get("dynamic_base"):
        raise RuntimeError(
            f"{output}: performance image still enables DYNAMIC_BASE"
        )
    print(f"[build {label}] sha256={digest} size={size}", flush=True)
    return {
        "sha256": digest,
        "size": size,
        "virtual_source_root": VIRTUAL_SOURCE_ROOT,
        **image,
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=("jit", "interp"), required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--exe-suffix", default="")
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    out_dir = args.out_dir.resolve()
    environ = dict(os.environ)
    builds = {}
    for label, source, revision in (
        ("baseline", args.baseline_source, args.baseline_sha),
        ("candidate", args.candidate_source, args.candidate_sha),
    ):
        builds[label] = {
            "revision": revision,
            **build_one(
                label=label,
                source=source,
                output=out_dir / f"{label}{args.exe_suffix}",
                engine=args.engine,
                target=args.target,
                exe_suffix=args.exe_suffix,
                environ=environ,
            ),
        }

    metadata = {
        "schema_version": 1,
        "platform": args.platform,
        "engine": args.engine,
        "target": args.target,
        "builds": builds,
        "identical_binaries": builds["baseline"]["sha256"] == builds["candidate"]["sha256"],
    }
    if sys.platform == "win32":
        bases = {build["image_base"] for build in builds.values()}
        if len(bases) != 1:
            raise RuntimeError(
                "baseline and candidate performance images use different image bases: "
                + ", ".join(sorted(bases))
            )
    metadata_path = out_dir / "build-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"[build comparison] identical_binaries={str(metadata['identical_binaries']).lower()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
