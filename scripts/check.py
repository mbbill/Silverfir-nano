#!/usr/bin/env python3
"""Concise validation runner for Silverfir Nano.

Usage:
  python3 scripts/check.py

There is one validation mode. A reduced "fast" mode used to exist because
the benchmark suite dominated a full run; the benchmarks are self-timing
now, so the run is the same work everywhere and the second code path was
only a way to be validated less. Narrow a run with --release-only /
--debug-only / --default-targets instead.

The runner keeps stdout compact and writes detailed command output to
target/check-logs/. Failures and warnings are summarized with the
exact command, log path, and the most useful diagnostic headers/locations.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "target"
ARMV7_TARGET = "armv7-unknown-linux-musleabihf"
RISCV64_TARGET = "riscv64gc-unknown-linux-musl"
RISCV32_TARGET = "riscv32gc-unknown-linux-musl"
X64_DARWIN_TARGET = "x86_64-apple-darwin"
X64_LINUX_TARGET = "x86_64-unknown-linux-gnu"
TESTSUITE_DIR = Path(os.environ.get("TESTSUITE_DIR", TARGET_DIR / "webassembly-testsuite"))
FULL_RUNTIME_FEATURES = "jit,wasi,validator,guard-pages"
EMULATOR_BACKEND_FEATURES = {"backend-emu64", "backend-emu32"}
RISCV32_LINKER = ROOT / "scripts" / "zig-riscv32-linux-musl-cc.sh"
RISCV64_SELECT_REGRESSION_ARGS = ["select", "call", "call_indirect", "if"]
RISCV32_SELECT_REGRESSION_ARGS = RISCV64_SELECT_REGRESSION_ARGS


CORE_OPTIONAL_FEATURES = [
    "backend-emu64",
    "backend-emu32",
    "wasi",
    "validator",
    "call-trace",
    "guard-pages",
    "ir-dump",
    "jitdump",
    "memprof",
    "thumb2-test",
]

CLI_OPTIONAL_FEATURES = [
    "call-trace",
    "memprof",
    "thumb2-test",
]

DEFAULT_TARGET_ROWS = [
    ("arm64-darwin", "aarch64-apple-darwin", "hosted", "", ""),
    ("arm64-linux", "aarch64-unknown-linux-gnu", "hosted", "", ""),
    ("x64-darwin", "x86_64-apple-darwin", "hosted", "", ""),
    ("x64-linux", "x86_64-unknown-linux-gnu", "hosted", "", ""),
    ("x64-windows", "x86_64-pc-windows-gnu", "hosted", "", ""),
]

EXTRA_TARGET_ROWS = [
    ("armv7-linux", ARMV7_TARGET, "hosted", "", ""),
    ("riscv64-linux", RISCV64_TARGET, "hosted", "", ""),
    ("riscv32-linux", RISCV32_TARGET, "hosted", "", ""),
    (
        "armv7m-linux",
        ARMV7_TARGET,
        "hosted",
        "--no-default-features --features jit,thumb2-test",
        "--no-default-features --features jit,thumb2-test",
    ),
]

BARE_SMOKE_TARGETS = [
    "thumbv8m.main-none-eabihf",
    "riscv32imac-unknown-none-elf",
]

@dataclass
class Host:
    kind: str
    machine: str

    @property
    def is_posix_runtime(self) -> bool:
        return self.kind in {"macos", "linux", "wsl"}


@dataclass
class Diagnostic:
    kind: str
    message: str
    location: Optional[str] = None

    def one_line(self) -> str:
        if self.location:
            return f"{self.location}: {self.message}"
        return self.message


@dataclass
class StepResult:
    index: int
    name: str
    status: str
    command: str
    log_path: Optional[Path]
    rc: int = 0
    warnings: int = 0
    errors: int = 0
    seconds: float = 0.0
    diagnostics: List[Diagnostic] = field(default_factory=list)
    message: Optional[str] = None


def detect_host() -> Host:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return Host("macos", machine)
    if system == "Linux":
        version = ""
        try:
            version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
        if "microsoft" in version.lower() or os.environ.get("WSL_DISTRO_NAME"):
            return Host("wsl", machine)
        return Host("linux", machine)
    if system == "Windows":
        return Host("windows", machine)
    return Host(system.lower(), machine)


def windows_path_to_wsl(path_text: str) -> Optional[str]:
    path = PureWindowsPath(path_text)
    drive = path.drive
    if len(drive) != 2 or drive[1] != ":":
        return None
    parts = [part for part in path.parts[1:] if part not in {"\\", "/"}]
    suffix = "/".join(parts)
    prefix = f"/mnt/{drive[0].lower()}"
    return f"{prefix}/{suffix}" if suffix else prefix


WINDOWS_TARGET_MATRIX_LABELS = {
    "arm64-darwin",
    "arm64-linux",
    "x64-darwin",
    "x64-windows",
}

UNSUPPORTED_ON_WINDOWS_TARGET_LABELS = {
    "arm64-darwin",
    "x64-darwin",
}


def wsl_env_overrides() -> List[str]:
    overrides: List[str] = []
    for name in ("TESTSUITE_DIR",):
        value = os.environ.get(name)
        if not value:
            continue
        converted = windows_path_to_wsl(value)
        overrides.append(f"{name}={converted or value}")
    overrides.append("SF_CHECK_SKIP_BARE_SMOKE=1")
    overrides.append(f"SF_CHECK_SKIP_TARGET_LABELS={','.join(sorted(WINDOWS_TARGET_MATRIX_LABELS))}")
    overrides.append("SF_CHECK_SUPPRESS_SUMMARY=1")
    return overrides


def reexec_in_wsl(argv: Sequence[str]) -> int:
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        print(
            "Silverfir check: native Windows detected, but wsl.exe was not found. "
            "Install WSL or run this from a WSL shell.",
            file=sys.stderr,
        )
        return 1

    linux_root = windows_path_to_wsl(str(ROOT.resolve()))
    if linux_root is None:
        print(
            f"Silverfir check: cannot map repository path to WSL: {ROOT.resolve()}",
            file=sys.stderr,
        )
        return 1

    cmd: List[str] = [wsl, "--cd", linux_root]
    env_args = wsl_env_overrides()
    if env_args:
        cmd.extend(["env", *env_args])
    cmd.extend(["python3", "-u", "scripts/check.py", *argv])

    print(f"Silverfir check: native Windows detected; delegating to WSL at {linux_root}", flush=True)
    try:
        completed = subprocess.run(cmd, check=False)
    except OSError as exc:
        print(f"Silverfir check: failed to launch WSL: {exc}", file=sys.stderr)
        return 1
    return completed.returncode


def shlex_join(argv: Sequence[object]) -> str:
    import shlex

    return shlex.join(str(a) for a in argv)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def clean_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def exe_suffix_for_target(target: Optional[str], host: Host) -> str:
    if target and "windows" in target:
        return ".exe"
    if target is None and host.kind == "windows":
        return ".exe"
    return ""


def profile_dir(profile_name: str) -> str:
    return "release" if profile_name == "release" else "debug"


def binary_path(bin_name: str, profile_name: str, host: Host, target: Optional[str] = None) -> Path:
    suffix = exe_suffix_for_target(target, host)
    if target:
        return TARGET_DIR / target / profile_dir(profile_name) / f"{bin_name}{suffix}"
    return TARGET_DIR / profile_dir(profile_name) / f"{bin_name}{suffix}"


def feature_args(spec: str) -> List[str]:
    if spec == "DEFAULT":
        return []
    if spec == "ALL":
        return ["--all-features"]
    if spec == "EMPTY":
        return ["--no-default-features"]
    return ["--no-default-features", "--features", spec]


def feature_spec_enables_emulator_backend(spec: Optional[str]) -> bool:
    # Emulator backends are smoke coverage in check.py. Keep compiling and
    # running them, but do not let warnings from those rows fail strict checks.
    if spec is None or spec in {"DEFAULT", "EMPTY"}:
        return False
    if spec == "ALL":
        return True
    return any(feature in EMULATOR_BACKEND_FEATURES for feature in spec.split(","))


def split_args(text: str) -> List[str]:
    if not text:
        return []
    import shlex

    return shlex.split(text)


def x64_target_for_host(host: Host) -> Optional[str]:
    if host.kind == "macos":
        return X64_DARWIN_TARGET
    if host.kind in {"linux", "wsl"}:
        return X64_LINUX_TARGET
    return None


_ROSETTA_PROBE_CACHE: Optional[bool] = None


def _has_rosetta_x86_64() -> bool:
    """Detect whether the host can run x86_64 Mach-O binaries via Rosetta 2.

    Apple Silicon Macs with Rosetta installed can run x86_64 binaries
    transparently when invoked via `arch -x86_64`. We probe by trying to
    execute a known x86_64 helper and caching the result.
    """
    global _ROSETTA_PROBE_CACHE
    if _ROSETTA_PROBE_CACHE is not None:
        return _ROSETTA_PROBE_CACHE
    try:
        proc = subprocess.run(
            ["/usr/bin/arch", "-x86_64", "/usr/bin/true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        _ROSETTA_PROBE_CACHE = proc.returncode == 0
    except (FileNotFoundError, OSError):
        _ROSETTA_PROBE_CACHE = False
    return _ROSETTA_PROBE_CACHE


def is_runnable_x64_host(host: Host) -> bool:
    if host.kind in {"linux", "wsl"} and host.machine in {"x86_64", "amd64"}:
        return True
    if host.kind == "macos":
        if host.machine in {"x86_64", "amd64"}:
            return True
        # Apple Silicon: x86_64 binaries run transparently under Rosetta 2,
        # which the spectest/wasitest paths invoke explicitly via
        # `arch -x86_64`. Skip x64 only if Rosetta isn't installed.
        return _has_rosetta_x86_64()
    return False


def cargo_env(target: Optional[str], env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    # Per-target linkers and codegen flags live in .cargo/config.toml. Setting
    # RUSTFLAGS here would *replace* `target.<triple>.rustflags` rather than
    # merge with it, so this runner deliberately sets no RUSTFLAGS at all;
    # strict mode fails on warnings via the WARN step count in final_report()
    # instead of via `-D warnings`.
    proc_env = dict(env or {})
    if target == RISCV32_TARGET:
        proc_env.setdefault("ZIG_GLOBAL_CACHE_DIR", str(TARGET_DIR / "zig-cache"))
    return proc_env


def cargo_argv(subcommand: str, target: Optional[str]) -> List[str]:
    if target == RISCV32_TARGET:
        return ["cargo", "+nightly", subcommand, "-Z", "build-std=std,panic_abort"]
    return ["cargo", subcommand]


DIAG_HEADER_RE = re.compile(r"^(error(?:\[[^\]]+\])?:|warning(?:\[[^\]]+\])?:)\s*(.*)$")
LOCATION_RE = re.compile(r"^\s+-->\s+(.+)$")
GENERATED_WARNINGS_RE = re.compile(r"generated\s+(\d+)\s+warnings?")
RELEVANT_RE = re.compile(
    r"(error(?:\[|:)|warning(?:\[|:)|fatal|failed|failure|panicked|not found|missing|FAIL)",
    re.IGNORECASE,
)
APP_ERROR_RE = re.compile(
    r"(\bERROR\b.*\b(FAIL|ERROR)\b|Some tests failed|Failed:\s+[1-9])",
    re.IGNORECASE,
)


def parse_log(log_path: Path, rc: int, max_diagnostics: int) -> Tuple[int, int, List[Diagnostic]]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return (1 if rc else 0, 0, [])

    errors = 0
    warnings = 0
    diagnostics: List[Diagnostic] = []

    for i, line in enumerate(lines):
        match = DIAG_HEADER_RE.match(line)
        if not match:
            continue
        kind = "error" if match.group(1).startswith("error") else "warning"
        if kind == "error":
            errors += 1
        else:
            warnings += 1

        location = None
        for lookahead in lines[i + 1 : i + 8]:
            loc_match = LOCATION_RE.match(lookahead)
            if loc_match:
                location = loc_match.group(1).strip()
                break
        if len(diagnostics) < max_diagnostics:
            diagnostics.append(Diagnostic(kind, f"{kind}: {match.group(2).strip()}", location))

    for line in lines:
        stripped = line.strip()
        if not APP_ERROR_RE.search(stripped):
            continue
        errors += 1
        if len(diagnostics) < max_diagnostics:
            diagnostics.append(Diagnostic("error", stripped))

    generated_max = 0
    for line in lines:
        gen = GENERATED_WARNINGS_RE.search(line)
        if gen:
            generated_max = max(generated_max, int(gen.group(1)))
    warnings = max(warnings, generated_max)

    if rc != 0 and errors == 0:
        errors = 1
        if not diagnostics:
            relevant = [line.strip() for line in lines if RELEVANT_RE.search(line)]
            tail = [line.strip() for line in lines[-20:] if line.strip()]
            for line in (relevant or tail)[:max_diagnostics]:
                diagnostics.append(Diagnostic("error", line))

    return errors, warnings, diagnostics


class CheckRunner:
    def __init__(
        self,
        host: Host,
        strict: bool,
        install_targets: bool,
        max_diagnostics: int,
        phase: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        self.host = host
        self.strict = strict
        self.install_targets = install_targets
        self.max_diagnostics = max_diagnostics
        self.phase = phase
        self.dry_run = dry_run
        log_suffix = f"-{phase}" if phase else ""
        self.log_dir = TARGET_DIR / f"check{log_suffix}-logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.log_dir / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[StepResult] = []
        self.step_index = 0
        self.colima_started = False
        self.installed_rust_targets: Optional[set[str]] = None

    def run(
        self,
        name: str,
        argv: Sequence[object],
        *,
        env: Optional[Dict[str, str]] = None,
        cwd: Path = ROOT,
        log_name: Optional[str] = None,
        ignore_warnings: bool = False,
    ) -> StepResult:
        self.step_index += 1
        log_path = self.log_dir / f"{log_name or clean_label(name)}.log"
        display = shlex_join(argv)

        # Dry-run short-circuit: record the step as OK without launching any
        # subprocess. Lets `--dry-run` surface the full per-step list for
        # tuning the progress output without a full build cycle.
        if self.dry_run:
            result = StepResult(
                index=self.step_index,
                name=name,
                status="OK",
                command=display,
                log_path=log_path,
                rc=0,
                warnings=0,
                errors=0,
                seconds=0.0,
                diagnostics=[],
            )
            self.results.append(result)
            self._print_step_result(result)
            return result

        start = time.monotonic()
        proc_env = os.environ.copy()
        if env:
            proc_env.update({k: str(v) for k, v in env.items()})

        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                log.write(f"$ {display}\n")
                log.write(f"# cwd: {cwd}\n\n")
                completed = subprocess.run(
                    [str(a) for a in argv],
                    cwd=str(cwd),
                    env=proc_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            rc = completed.returncode
        except FileNotFoundError as exc:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                log.write(f"$ {display}\n")
                log.write(f"# cwd: {cwd}\n\n")
                log.write(f"missing executable: {exc.filename}\n")
            rc = 127

        seconds = time.monotonic() - start
        errors, warnings, diagnostics = parse_log(log_path, rc, self.max_diagnostics)
        if ignore_warnings and rc == 0:
            warnings = 0
            diagnostics = [d for d in diagnostics if d.kind != "warning"]

        if rc != 0:
            status = "FAIL"
        elif warnings > 0:
            status = "WARN"
        else:
            status = "OK"

        result = StepResult(
            index=self.step_index,
            name=name,
            status=status,
            command=display,
            log_path=log_path,
            rc=rc,
            warnings=warnings,
            errors=errors,
            seconds=seconds,
            diagnostics=diagnostics,
        )
        self.results.append(result)
        self._print_step_result(result)
        return result

    def skip(self, name: str, reason: str) -> StepResult:
        self.step_index += 1
        result = StepResult(
            index=self.step_index,
            name=name,
            status="SKIP",
            command="-",
            log_path=None,
            message=reason,
        )
        self.results.append(result)
        self._print_step_result(result)
        return result

    def fail(self, name: str, message: str, *, command: str = "-") -> StepResult:
        self.step_index += 1
        log_path = self.log_dir / f"{clean_label(name)}.log"
        log_path.write_text(message.rstrip() + "\n", encoding="utf-8")
        result = StepResult(
            index=self.step_index,
            name=name,
            status="FAIL",
            command=command,
            log_path=log_path,
            rc=1,
            errors=1,
            diagnostics=[Diagnostic("error", message)],
            message=message,
        )
        self.results.append(result)
        self._print_step_result(result)
        return result

    def _print_step_result(self, result: StepResult) -> None:
        duration = f"{result.seconds:.1f}s" if result.seconds else ""
        # Status prefix is a fixed-width tag so progress is easy to scan and
        # grep. Step index is dropped — it carried no information users
        # acted on and made it harder to spot the status at a glance.
        if result.status == "OK":
            print(f"[OK]   {result.name} {duration}".rstrip())
        elif result.status == "WARN":
            print(f"[WARN] {result.name}")
            print(f"       {result.warnings} warning(s), log: {rel(result.log_path)}")
        elif result.status == "FAIL":
            print(f"[FAIL] {result.name}")
            print(f"       {result.errors} error(s), log: {rel(result.log_path)}")
        else:
            print(f"[SKIP] {result.name}")
            print(f"       {result.message}")

    def final_report(self, *, print_report: bool = True) -> int:
        counts = {status: 0 for status in ("OK", "WARN", "FAIL", "SKIP")}
        for result in self.results:
            counts[result.status] += 1

        fail_on_warn = self.strict and counts["WARN"] > 0
        rc = 1 if counts["FAIL"] > 0 or fail_on_warn else 0
        if not print_report:
            return rc

        print()
        print("=== Summary ===")
        phase = f"phase={self.phase} " if self.phase else ""
        print(
            f"{phase}host={self.host.kind}/{self.host.machine} "
            f"ok={counts['OK']} warn={counts['WARN']} fail={counts['FAIL']} skip={counts['SKIP']}"
        )
        print(f"logs: {rel(self.log_dir)}")

        issues = [r for r in self.results if r.status in {"FAIL", "WARN"}]
        if issues:
            print()
            print("=== Issues ===")
            for result in issues:
                print(f"{result.status}: {result.name}")
                if result.command != "-":
                    print(f"  command: {result.command}")
                if result.log_path:
                    print(f"  log: {rel(result.log_path)}")
                if result.diagnostics:
                    for diagnostic in result.diagnostics:
                        print(f"  - {diagnostic.one_line()}")
                elif result.message:
                    print(f"  - {result.message}")

        # Skipped steps are already printed inline with their reason, so
        # omit the trailing summary section — it was duplicate noise.

        print()
        if rc:
            print("FAILED")
            return rc
        print("PASSED")
        return rc


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def rustup_installed_targets() -> Optional[set[str]]:
    if not command_exists("rustup"):
        return None
    proc = subprocess.run(
        ["rustup", "target", "list", "--installed"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def cached_rustup_installed_targets(runner: CheckRunner) -> Optional[set[str]]:
    if runner.installed_rust_targets is None:
        runner.installed_rust_targets = rustup_installed_targets()
    return runner.installed_rust_targets


def rustup_target_scope(runner: CheckRunner) -> str:
    if runner.host.kind == "wsl":
        return "WSL rustup target"
    if runner.host.kind == "windows":
        return "Windows rustup target"
    if runner.host.kind == "macos":
        return "macOS rustup target"
    return f"{runner.host.kind} rustup target"


def missing_rust_target_message(runner: CheckRunner, target: str) -> str:
    scope = rustup_target_scope(runner)
    if runner.host.kind == "wsl":
        return f"missing {scope} {target}; use --install-targets or run in WSL: rustup target add {target}"
    return f"missing {scope} {target}; use --install-targets or run: rustup target add {target}"


def install_rust_target(runner: CheckRunner, target: str) -> bool:
    result = runner.run(
        f"install {rustup_target_scope(runner)} {target}",
        ["rustup", "target", "add", target],
        log_name=f"rustup-target-add-{clean_label(target)}",
    )
    if result.status != "FAIL" and runner.installed_rust_targets is not None:
        # In dry-run the rustup command is not actually executed. Marking the
        # target available still models the rest of an --install-targets run and
        # avoids duplicate install rows for targets used by multiple checks.
        runner.installed_rust_targets.add(target)
    return result.status != "FAIL"


def ensure_rust_target(runner: CheckRunner, target: str) -> bool:
    if target == RISCV32_TARGET:
        return ensure_riscv32_build_toolchain(runner)
    installed = cached_rustup_installed_targets(runner)
    if installed is None:
        runner.fail(
            f"rustup target check ({target})",
            f"{rustup_target_scope(runner)} list could not be read; rustup is required to verify cross-compile targets.",
            command="rustup target list --installed",
        )
        return False
    if target in installed:
        return True
    if runner.install_targets:
        return install_rust_target(runner, target)
    runner.fail(
        f"{rustup_target_scope(runner)} missing ({target})",
        missing_rust_target_message(runner, target),
        command=f"rustup target add {target}",
    )
    return False


def ensure_riscv32_build_toolchain(runner: CheckRunner) -> bool:
    if runner.dry_run:
        return True
    if not command_exists("cargo"):
        runner.fail("riscv32 build toolchain", "missing cargo.")
        return False
    if not command_exists("zig"):
        runner.fail(
            "riscv32 build toolchain",
            "missing zig; RV32 musl linking uses scripts/zig-riscv32-linux-musl-cc.sh.",
        )
        return False
    nightly = subprocess.run(
        ["cargo", "+nightly", "--version"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if nightly.returncode != 0:
        runner.fail(
            "riscv32 build toolchain",
            "missing usable nightly toolchain; RV32 uses cargo +nightly -Z build-std.",
        )
        return False
    if not RISCV32_LINKER.exists():
        runner.fail("riscv32 build toolchain", f"missing linker wrapper: {RISCV32_LINKER}")
        return False
    return True


def ensure_testsuite(runner: CheckRunner) -> bool:
    if TESTSUITE_DIR.is_dir():
        return True
    runner.fail(
        "setup: spectest testsuite",
        f"missing testsuite directory: {TESTSUITE_DIR}. Set TESTSUITE_DIR or populate target/webassembly-testsuite.",
    )
    return False


def ensure_armv7_runner(runner: CheckRunner) -> bool:
    if runner.host.kind == "macos":
        if not command_exists("colima"):
            runner.fail("armv7 qemu runner", "missing colima; install Colima or run ARMv7 checks on Linux/WSL.")
            return False
        status = subprocess.run(
            ["colima", "status"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if status.returncode != 0:
            start = runner.run("colima start", ["colima", "start"], log_name="colima-start")
            if start.status == "FAIL":
                return False
            runner.colima_started = True
        qemu = subprocess.run(
            ["colima", "ssh", "--", "which", "qemu-arm-static"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if qemu.returncode != 0:
            update = runner.run(
                "colima install qemu-user-static (apt update)",
                ["colima", "ssh", "--", "sudo", "apt-get", "update", "-qq"],
                log_name="colima-apt-update",
            )
            if update.status == "FAIL":
                return False
            install = runner.run(
                "colima install qemu-user-static",
                ["colima", "ssh", "--", "sudo", "apt-get", "install", "-y", "-qq", "qemu-user-static"],
                log_name="colima-install-qemu-user-static",
            )
            return install.status != "FAIL"
        return True

    if runner.host.kind in {"linux", "wsl"}:
        if command_exists("qemu-arm-static"):
            return True
        runner.fail(
            "armv7 qemu runner",
            "missing qemu-arm-static; on Debian/Ubuntu/WSL install it with: sudo apt-get install -y qemu-user-static",
        )
        return False

    runner.skip("armv7 qemu runner", "ARMv7 runtime checks require macOS+Colima or Linux/WSL.")
    return False


def ensure_riscv64_runner(runner: CheckRunner) -> bool:
    if runner.host.kind == "macos":
        if not command_exists("colima"):
            runner.fail("riscv64 qemu runner", "missing colima; install Colima or run RV64 checks on Linux/WSL.")
            return False
        status = subprocess.run(
            ["colima", "status"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if status.returncode != 0:
            start = runner.run("colima start", ["colima", "start"], log_name="colima-start")
            if start.status == "FAIL":
                return False
            runner.colima_started = True
            if runner.dry_run:
                return True
        qemu = subprocess.run(
            ["colima", "ssh", "--", "which", "qemu-riscv64-static"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if qemu.returncode != 0:
            update = runner.run(
                "colima install qemu-user-static (apt update)",
                ["colima", "ssh", "--", "sudo", "apt-get", "update", "-qq"],
                log_name="colima-apt-update-riscv64",
            )
            if update.status == "FAIL":
                return False
            install = runner.run(
                "colima install qemu-user-static",
                ["colima", "ssh", "--", "sudo", "apt-get", "install", "-y", "-qq", "qemu-user-static"],
                log_name="colima-install-qemu-user-static-riscv64",
            )
            return install.status != "FAIL"
        return True

    if runner.host.kind in {"linux", "wsl"}:
        if command_exists("qemu-riscv64-static"):
            return True
        runner.fail(
            "riscv64 qemu runner",
            "missing qemu-riscv64-static; on Debian/Ubuntu/WSL install it with: sudo apt-get install -y qemu-user-static",
        )
        return False

    runner.skip("riscv64 qemu runner", "RV64 runtime checks require macOS+Colima or Linux/WSL.")
    return False


def ensure_riscv32_runner(runner: CheckRunner) -> bool:
    if runner.host.kind == "macos":
        if not command_exists("colima"):
            runner.fail("riscv32 qemu runner", "missing colima; install Colima or run RV32 checks on Linux/WSL.")
            return False
        status = subprocess.run(
            ["colima", "status"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if status.returncode != 0:
            start = runner.run("colima start", ["colima", "start"], log_name="colima-start")
            if start.status == "FAIL":
                return False
            runner.colima_started = True
            if runner.dry_run:
                return True
        qemu = subprocess.run(
            ["colima", "ssh", "--", "which", "qemu-riscv32-static"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if qemu.returncode != 0:
            update = runner.run(
                "colima install qemu-user-static (apt update)",
                ["colima", "ssh", "--", "sudo", "apt-get", "update", "-qq"],
                log_name="colima-apt-update-riscv32",
            )
            if update.status == "FAIL":
                return False
            install = runner.run(
                "colima install qemu-user-static",
                ["colima", "ssh", "--", "sudo", "apt-get", "install", "-y", "-qq", "qemu-user-static"],
                log_name="colima-install-qemu-user-static-riscv32",
            )
            return install.status != "FAIL"
        return True

    if runner.host.kind in {"linux", "wsl"}:
        if command_exists("qemu-riscv32-static"):
            return True
        runner.fail(
            "riscv32 qemu runner",
            "missing qemu-riscv32-static; on Debian/Ubuntu/WSL install it with: sudo apt-get install -y qemu-user-static",
        )
        return False

    runner.skip("riscv32 qemu runner", "RV32 runtime checks require macOS+Colima or Linux/WSL.")
    return False


def cleanup_colima(runner: CheckRunner) -> None:
    if not runner.colima_started:
        return
    runner.run("colima stop", ["colima", "stop"], log_name="colima-stop")


def cargo_check(
    runner: CheckRunner,
    name: str,
    package: Optional[str],
    *,
    profile_name: str = "dev",
    target: Optional[str] = None,
    feature_spec: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Dict[str, str]] = None,
    log_name: Optional[str] = None,
    cwd: Path = ROOT,
    ignore_warnings: bool = False,
) -> StepResult:
    ignore_warnings = ignore_warnings or feature_spec_enables_emulator_backend(feature_spec)
    argv = cargo_argv("check", target)
    if profile_name == "release":
        argv.append("--release")
    if target:
        argv.extend(["--target", target])
    if package:
        argv.extend(["-p", package])
    if feature_spec is not None:
        argv.extend(feature_args(feature_spec))
    if extra_args:
        argv.extend(extra_args)

    return runner.run(
        name,
        argv,
        env=cargo_env(target, env),
        cwd=cwd,
        log_name=log_name,
        ignore_warnings=ignore_warnings,
    )


def cargo_build(
    runner: CheckRunner,
    name: str,
    package: str,
    *,
    profile_name: str,
    target: Optional[str],
    features: str,
    log_name: Optional[str] = None,
) -> StepResult:
    argv = cargo_argv("build", target)
    if profile_name == "release":
        argv.append("--release")
    if target:
        argv.extend(["--target", target])
    argv.extend(["-p", package, "--no-default-features", "--features", features])
    return runner.run(
        name,
        argv,
        env=cargo_env(target),
        log_name=log_name,
        ignore_warnings=feature_spec_enables_emulator_backend(features),
    )


def cargo_build_package(
    runner: CheckRunner,
    name: str,
    package: str,
    *,
    profile_name: str,
    target: Optional[str],
    log_name: Optional[str] = None,
) -> StepResult:
    argv = cargo_argv("build", target)
    if profile_name == "release":
        argv.append("--release")
    if target:
        argv.extend(["--target", target])
    argv.extend(["-p", package])
    return runner.run(name, argv, env=cargo_env(target), log_name=log_name)


def skip_after_failed_dependency(runner: CheckRunner, name: str, dependency: StepResult) -> None:
    runner.skip(name, f"dependency failed: {dependency.name}; see {rel(dependency.log_path)}")


def run_workspace_tests(runner: CheckRunner, profile_name: str = "debug") -> None:
    argv = ["cargo", "test", "--workspace"]
    if profile_name == "release":
        argv.append("--release")
    runner.run(
        f"cargo test: workspace ({profile_name})",
        argv,
        env=cargo_env(None),
        log_name=f"workspace-tests-{profile_name}",
    )


def full_core_combos() -> List[Tuple[str, str]]:
    combos = [
        ("default", "DEFAULT"),
        ("all-features", "ALL"),
        ("only-jit", "jit"),
    ]
    for feature in CORE_OPTIONAL_FEATURES:
        combos.append((f"jit+{feature}", f"jit,{feature}"))
    for i, feature_a in enumerate(CORE_OPTIONAL_FEATURES):
        for feature_b in CORE_OPTIONAL_FEATURES[i + 1 :]:
            combos.append((f"jit+{feature_a}+{feature_b}", f"jit,{feature_a},{feature_b}"))
    combos.extend(
        [
            ("cli", "jit,wasi,guard-pages"),
            ("cli+tracing", "jit,wasi,guard-pages,call-trace"),
            ("cli+ir-dump", "jit,wasi,guard-pages,ir-dump"),
            ("cli+jitdump", "jit,wasi,guard-pages,jitdump"),
            ("cli+memprof", "jit,wasi,guard-pages,memprof"),
            ("cli+emu64", "jit,wasi,guard-pages,backend-emu64"),
            ("cli+emu32", "jit,wasi,guard-pages,backend-emu32"),
            ("spectest", "jit,validator,guard-pages"),
            ("spectest+emu64", "jit,validator,guard-pages,backend-emu64"),
            ("spectest+emu32", "jit,validator,guard-pages,backend-emu32"),
            ("full-runtime", FULL_RUNTIME_FEATURES),
            ("full-runtime+emu64", f"{FULL_RUNTIME_FEATURES},backend-emu64"),
            ("full-runtime+emu32", f"{FULL_RUNTIME_FEATURES},backend-emu32"),
            ("full-dev", "jit,wasi,validator,call-trace,guard-pages,ir-dump,jitdump,memprof"),
            ("cli+thumb2", "jit,wasi,guard-pages,thumb2-test"),
            ("spectest+thumb2", "jit,validator,guard-pages,thumb2-test"),
        ]
    )
    return combos


def full_cli_combos() -> List[Tuple[str, str]]:
    combos = [
        ("default", "DEFAULT"),
        ("all-features", "ALL"),
        ("only-jit", "jit"),
    ]
    for feature in CLI_OPTIONAL_FEATURES:
        combos.append((f"jit+{feature}", f"jit,{feature}"))
    for i, feature_a in enumerate(CLI_OPTIONAL_FEATURES):
        for feature_b in CLI_OPTIONAL_FEATURES[i + 1 :]:
            combos.append((f"jit+{feature_a}+{feature_b}", f"jit,{feature_a},{feature_b}"))
    return combos


def run_feature_checks(runner: CheckRunner, *, profiles: Sequence[str]) -> None:
    core_combos = full_core_combos()
    cli_combos = full_cli_combos()

    for profile_name in profiles:
        for label, spec in core_combos:
            cargo_check(
                runner,
                f"cargo check: core feature {label} ({profile_name})",
                "sf-nano-core",
                profile_name=profile_name,
                feature_spec=spec,
                log_name=f"feature-core-{clean_label(label)}-{profile_name}",
            )

        for label, spec in cli_combos:
            cargo_check(
                runner,
                f"cargo check: cli feature {label} ({profile_name})",
                "sf-nano-cli",
                profile_name=profile_name,
                feature_spec=spec,
                log_name=f"feature-cli-{clean_label(label)}-{profile_name}",
            )

    for profile_name in profiles:
        cargo_check(
            runner,
            f"cargo check: workspace default features ({profile_name})",
            None,
            profile_name=profile_name,
            extra_args=["--workspace"],
            log_name=f"workspace-{profile_name}",
        )

    cli_minimal = ROOT / "sf-nano-cli-minimal"
    if cli_minimal.is_dir():
        for profile_name in profiles:
            cargo_check(
                runner,
                f"cargo check: cli-minimal ({profile_name})",
                None,
                profile_name=profile_name,
                cwd=cli_minimal,
                log_name=f"cli-minimal-{profile_name}",
            )


def run_target_matrix(
    runner: CheckRunner,
    *,
    include_all: bool,
    include_bare_smoke: bool = True,
    only_labels: Optional[set[str]] = None,
    exclude_labels: Optional[set[str]] = None,
    unsupported_labels: Optional[set[str]] = None,
) -> None:
    if not command_exists("rustup"):
        runner.fail("target matrix prerequisites", "rustup is required for target matrix checks.")
        return

    rows = list(DEFAULT_TARGET_ROWS)
    if include_all:
        rows.extend(EXTRA_TARGET_ROWS)

    package_keys = ["core", "cli", "spectest", "wasitest"]
    for label, target, kind, core_extra, cli_spectest_override in rows:
        if only_labels is not None and label not in only_labels:
            continue
        if exclude_labels is not None and label in exclude_labels:
            continue
        if unsupported_labels is not None and label in unsupported_labels:
            runner.skip(
                f"target matrix {label}",
                f"{target} cannot be run on this Windows+WSL machine; compile-only coverage is not treated as platform validation.",
            )
            continue
        if not target_installed_or_install(runner, target, f"target matrix {label}"):
            continue

        if kind == "hosted":
            for package_key in package_keys:
                run_target_package_check(
                    runner,
                    label,
                    target,
                    package_key,
                    core_extra,
                    cli_spectest_override,
                )
        else:
            run_target_package_check(runner, label, target, "core", core_extra, "")

    if include_bare_smoke:
        run_bare_smoke_checks(runner)


def run_bare_smoke_checks(runner: CheckRunner) -> None:
    bare_smoke_dir = ROOT / "sf-nano-bare-smoke"
    if not bare_smoke_dir.is_dir():
        return
    for bare_target in BARE_SMOKE_TARGETS:
        if not target_installed_or_install(runner, bare_target, f"bare-smoke {bare_target}"):
            continue
        target_dir = TARGET_DIR / "check-targets-build" / f"bare-smoke-{bare_target}"
        target_dir.mkdir(parents=True, exist_ok=True)
        runner.run(
            f"bare-smoke cargo check ({bare_target})",
            ["cargo", "check", "--target", bare_target],
            cwd=bare_smoke_dir,
            env=cargo_env(bare_target, {"CARGO_TARGET_DIR": str(target_dir)}),
            log_name=f"target-bare-smoke-{clean_label(bare_target)}",
        )


def target_installed_or_install(runner: CheckRunner, target: str, name: str) -> bool:
    if target == RISCV32_TARGET:
        return ensure_riscv32_build_toolchain(runner)
    installed = cached_rustup_installed_targets(runner)
    if installed is None:
        runner.fail(
            name,
            f"could not read installed {rustup_target_scope(runner)} list.",
            command="rustup target list --installed",
        )
        return False
    if target in installed:
        return True
    if runner.install_targets:
        return install_rust_target(runner, target)
    runner.fail(name, missing_rust_target_message(runner, target), command=f"rustup target add {target}")
    return False


def run_target_package_check(
    runner: CheckRunner,
    label: str,
    target: str,
    package_key: str,
    core_extra: str,
    cli_spectest_override: str,
) -> None:
    if package_key == "core":
        package = "sf-nano-core"
        extra = split_args(core_extra)
    elif package_key == "cli":
        package = "sf-nano-cli"
        extra = split_args(cli_spectest_override) or ["--no-default-features", "--features", "jit"]
    elif package_key == "spectest":
        package = "sf-nano-spectest"
        extra = split_args(cli_spectest_override) or ["--no-default-features", "--features", "jit"]
    elif package_key == "wasitest":
        package = "sf-nano-wasitest"
        extra = []
    else:
        raise ValueError(package_key)

    target_dir = TARGET_DIR / "check-targets-build" / f"{label}-{package_key}"
    target_dir.mkdir(parents=True, exist_ok=True)
    env = cargo_env(target, {"CARGO_TARGET_DIR": str(target_dir)})

    argv = [*cargo_argv("check", target), "--target", target, "-p", package, *extra]
    runner.run(
        f"cargo check: target {label}/{package_key}",
        argv,
        env=env,
        log_name=f"target-{clean_label(label)}-{package_key}",
    )


def run_spectest_suite(runner: CheckRunner, *, profiles: Sequence[str], extra_args: Sequence[str]) -> None:
    if not ensure_testsuite(runner):
        return

    env = {"TESTSUITE_DIR": str(TESTSUITE_DIR)}
    for profile_name in profiles:
        native_build = cargo_build(
            runner,
            f"cargo build: spectest native ({profile_name})",
            "sf-nano-spectest",
            profile_name=profile_name,
            target=None,
            features="jit",
            log_name=f"spectest-build-native-{profile_name}",
        )
        host_bin = binary_path("sf-nano-spectest", profile_name, runner.host)
        if native_build.status == "FAIL":
            skip_after_failed_dependency(runner, f"run: spectest native ({profile_name})", native_build)
        else:
            run_binary_if_present(
                runner,
                f"run: spectest native ({profile_name})",
                [host_bin, "--backend", "native", *extra_args],
                env=env,
                log_name=f"spectest-native-{profile_name}",
            )

        emu64_build = cargo_build(
            runner,
            f"cargo build: spectest emu64 ({profile_name})",
            "sf-nano-spectest",
            profile_name=profile_name,
            target=None,
            features="jit,backend-emu64",
            log_name=f"spectest-build-emu64-{profile_name}",
        )
        if emu64_build.status == "FAIL":
            skip_after_failed_dependency(runner, f"run: spectest emu64 ({profile_name})", emu64_build)
        else:
            run_binary_if_present(
                runner,
                f"run: spectest emu64 ({profile_name})",
                [host_bin, *extra_args],
                env=env,
                log_name=f"spectest-emu64-{profile_name}",
                ignore_warnings=True,
            )

        emu32_build = cargo_build(
            runner,
            f"cargo build: spectest emu32 ({profile_name})",
            "sf-nano-spectest",
            profile_name=profile_name,
            target=None,
            features="jit,backend-emu32",
            log_name=f"spectest-build-emu32-{profile_name}",
        )
        if emu32_build.status == "FAIL":
            skip_after_failed_dependency(runner, f"run: spectest emu32 ({profile_name})", emu32_build)
        else:
            run_binary_if_present(
                runner,
                f"run: spectest emu32 ({profile_name})",
                [host_bin, *extra_args],
                env=env,
                log_name=f"spectest-emu32-{profile_name}",
                ignore_warnings=True,
            )

        # Cross-arch runtime checks. They require QEMU user-mode (or Colima
        # on macOS) and, for riscv32, a working Zig + nightly Rust. CI jobs
        # for hosts not provisioned with that tooling set SF_CHECK_SKIP_CROSS_RUNTIME=1
        # to opt out of these rather than failing on missing prerequisites.
        if os.environ.get("SF_CHECK_SKIP_CROSS_RUNTIME"):
            runner.skip(
                f"run: spectest cross-arch ({profile_name})",
                "SF_CHECK_SKIP_CROSS_RUNTIME=1: x64/armv7/riscv64/riscv32 runtime checks deferred to the dedicated cross-arch CI job.",
            )
        else:
            run_x64_spectest(runner, profile_name, extra_args, env)
            run_riscv64_spectest(runner, profile_name, extra_args, env)
            run_riscv32_spectest(runner, profile_name, extra_args, env)
            run_armv7_spectests(runner, profile_name, extra_args, env)


def run_binary_if_present(
    runner: CheckRunner,
    name: str,
    argv: Sequence[object],
    *,
    env: Optional[Dict[str, str]] = None,
    log_name: Optional[str] = None,
    ignore_warnings: bool = False,
) -> Optional[StepResult]:
    binary_text = str(argv[0])
    binary = Path(binary_text)
    # In dry-run we never actually built the binary; skip the pre-check so
    # the step still surfaces in the planned step list as OK rather than
    # FAIL'ing on a missing artifact.
    if not runner.dry_run:
        if binary.is_absolute() or "/" in binary_text:
            exists = binary.exists()
        else:
            exists = shutil.which(binary_text) is not None
        if not exists:
            runner.fail(name, f"expected binary not found: {binary_text}", command=shlex_join(argv))
            return None
    return runner.run(name, argv, env=env, log_name=log_name, ignore_warnings=ignore_warnings)


def run_x64_spectest(
    runner: CheckRunner,
    profile_name: str,
    extra_args: Sequence[str],
    env: Dict[str, str],
) -> None:
    target = x64_target_for_host(runner.host)
    if target is None:
        runner.skip(f"run: spectest x64 ({profile_name})", "x64 runtime checks require macOS, Linux, or WSL.")
        return
    if not is_runnable_x64_host(runner.host):
        runner.skip(f"run: spectest x64 ({profile_name})", f"host architecture {runner.host.machine} cannot run x64 binary directly.")
        return
    if not ensure_rust_target(runner, target):
        return

    build = cargo_build(
        runner,
        f"cargo build: spectest x64 ({profile_name})",
        "sf-nano-spectest",
        profile_name=profile_name,
        target=target,
        features="jit",
        log_name=f"spectest-build-x64-{profile_name}",
    )
    if build.status == "FAIL":
        skip_after_failed_dependency(runner, f"run: spectest x64 ({profile_name})", build)
        return
    x64_bin = binary_path("sf-nano-spectest", profile_name, runner.host, target)
    argv: List[object]
    if runner.host.kind == "macos":
        argv = ["arch", "-x86_64", x64_bin, "--backend", "native", *extra_args]
    else:
        argv = [x64_bin, "--backend", "native", *extra_args]
    run_binary_if_present(runner, f"run: spectest x64 ({profile_name})", argv, env=env, log_name=f"spectest-x64-{profile_name}")


def run_riscv64_spectest(
    runner: CheckRunner,
    profile_name: str,
    extra_args: Sequence[str],
    env: Dict[str, str],
) -> None:
    if not ensure_rust_target(runner, RISCV64_TARGET):
        return
    if not ensure_riscv64_runner(runner):
        return

    build = cargo_build(
        runner,
        f"cargo build: spectest riscv64 ({profile_name})",
        "sf-nano-spectest",
        profile_name=profile_name,
        target=RISCV64_TARGET,
        features="jit",
        log_name=f"spectest-build-riscv64-{profile_name}",
    )
    if build.status == "FAIL":
        skip_after_failed_dependency(runner, f"run: spectest riscv64 ({profile_name})", build)
        return

    rv64_bin = binary_path("sf-nano-spectest", profile_name, runner.host, RISCV64_TARGET)
    smoke = run_riscv64_binary(
        runner,
        f"run: spectest riscv64 select/call smoke ({profile_name})",
        rv64_bin,
        ["--backend", "native", *RISCV64_SELECT_REGRESSION_ARGS],
        env,
        f"spectest-riscv64-select-call-smoke-{profile_name}",
    )
    if smoke is None:
        return
    if smoke.status == "FAIL":
        skip_after_failed_dependency(runner, f"run: spectest riscv64 ({profile_name})", smoke)
        return

    if list(extra_args) == RISCV64_SELECT_REGRESSION_ARGS:
        return
    run_riscv64_binary(
        runner,
        f"run: spectest riscv64 ({profile_name})",
        rv64_bin,
        ["--backend", "native", *extra_args],
        env,
        f"spectest-riscv64-{profile_name}",
    )


def run_riscv32_spectest(
    runner: CheckRunner,
    profile_name: str,
    extra_args: Sequence[str],
    env: Dict[str, str],
) -> None:
    if not ensure_rust_target(runner, RISCV32_TARGET):
        return
    if not ensure_riscv32_runner(runner):
        return

    build = cargo_build(
        runner,
        f"cargo build: spectest riscv32 ({profile_name})",
        "sf-nano-spectest",
        profile_name=profile_name,
        target=RISCV32_TARGET,
        features="jit",
        log_name=f"spectest-build-riscv32-{profile_name}",
    )
    if build.status == "FAIL":
        skip_after_failed_dependency(runner, f"run: spectest riscv32 ({profile_name})", build)
        return

    rv32_bin = binary_path("sf-nano-spectest", profile_name, runner.host, RISCV32_TARGET)
    smoke = run_riscv32_binary(
        runner,
        f"run: spectest riscv32 select/call smoke ({profile_name})",
        rv32_bin,
        ["--backend", "native", *RISCV32_SELECT_REGRESSION_ARGS],
        env,
        f"spectest-riscv32-select-call-smoke-{profile_name}",
    )
    if smoke is None:
        return
    if smoke.status == "FAIL":
        skip_after_failed_dependency(runner, f"run: spectest riscv32 ({profile_name})", smoke)
        return

    if list(extra_args) == RISCV32_SELECT_REGRESSION_ARGS:
        return
    run_riscv32_binary(
        runner,
        f"run: spectest riscv32 ({profile_name})",
        rv32_bin,
        ["--backend", "native", *extra_args],
        env,
        f"spectest-riscv32-{profile_name}",
    )


def run_armv7_spectests(
    runner: CheckRunner,
    profile_name: str,
    extra_args: Sequence[str],
    env: Dict[str, str],
) -> None:
    if not ensure_rust_target(runner, ARMV7_TARGET):
        return
    if not ensure_armv7_runner(runner):
        return

    build_a = cargo_build(
        runner,
        f"cargo build: spectest armv7a ({profile_name})",
        "sf-nano-spectest",
        profile_name=profile_name,
        target=ARMV7_TARGET,
        features="jit",
        log_name=f"spectest-build-armv7a-{profile_name}",
    )
    arm_bin = binary_path("sf-nano-spectest", profile_name, runner.host, ARMV7_TARGET)
    saved_a = arm_bin.with_name(f"{arm_bin.name}.armv7a")
    if build_a.status == "FAIL":
        skip_after_failed_dependency(runner, f"run: spectest armv7a ({profile_name})", build_a)
    elif arm_bin.exists() or runner.dry_run:
        # In dry-run we skip the copy (the artifact wasn't really built);
        # run_armv7_binary's runner.run still short-circuits to OK below.
        if arm_bin.exists():
            shutil.copy2(arm_bin, saved_a)
        run_armv7_binary(
            runner,
            f"run: spectest armv7a ({profile_name})",
            saved_a,
            ["--backend", "native", *extra_args],
            env,
            f"spectest-armv7a-{profile_name}",
        )
    else:
        runner.fail(f"run: spectest armv7a ({profile_name})", f"expected binary not found: {arm_bin}")

    build_m = cargo_build(
        runner,
        f"cargo build: spectest armv7m ({profile_name})",
        "sf-nano-spectest",
        profile_name=profile_name,
        target=ARMV7_TARGET,
        features="jit,thumb2-test",
        log_name=f"spectest-build-armv7m-{profile_name}",
    )
    saved_m = arm_bin.with_name(f"{arm_bin.name}.armv7m")
    if build_m.status == "FAIL":
        skip_after_failed_dependency(runner, f"run: spectest armv7m ({profile_name})", build_m)
    elif arm_bin.exists() or runner.dry_run:
        if arm_bin.exists():
            shutil.copy2(arm_bin, saved_m)
        run_armv7_binary(
            runner,
            f"run: spectest armv7m ({profile_name})",
            saved_m,
            ["--backend", "native", *extra_args],
            env,
            f"spectest-armv7m-{profile_name}",
        )
    else:
        runner.fail(f"run: spectest armv7m ({profile_name})", f"expected binary not found: {arm_bin}")


def run_armv7_binary(
    runner: CheckRunner,
    name: str,
    binary: Path,
    args: Sequence[str],
    env: Dict[str, str],
    log_name: str,
) -> None:
    if not runner.dry_run and not binary.exists():
        runner.fail(name, f"expected binary not found: {binary}")
        return
    if runner.host.kind == "macos":
        argv: List[object] = [
            "colima",
            "ssh",
            "--",
            "env",
            f"TESTSUITE_DIR={env['TESTSUITE_DIR']}",
            "qemu-arm-static",
            "-cpu",
            "cortex-a15",
            binary,
            *args,
        ]
        runner.run(name, argv, log_name=log_name, ignore_warnings=True)
        return

    argv = ["qemu-arm-static", "-cpu", "cortex-a15", binary, *args]
    runner.run(name, argv, env=env, log_name=log_name, ignore_warnings=True)


def run_riscv64_binary(
    runner: CheckRunner,
    name: str,
    binary: Path,
    args: Sequence[str],
    env: Dict[str, str],
    log_name: str,
) -> Optional[StepResult]:
    if not runner.dry_run and not binary.exists():
        runner.fail(name, f"expected binary not found: {binary}")
        return None
    if runner.host.kind == "macos":
        argv: List[object] = [
            "colima",
            "ssh",
            "--",
            "env",
            f"TESTSUITE_DIR={env['TESTSUITE_DIR']}",
            "qemu-riscv64-static",
            "-cpu",
            "rv64",
            binary,
            *args,
        ]
        return runner.run(name, argv, log_name=log_name, ignore_warnings=True)

    argv = ["qemu-riscv64-static", "-cpu", "rv64", binary, *args]
    return runner.run(name, argv, env=env, log_name=log_name, ignore_warnings=True)


def run_riscv32_binary(
    runner: CheckRunner,
    name: str,
    binary: Path,
    args: Sequence[str],
    env: Dict[str, str],
    log_name: str,
) -> Optional[StepResult]:
    if not runner.dry_run and not binary.exists():
        runner.fail(name, f"expected binary not found: {binary}")
        return None
    if runner.host.kind == "macos":
        argv: List[object] = [
            "colima",
            "ssh",
            "--",
            "env",
            f"TESTSUITE_DIR={env['TESTSUITE_DIR']}",
            "qemu-riscv32-static",
            "-cpu",
            "rv32",
            binary,
            *args,
        ]
        return runner.run(name, argv, log_name=log_name, ignore_warnings=True)

    argv = ["qemu-riscv32-static", "-cpu", "rv32", binary, *args]
    return runner.run(name, argv, env=env, log_name=log_name, ignore_warnings=True)


def run_wasitest_suite(runner: CheckRunner, *, profiles: Sequence[str], extra_args: Sequence[str]) -> None:
    for profile_name in profiles:
        run_wasitest_host_profile(runner, profile_name, extra_args)

    if "release" in profiles:
        if os.environ.get("SF_CHECK_SKIP_CROSS_RUNTIME"):
            runner.skip(
                "run: wasitest cross-arch (release)",
                "SF_CHECK_SKIP_CROSS_RUNTIME=1: x64/armv7/riscv64/riscv32 WASI runs deferred to the dedicated cross-arch CI job.",
            )
        else:
            run_wasitest_release_cross_targets(runner, extra_args)


def run_wasitest_host_profile(runner: CheckRunner, profile_name: str, extra_args: Sequence[str]) -> None:
    cli_build = cargo_build(
        runner,
        f"cargo build: wasitest cli native ({profile_name})",
        "sf-nano-cli",
        profile_name=profile_name,
        target=None,
        features="jit",
        log_name=f"wasitest-build-cli-native-{profile_name}",
    )
    harness_build = cargo_build_package(
        runner,
        f"cargo build: wasitest harness native ({profile_name})",
        "sf-nano-wasitest",
        profile_name=profile_name,
        target=None,
        log_name=f"wasitest-build-harness-native-{profile_name}",
    )
    cli_bin = binary_path("sf-nano-cli", profile_name, runner.host)
    wasitest_bin = binary_path("sf-nano-wasitest", profile_name, runner.host)
    if cli_build.status == "FAIL":
        skip_after_failed_dependency(runner, f"run: wasitest native ({profile_name})", cli_build)
    elif harness_build.status == "FAIL":
        skip_after_failed_dependency(runner, f"run: wasitest native ({profile_name})", harness_build)
    else:
        run_binary_if_present(
            runner,
            f"run: wasitest native ({profile_name})",
            [wasitest_bin, "--cli-path", cli_bin, "--backend", "native", *extra_args],
            log_name=f"wasitest-native-{profile_name}",
        )

    for backend in ("emu64", "emu32"):
        emu_build = cargo_build(
            runner,
            f"cargo build: wasitest cli {backend} ({profile_name})",
            "sf-nano-cli",
            profile_name=profile_name,
            target=None,
            features=f"jit,backend-{backend}",
            log_name=f"wasitest-build-cli-{backend}-{profile_name}",
        )
        if emu_build.status == "FAIL":
            skip_after_failed_dependency(runner, f"run: wasitest {backend} ({profile_name})", emu_build)
        elif harness_build.status == "FAIL":
            skip_after_failed_dependency(runner, f"run: wasitest {backend} ({profile_name})", harness_build)
        else:
            run_binary_if_present(
                runner,
                f"run: wasitest {backend} ({profile_name})",
                [wasitest_bin, "--cli-path", cli_bin, *extra_args],
                log_name=f"wasitest-{backend}-{profile_name}",
                ignore_warnings=True,
            )


def run_wasitest_release_cross_targets(runner: CheckRunner, extra_args: Sequence[str]) -> None:
    target = x64_target_for_host(runner.host)
    if target and is_runnable_x64_host(runner.host) and ensure_rust_target(runner, target):
        cli_build = cargo_build(
            runner,
            "cargo build: wasitest cli x64 (release)",
            "sf-nano-cli",
            profile_name="release",
            target=target,
            features="jit",
            log_name="wasitest-build-cli-x64-release",
        )
        harness_build = cargo_build_package(
            runner,
            "cargo build: wasitest harness x64 (release)",
            "sf-nano-wasitest",
            profile_name="release",
            target=target,
            log_name="wasitest-build-harness-x64-release",
        )
        cli_bin = binary_path("sf-nano-cli", "release", runner.host, target)
        wasitest_bin = binary_path("sf-nano-wasitest", "release", runner.host, target)
        if cli_build.status == "FAIL":
            skip_after_failed_dependency(runner, "run: wasitest x64 (release)", cli_build)
        elif harness_build.status == "FAIL":
            skip_after_failed_dependency(runner, "run: wasitest x64 (release)", harness_build)
        else:
            if runner.host.kind == "macos":
                argv: List[object] = ["arch", "-x86_64", wasitest_bin, "--cli-path", cli_bin, "--backend", "native", *extra_args]
            else:
                argv = [wasitest_bin, "--cli-path", cli_bin, "--backend", "native", *extra_args]
            run_binary_if_present(runner, "run: wasitest x64 (release)", argv, log_name="wasitest-x64-release")
    else:
        runner.skip("run: wasitest x64 (release)", "x64 runtime checks require runnable x64 host support.")

    qemu_harness_build = cargo_build_package(
        runner,
        "cargo build: wasitest harness native (release for qemu)",
        "sf-nano-wasitest",
        profile_name="release",
        target=None,
        log_name="wasitest-build-harness-native-release-for-qemu",
    )
    native_wasitest = binary_path("sf-nano-wasitest", "release", runner.host)

    run_riscv64_wasitest_release(runner, extra_args, qemu_harness_build, native_wasitest)
    run_riscv32_wasitest_release(runner, extra_args, qemu_harness_build, native_wasitest)

    if not ensure_rust_target(runner, ARMV7_TARGET):
        return
    if not ensure_armv7_runner(runner):
        return

    for label, features in (("armv7a", "jit"), ("armv7m", "jit,thumb2-test")):
        cli_build = cargo_build(
            runner,
            f"cargo build: wasitest cli {label} (release)",
            "sf-nano-cli",
            profile_name="release",
            target=ARMV7_TARGET,
            features=features,
            log_name=f"wasitest-build-cli-{label}-release",
        )
        cli_bin = binary_path("sf-nano-cli", "release", runner.host, ARMV7_TARGET)
        saved = cli_bin.with_name(f"{cli_bin.name}.{label}")
        if qemu_harness_build.status == "FAIL":
            skip_after_failed_dependency(runner, f"run: wasitest {label} (release)", qemu_harness_build)
            continue
        if cli_build.status == "FAIL":
            skip_after_failed_dependency(runner, f"run: wasitest {label} (release)", cli_build)
            continue
        if cli_bin.exists():
            shutil.copy2(cli_bin, saved)
        elif not runner.dry_run:
            runner.fail(f"run: wasitest {label} (release)", f"expected binary not found: {cli_bin}")
            continue
        wrapper = make_armv7_wasi_wrapper(runner, label, saved)
        run_binary_if_present(
            runner,
            f"run: wasitest {label} (release)",
            [native_wasitest, "--cli-path", wrapper, "--backend", "native", *extra_args],
            env={"TMPDIR": str(runner.tmp_dir)},
            log_name=f"wasitest-{label}-release",
            ignore_warnings=True,
        )


def run_riscv64_wasitest_release(
    runner: CheckRunner,
    extra_args: Sequence[str],
    harness_build: StepResult,
    native_wasitest: Path,
) -> None:
    if not ensure_rust_target(runner, RISCV64_TARGET):
        return
    if not ensure_riscv64_runner(runner):
        return

    cli_build = cargo_build(
        runner,
        "cargo build: wasitest cli riscv64 (release)",
        "sf-nano-cli",
        profile_name="release",
        target=RISCV64_TARGET,
        features="jit",
        log_name="wasitest-build-cli-riscv64-release",
    )
    cli_bin = binary_path("sf-nano-cli", "release", runner.host, RISCV64_TARGET)
    if harness_build.status == "FAIL":
        skip_after_failed_dependency(runner, "run: wasitest riscv64 (release)", harness_build)
        return
    if cli_build.status == "FAIL":
        skip_after_failed_dependency(runner, "run: wasitest riscv64 (release)", cli_build)
        return

    wrapper = make_riscv64_wasi_wrapper(runner, cli_bin)
    run_binary_if_present(
        runner,
        "run: wasitest riscv64 (release)",
        [native_wasitest, "--cli-path", wrapper, "--backend", "native", *extra_args],
        env={"TMPDIR": str(runner.tmp_dir)},
        log_name="wasitest-riscv64-release",
        ignore_warnings=True,
    )


def run_riscv32_wasitest_release(
    runner: CheckRunner,
    extra_args: Sequence[str],
    harness_build: StepResult,
    native_wasitest: Path,
) -> None:
    if not ensure_rust_target(runner, RISCV32_TARGET):
        return
    if not ensure_riscv32_runner(runner):
        return

    cli_build = cargo_build(
        runner,
        "cargo build: wasitest cli riscv32 (release)",
        "sf-nano-cli",
        profile_name="release",
        target=RISCV32_TARGET,
        features="jit",
        log_name="wasitest-build-cli-riscv32-release",
    )
    cli_bin = binary_path("sf-nano-cli", "release", runner.host, RISCV32_TARGET)
    if harness_build.status == "FAIL":
        skip_after_failed_dependency(runner, "run: wasitest riscv32 (release)", harness_build)
        return
    if cli_build.status == "FAIL":
        skip_after_failed_dependency(runner, "run: wasitest riscv32 (release)", cli_build)
        return

    wrapper = make_riscv32_wasi_wrapper(runner, cli_bin)
    run_binary_if_present(
        runner,
        "run: wasitest riscv32 (release)",
        [
            native_wasitest,
            "--cli-path",
            wrapper,
            "--backend",
            "native",
            "--skip-rv32-qemu-timestamp-tests",
            *extra_args,
        ],
        env={"TMPDIR": str(runner.tmp_dir)},
        log_name="wasitest-riscv32-release",
        ignore_warnings=True,
    )


def make_qemu_wasi_wrapper_script(
    runner: CheckRunner,
    cli_bin: Path,
    *,
    qemu_bin: str,
    qemu_cmd: str,
    cpu: str,
) -> str:
    if runner.host.kind == "macos":
        # Colima's shared /Users mount does not preserve every Linux filesystem
        # semantic the WASI suite checks. In particular, hard-linking a
        # dangling symlink fails there even though the same binary passes on VM
        # /tmp.
        # Copy preopened directories into the VM before invoking qemu so the
        # tests exercise Linux semantics instead of host-share behavior.
        host_home = shlex_quote(os.environ.get("HOME") or str(Path.home()))
        return f"""\
#!/usr/bin/env bash
set -euo pipefail
export HOME={host_home}
env_args=()
while IFS= read -r key; do
    case "$key" in
        _|HOME|PWD|SHLVL) continue ;;
    esac
    env_args+=("$key=${{!key}}")
done < <(compgen -e)

remote_root="/tmp/sf-nano-wasitest-{cpu}-$$-${{RANDOM}}"
remote_args=()
copy_specs=()
preopen_index=0
while (($#)); do
    case "$1" in
        --dir)
            if (($# < 2)); then
                echo "missing value for --dir" >&2
                exit 2
            fi
            dir_arg="$2"
            shift 2
            if [[ "$dir_arg" == *::* ]]; then
                guest="${{dir_arg%%::*}}"
                host_path="${{dir_arg#*::}}"
            else
                guest="$dir_arg"
                host_path="$dir_arg"
            fi
            remote_dir="$remote_root/preopens/$preopen_index"
            preopen_index=$((preopen_index + 1))
            copy_specs+=("$remote_dir"$'\\t'"$host_path")
            remote_args+=(--dir "$guest::$remote_dir")
            ;;
        *)
            remote_args+=("$1")
            shift
            ;;
    esac
done

cleanup() {{
    colima ssh -- rm -rf "$remote_root" >/dev/null 2>&1 || true
}}
trap cleanup EXIT

colima ssh -- sh -c 'set -eu
root="$1"
rm -rf "$root"
mkdir -p "$root"
' sh "$remote_root"

if ((${{#copy_specs[@]}})); then
    for spec in "${{copy_specs[@]}}"; do
        remote_dir="${{spec%%$'\\t'*}}"
        host_path="${{spec#*$'\\t'}}"
        colima ssh -- sh -c 'set -eu
dst="$1"
src="$2"
rm -rf "$dst"
mkdir -p "$dst"
if [ -d "$src" ]; then
    cp -a "$src"/. "$dst"/
elif [ -e "$src" ]; then
    cp -a "$src" "$dst"/
fi
' sh "$remote_dir" "$host_path"
    done
fi

status_file="$remote_root/status"

# `colima ssh` reports any nonzero remote process as host status 1 and writes
# "exit status N" to stderr. WASI tests assert exact exit codes, so record the
# guest status inside the VM and keep the SSH command itself successful.
set +e
colima ssh -- /bin/sh -c '
status_file="$1"
shift
/usr/bin/env -i "$@"
status=$?
printf "%s\\n" "$status" > "$status_file"
exit 0
' sh "$status_file" "${{env_args[@]}}" {qemu_bin} -cpu {cpu} {shlex_quote(str(cli_bin))} "${{remote_args[@]}}"
ssh_status=$?
set -e

if ((ssh_status != 0)); then
    exit "$ssh_status"
fi

status="$(colima ssh -- cat "$status_file" | tr -d '\\r\\n')"
case "$status" in
    ''|*[!0-9]*)
        echo "invalid remote qemu status: $status" >&2
        exit 1
        ;;
esac
exit "$status"
"""
    return f"""\
#!/usr/bin/env bash
set -euo pipefail
exec {qemu_cmd} -cpu {cpu} {shlex_quote(str(cli_bin))} "$@"
"""


def make_riscv_wasi_wrapper_script(runner: CheckRunner, cli_bin: Path, cpu: str) -> str:
    return make_qemu_wasi_wrapper_script(
        runner,
        cli_bin,
        qemu_bin=f"/usr/bin/qemu-riscv{cpu[-2:]}-static",
        qemu_cmd=f"qemu-riscv{cpu[-2:]}-static",
        cpu=cpu,
    )


def make_riscv64_wasi_wrapper(runner: CheckRunner, cli_bin: Path) -> Path:
    wrapper = runner.tmp_dir / "wasitest-riscv64-qemu-wrapper.sh"
    script = make_riscv_wasi_wrapper_script(runner, cli_bin, "rv64")
    wrapper.write_text(script, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def make_riscv32_wasi_wrapper(runner: CheckRunner, cli_bin: Path) -> Path:
    wrapper = runner.tmp_dir / "wasitest-riscv32-qemu-wrapper.sh"
    script = make_riscv_wasi_wrapper_script(runner, cli_bin, "rv32")
    wrapper.write_text(script, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def make_armv7_wasi_wrapper(runner: CheckRunner, label: str, cli_bin: Path) -> Path:
    wrapper = runner.tmp_dir / f"wasitest-{label}-qemu-wrapper.sh"
    script = make_qemu_wasi_wrapper_script(
        runner,
        cli_bin,
        qemu_bin="/usr/bin/qemu-arm-static",
        qemu_cmd="qemu-arm-static",
        cpu="cortex-a15",
    )
    wrapper.write_text(script, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def shlex_quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


def run_windows_native_spectest(
    runner: CheckRunner,
    *,
    profiles: Sequence[str],
    extra_args: Sequence[str],
) -> None:
    if not ensure_testsuite(runner):
        return

    env = {"TESTSUITE_DIR": str(TESTSUITE_DIR)}
    for profile_name in profiles:
        build = cargo_build(
            runner,
            f"windows x64 build spectest native ({profile_name})",
            "sf-nano-spectest",
            profile_name=profile_name,
            target=None,
            features="jit",
            log_name=f"windows-spectest-build-native-{profile_name}",
        )
        host_bin = binary_path("sf-nano-spectest", profile_name, runner.host)
        if build.status == "FAIL":
            skip_after_failed_dependency(runner, f"windows x64 spectest native ({profile_name})", build)
        else:
            run_binary_if_present(
                runner,
                f"windows x64 spectest native ({profile_name})",
                [host_bin, "--backend", "native", *extra_args],
                env=env,
                log_name=f"windows-spectest-native-{profile_name}",
            )


def run_windows_native_full(
    runner: CheckRunner,
    *,
    debug_only: bool,
    release_only: bool,
    extra_args: Sequence[str],
) -> None:
    profiles = selected_profiles(debug_only, release_only)
    if "debug" in profiles:
        runner.run(
            "windows x64 workspace check (debug)",
            ["cargo", "check"],
            log_name="windows-cargo-check-debug",
        )
    if "release" in profiles:
        runner.run(
            "windows x64 release build",
            ["cargo", "build", "--release"],
            log_name="windows-cargo-build-release",
        )

    for profile_name in profiles:
        run_workspace_tests(runner, profile_name)

    run_target_matrix(
        runner,
        include_all=False,
        include_bare_smoke=False,
        only_labels=WINDOWS_TARGET_MATRIX_LABELS,
        unsupported_labels=UNSUPPORTED_ON_WINDOWS_TARGET_LABELS,
    )
    run_bare_smoke_checks(runner)
    run_windows_native_spectest(runner, profiles=profiles, extra_args=extra_args)

    for profile_name in profiles:
        run_wasitest_host_profile(runner, profile_name, [])


def run_windows_native_phase(args: argparse.Namespace, host: Host, *, print_report: bool = True) -> int:
    runner = CheckRunner(
        host,
        strict=args.strict,
        install_targets=args.install_targets,
        max_diagnostics=max(1, args.max_diagnostics),
        dry_run=args.dry_run,
        phase="windows",
    )

    print(f"Silverfir check: phase=windows host={host.kind}/{host.machine}")
    if args.dry_run:
        print("dry-run: listing steps only; no subprocesses will execute")
    print(f"log dir: {rel(runner.log_dir)}")

    run_windows_native_full(
        runner,
        debug_only=args.debug_only,
        release_only=args.release_only,
        extra_args=args.extra_args,
    )

    return runner.final_report(print_report=print_report)


def print_overall_summary(windows_rc: int, wsl_rc: int) -> None:
    overall_rc = 1 if windows_rc or wsl_rc else 0
    windows_status = "PASSED" if windows_rc == 0 else "FAILED"
    wsl_status = "PASSED" if wsl_rc == 0 else "FAILED"
    overall_status = "PASSED" if overall_rc == 0 else "FAILED"

    print()
    print("=== Summary ===")
    print(f"host=windows+WSL windows={windows_status} wsl={wsl_status}")
    print()
    print(overall_status)


def skipped_target_labels_from_env() -> set[str]:
    return {label for label in os.environ.get("SF_CHECK_SKIP_TARGET_LABELS", "").split(",") if label}


def run_full(
    runner: CheckRunner,
    *,
    debug_only: bool,
    release_only: bool,
    include_all_targets: bool,
    extra_args: Sequence[str],
    skip_target_labels: Optional[set[str]] = None,
) -> None:
    profiles = selected_profiles(debug_only, release_only)
    for profile_name in profiles:
        run_workspace_tests(runner, profile_name)
    run_feature_checks(runner, profiles=profiles)
    run_target_matrix(
        runner,
        include_all=include_all_targets,
        include_bare_smoke=not os.environ.get("SF_CHECK_SKIP_BARE_SMOKE"),
        exclude_labels=skip_target_labels,
    )
    run_spectest_suite(runner, profiles=profiles, extra_args=extra_args)
    run_wasitest_suite(runner, profiles=profiles, extra_args=[])


def selected_profiles(debug_only: bool, release_only: bool) -> List[str]:
    if debug_only and release_only:
        raise SystemExit("--debug-only and --release-only are mutually exclusive")
    if debug_only:
        return ["debug"]
    if release_only:
        return ["release"]
    return ["debug", "release"]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parse_argv = list(argv)
    extra_args: List[str] = []
    if "--" in parse_argv:
        split_at = parse_argv.index("--")
        extra_args = parse_argv[split_at + 1 :]
        parse_argv = parse_argv[:split_at]

    parser = argparse.ArgumentParser(
        description="Run Silverfir Nano validation checks with concise output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python3 scripts/check.py
              python3 scripts/check.py --release-only --install-targets
              python3 scripts/check.py -- i32 --log-level info
            """
        ),
    )
    parser.add_argument("--strict", action="store_true", help="make warnings fail where supported")
    parser.add_argument("--install-targets", action="store_true", help="install missing rustup targets")
    parser.add_argument("--max-diagnostics", type=int, default=8, help="diagnostics to print per issue step")
    parser.add_argument("--debug-only", action="store_true", help="run debug profile checks only")
    parser.add_argument("--release-only", action="store_true", help="run release profile checks only")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use this to check what will be tested on this platform and what is skipped "
        "(lists every scheduled step without running any subprocess)",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--all-targets", action="store_true", help="include extra target rows")
    target_group.add_argument("--default-targets", action="store_true", help="use default target rows")
    args = parser.parse_args(parse_argv)
    args.extra_args = extra_args
    return args


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    host = detect_host()
    if host.kind == "windows":
        if os.environ.get("SF_CHECK_NO_WSL_REEXEC"):
            return run_windows_native_phase(args, host)
        print(
            "Silverfir check: native Windows detected; running Windows x64 phase first, "
            "then WSL/Linux phase.",
            flush=True,
        )
        windows_rc = run_windows_native_phase(args, host, print_report=False)
        print()
        print("Silverfir check: starting WSL/Linux phase", flush=True)
        wsl_rc = reexec_in_wsl(argv)
        print_overall_summary(windows_rc, wsl_rc)
        return 1 if windows_rc or wsl_rc else 0

    runner = CheckRunner(
        host,
        strict=args.strict,
        install_targets=args.install_targets,
        max_diagnostics=max(1, args.max_diagnostics),
        dry_run=args.dry_run,
    )

    print(f"Silverfir check: host={host.kind}/{host.machine}")
    if args.dry_run:
        print("dry-run: listing steps only; no subprocesses will execute")
    print(f"log dir: {rel(runner.log_dir)}")

    try:
        include_all_targets = not args.default_targets
        run_full(
            runner,
            debug_only=args.debug_only,
            release_only=args.release_only,
            include_all_targets=include_all_targets,
            extra_args=args.extra_args,
            skip_target_labels=skipped_target_labels_from_env(),
        )
    finally:
        cleanup_colima(runner)

    return runner.final_report(print_report=not os.environ.get("SF_CHECK_SUPPRESS_SUMMARY"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
