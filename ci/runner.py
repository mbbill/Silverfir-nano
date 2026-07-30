"""Small, warning-aware command runner shared by CI gates."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "target"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DIAGNOSTIC_RE = re.compile(
    r"^(?P<header>error(?:\[[^\]]+\])?|warning(?:\[[^\]]+\])?):\s*(?P<message>.*)$"
)
LOCATION_RE = re.compile(r"^\s+-->\s+(.+)$")
GENERATED_WARNINGS_RE = re.compile(r"generated\s+(\d+)\s+warnings?")
APPLICATION_ERROR_RE = re.compile(
    r"(\bERROR\b.*\b(?:FAIL|ERROR)\b|^FAIL\b|Some tests failed|Failed:\s+[1-9])",
    re.IGNORECASE,
)
RELEVANT_ERROR_RE = re.compile(
    r"(error(?:\[|:)|fatal|failed|failure|panicked|not found|missing|FAIL)",
    re.IGNORECASE,
)
# Linker chatter that `-Z build-std` provokes for a tier-3 target and that no
# change to this repository can silence. `riscv32gc-unknown-linux-musl` is not
# a rustup target -- std is compiled from source into the target directory --
# so the linker probes a `rustlib` path that legitimately does not exist. These
# stay visible as diagnostics; they simply do not fail the gate, because the
# alternative is a job that is permanently red and therefore tells nobody
# anything when rv32 actually breaks. Every other warning, linker or compiler,
# still fails.
BENIGN_LINKER_WARNING_RE = re.compile(
    r"^warning:\s*linker stderr:\s*(?:"
    r"ignoring deprecated linker optimization setting"
    r"|unable to open library directory '[^']*[/\\]rustlib[/\\][^']*': FileNotFound"
    r")"
)


@dataclass(frozen=True)
class Diagnostic:
    kind: str
    message: str
    location: str | None = None

    def render(self) -> str:
        prefix = f"{self.location}: " if self.location else ""
        return f"{prefix}{self.message}"


@dataclass
class Result:
    name: str
    status: str
    argv: tuple[str, ...]
    log: Path | None
    returncode: int = 0
    warnings: int = 0
    errors: int = 0
    seconds: float = 0.0
    diagnostics: tuple[Diagnostic, ...] = ()
    note: str | None = None

    @property
    def produced_output(self) -> bool:
        """A warning is a gate failure but does not invalidate build output."""

        return self.returncode == 0


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-").lower()


def display_command(argv: Sequence[object]) -> str:
    return shlex.join(str(part) for part in argv)


def parse_log(path: Path, returncode: int) -> tuple[int, int, tuple[Diagnostic, ...]]:
    try:
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        raw_lines = []
    lines = [ANSI_RE.sub("", line) for line in raw_lines]

    errors = 0
    warnings = 0
    benign = 0
    diagnostics: list[Diagnostic] = []
    seen: set[Diagnostic] = set()

    def add(diagnostic: Diagnostic) -> None:
        if diagnostic not in seen:
            seen.add(diagnostic)
            diagnostics.append(diagnostic)

    for index, line in enumerate(lines):
        match = DIAGNOSTIC_RE.match(line)
        if not match:
            continue
        header = match.group("header")
        kind = "error" if header.startswith("error") else "warning"
        if kind == "error":
            errors += 1
        elif BENIGN_LINKER_WARNING_RE.match(line.strip()):
            benign += 1
        elif not GENERATED_WARNINGS_RE.search(line):
            warnings += 1

        location = None
        for following in lines[index + 1 : index + 8]:
            location_match = LOCATION_RE.match(following)
            if location_match:
                location = location_match.group(1).strip()
                break
        add(Diagnostic(kind, f"{header}: {match.group('message').strip()}", location))

    for line in lines:
        stripped = line.strip()
        if APPLICATION_ERROR_RE.search(stripped):
            errors += 1
            add(Diagnostic("error", stripped))

    generated_count = max(
        (int(match.group(1)) for line in lines if (match := GENERATED_WARNINGS_RE.search(line))),
        default=0,
    )
    # Cargo's per-crate "generated N warnings" tally counts the benign linker
    # lines too, so discount them here as well or they come back through the
    # aggregate.
    warnings = max(warnings, max(0, generated_count - benign))

    if returncode and errors == 0:
        errors = 1
        relevant = [line.strip() for line in lines if RELEVANT_ERROR_RE.search(line)]
        tail = [line.strip() for line in lines[-20:] if line.strip()]
        for line in (relevant or tail)[-8:]:
            add(Diagnostic("error", line))

    return errors, warnings, tuple(diagnostics)


class Runner:
    def __init__(self, label: str) -> None:
        self.label = label
        self.log_dir = TARGET / "ci-logs" / slug(label)
        self.tmp_dir = self.log_dir / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[Result] = []

    def run(
        self,
        name: str,
        argv: Sequence[object],
        *,
        cwd: Path = ROOT,
        env: Mapping[str, object] | None = None,
        log_name: str | None = None,
    ) -> Result:
        command = tuple(str(part) for part in argv)
        log = self.log_dir / f"{slug(log_name or name)}.log"
        process_env = os.environ.copy()
        process_env["CARGO_TERM_COLOR"] = "never"
        if env:
            process_env.update({key: str(value) for key, value in env.items()})

        started = time.monotonic()
        try:
            with log.open("w", encoding="utf-8", errors="replace") as stream:
                stream.write(f"$ {display_command(command)}\n")
                stream.write(f"# cwd: {cwd}\n\n")
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    env=process_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            returncode = completed.returncode
        except OSError as error:
            log.write_text(
                f"$ {display_command(command)}\n# cwd: {cwd}\n\n{error}\n",
                encoding="utf-8",
            )
            returncode = 127

        errors, warnings, diagnostics = parse_log(log, returncode)
        status = "FAIL" if returncode else "WARN" if warnings else "OK"
        result = Result(
            name=name,
            status=status,
            argv=command,
            log=log,
            returncode=returncode,
            warnings=warnings,
            errors=errors,
            seconds=time.monotonic() - started,
            diagnostics=diagnostics,
        )
        self.results.append(result)
        detail = ""
        if warnings:
            detail += f" warnings={warnings}"
        if errors:
            detail += f" errors={errors}"
        print(f"[{status:<4}] {name} ({result.seconds:.1f}s){detail}", flush=True)
        return result

    def skip(self, name: str, reason: str) -> Result:
        result = Result(
            name=name,
            status="SKIP",
            argv=(),
            log=None,
            note=reason,
        )
        self.results.append(result)
        print(f"[SKIP] {name}: {reason}", flush=True)
        return result

    def diagnostics_by_issue(self) -> dict[Diagnostic, list[str]]:
        grouped: dict[Diagnostic, list[str]] = defaultdict(list)
        for result in self.results:
            if result.status not in {"WARN", "FAIL"}:
                continue
            for diagnostic in result.diagnostics:
                grouped[diagnostic].append(result.name)
        return dict(grouped)

    def finish(self) -> int:
        counts = {
            status: sum(result.status == status for result in self.results)
            for status in ("OK", "WARN", "FAIL", "SKIP")
        }
        issues = self.diagnostics_by_issue()

        print()
        print("=== CI correctness summary ===")
        print(
            f"suite={self.label} ok={counts['OK']} warn={counts['WARN']} "
            f"fail={counts['FAIL']} skip={counts['SKIP']}"
        )
        print(f"logs={self.log_dir.relative_to(ROOT)}")
        if issues:
            print()
            print("=== Unique diagnostics ===")
            for diagnostic, steps in sorted(
                issues.items(),
                key=lambda item: (item[0].kind, item[0].location or "", item[0].message),
            ):
                affected = ", ".join(steps[:4])
                if len(steps) > 4:
                    affected += f", +{len(steps) - 4} more"
                print(f"{diagnostic.kind.upper()}: {diagnostic.render()}")
                print(f"  seen in: {affected}")

        failed = bool(counts["WARN"] or counts["FAIL"])
        print()
        print("FAILED" if failed else "PASSED")
        self._write_github_summary(counts, issues, failed)
        return 1 if failed else 0

    def _write_github_summary(
        self,
        counts: Mapping[str, int],
        issues: Mapping[Diagnostic, list[str]],
        failed: bool,
    ) -> None:
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_path:
            return
        lines = [
            f"## Correctness: {self.label}",
            "",
            f"**{'FAILED' if failed else 'PASSED'}**",
            "",
            "| OK | Warnings | Failures | Skipped |",
            "|---:|---:|---:|---:|",
            f"| {counts['OK']} | {counts['WARN']} | {counts['FAIL']} | {counts['SKIP']} |",
        ]
        if issues:
            lines.extend(["", "### Unique diagnostics", ""])
            for diagnostic, steps in sorted(
                issues.items(),
                key=lambda item: (item[0].kind, item[0].location or "", item[0].message),
            ):
                rendered = diagnostic.render().replace("|", "\\|")
                lines.append(
                    f"- **{diagnostic.kind.upper()}** {rendered} "
                    f"_(seen in {len(steps)} step{'s' if len(steps) != 1 else ''})_"
                )
        with Path(summary_path).open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")


def require_tools(runner: Runner, names: Iterable[str]) -> bool:
    import shutil

    missing = [name for name in names if shutil.which(name) is None]
    if not missing:
        return True
    runner.skip("tool prerequisites", f"missing required tools: {', '.join(missing)}")
    runner.results[-1].status = "FAIL"
    runner.results[-1].returncode = 1
    runner.results[-1].errors = 1
    runner.results[-1].diagnostics = (
        Diagnostic("error", f"missing required tools: {', '.join(missing)}"),
    )
    return False
