#!/usr/bin/env python3
"""Reject broad Rust lint suppressions unless they state why they exist.

Every `allow`/`expect` of `warnings`, `dead_code` or `unused*` must carry an
inline `reason = "..."`, which puts the justification in front of the next
person to read the code rather than in a file they will never open. Sites that
cannot carry one inline -- a generated file, an attribute the toolchain will
not accept a reason on -- go in `ci/lint_suppressions.toml` instead.

What this cannot check is whether a human agreed with the reason. Nothing
mechanical can: a plausible sentence is exactly what an agent writing its own
exception would produce. So the audit guarantees only that every suppression is
*stated and greppable*, and the rule that an agent never writes one on its own
judgment lives in CLAUDE.md and AGENTS.md, enforced by review.

Suppressing `warnings` wholesale is refused outright, reason or not.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on interpreter version
    raise SystemExit(
        "ci.lint_policy requires Python 3.11+ (the `tomllib` module); "
        "run it with a newer python3"
    )
from pathlib import Path
from typing import Iterable


CONTROLLED_LINT_RE = re.compile(r"\b(?:warnings|dead_code|unused(?:_[A-Za-z0-9_]+)?)\b")
ATTRIBUTE_RE = re.compile(r"#!?\s*\[(?P<body>[^\]]*)\]", re.MULTILINE | re.DOTALL)
SUPPRESSION_RE = re.compile(
    r"\b(?P<kind>allow|expect)\s*\((?P<body>[^)]*)\)",
    re.MULTILINE | re.DOTALL,
)
REASON_RE = re.compile(r"\breason\s*=")
# `.claude` holds agent worktrees (full repo copies whose sources are not part
# of this checkout) and session state; scanning it makes local runs report
# other checkouts' findings. `.cargo`, `.github`, and other dot-directories
# stay scanned -- the flag scan depends on them.
SKIP_DIRS = {".git", "target", ".claude"}
FORBIDDEN_FLAG_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:-A(?:\s*|=|[\"']\s*,\s*[\"'])|"
    r"--allow(?:\s+|=|[\"']\s*,\s*[\"']))"
    r"(?:warnings|dead_code|unused(?:_[A-Za-z0-9_]+)?)\b"
    r"|--cap-lints(?:=|\s+|[\"']\s*,\s*[\"'])(?:allow|warn)\b"
)
FLAG_SCAN_SUFFIXES = {".py", ".ps1", ".rs", ".sh", ".toml", ".yaml", ".yml"}
FLAG_SCAN_EXCLUDES = {"ci/test_lint_policy.py"}


@dataclasses.dataclass(frozen=True, order=True)
class SuppressionKey:
    path: str
    kind: str
    lints: tuple[str, ...]
    anchor: str


@dataclasses.dataclass(frozen=True)
class Suppression:
    key: SuppressionKey
    line: int
    has_reason: bool


def controlled_lints(body: str) -> tuple[str, ...]:
    lint_list = body.split("reason", 1)[0]
    return tuple(sorted(set(CONTROLLED_LINT_RE.findall(lint_list))))


def occurrence_is_in_line_comment(text: str, start: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start].lstrip()
    return prefix.startswith(("//", "///", "//!"))


def normalize_anchor(line: str) -> str:
    anchor = line.strip()
    # Generated Rust source in build scripts appears inside escaped string
    # literal lines ("...\n\" continuations). Strip the quoting and escape
    # artifacts so the anchor identifies the generated line's content rather
    # than its string encoding.
    anchor = anchor.removeprefix('"')
    while True:
        for suffix in ("\\n", "\\r", "\\t", "\\", '",', '";', '"'):
            if anchor.endswith(suffix):
                anchor = anchor.removesuffix(suffix)
                break
        else:
            break
    return " ".join(anchor.split())[:240]


def anchor_after(text: str, end: int) -> str:
    for line in text[end:].splitlines():
        anchor = normalize_anchor(line)
        if not anchor or anchor in {"]", "];", ']"', '];"'}:
            continue
        if anchor.startswith(("#[", "#![")):
            continue
        return anchor
    return "<eof>"


def scan_file(path: Path, root: Path) -> list[Suppression]:
    text = path.read_text(encoding="utf-8", errors="replace")
    relative = path.relative_to(root).as_posix()
    found: list[Suppression] = []
    for attribute in ATTRIBUTE_RE.finditer(text):
        if occurrence_is_in_line_comment(text, attribute.start()):
            continue
        for match in SUPPRESSION_RE.finditer(attribute.group("body")):
            lints = controlled_lints(match.group("body"))
            if not lints:
                continue
            found.append(
                Suppression(
                    key=SuppressionKey(
                        path=relative,
                        kind=match.group("kind"),
                        lints=lints,
                        anchor=anchor_after(text, attribute.end()),
                    ),
                    line=text.count("\n", 0, attribute.start()) + 1,
                    has_reason=bool(REASON_RE.search(match.group("body"))),
                )
            )
    return found


def repository_files(root: Path) -> Iterable[Path]:
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if name not in SKIP_DIRS)
        base = Path(directory)
        for name in sorted(names):
            yield base / name


def rust_files(root: Path) -> Iterable[Path]:
    return (path for path in repository_files(root) if path.suffix == ".rs")


def check_cargo_lints(root: Path) -> list[str]:
    errors: list[str] = []
    manifests = sorted(
        path for path in repository_files(root) if path.name == "Cargo.toml"
    )
    for path in manifests:
        relative = path.relative_to(root).as_posix()
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        lint_tables: list[tuple[str, dict[str, object]]] = []
        package_rust_lints = data.get("lints", {}).get("rust", {})
        workspace_rust_lints = data.get("workspace", {}).get("lints", {}).get("rust", {})
        if package_rust_lints:
            lint_tables.append(("[lints.rust]", package_rust_lints))
        if workspace_rust_lints:
            lint_tables.append(("[workspace.lints.rust]", workspace_rust_lints))
        for table_name, lint_table in lint_tables:
            for lint, setting in lint_table.items():
                if not CONTROLLED_LINT_RE.fullmatch(lint):
                    continue
                level = setting.get("level") if isinstance(setting, dict) else setting
                if level not in {"deny", "forbid"}:
                    errors.append(
                        f"{relative}: {table_name} must not lower {lint} to {level!r}"
                    )

    return errors


def check_lint_flags(root: Path) -> list[str]:
    errors: list[str] = []
    for path in repository_files(root):
        if path.suffix not in FLAG_SCAN_SUFFIXES:
            continue
        relative = path.relative_to(root)
        relative_posix = relative.as_posix()
        if (
            any(part in SKIP_DIRS for part in relative.parts)
            or relative_posix in FLAG_SCAN_EXCLUDES
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in FORBIDDEN_FLAG_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{relative_posix}:{line}: forbidden compiler lint override "
                f"`{match.group(0).strip()}`"
            )
    return errors


def load_approved(path: Path) -> tuple[collections.Counter[SuppressionKey], list[str]]:
    errors: list[str] = []
    if not path.exists():
        return collections.Counter(), [f"missing suppression manifest: {path}"]
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        errors.append(f"{path}: version must be 1")
    approved: collections.Counter[SuppressionKey] = collections.Counter()
    for index, entry in enumerate(data.get("suppression", []), start=1):
        reason = str(entry.get("reason", "")).strip()
        if not reason:
            errors.append(f"{path}: suppression #{index} has no review reason")
        key = SuppressionKey(
            path=str(entry.get("path", "")),
            kind=str(entry.get("kind", "")),
            lints=tuple(sorted(str(lint) for lint in entry.get("lints", []))),
            anchor=str(entry.get("anchor", "")),
        )
        if not all((key.path, key.kind, key.lints, key.anchor)):
            errors.append(f"{path}: suppression #{index} is incomplete")
        approved[key] += 1
    return approved, errors


def check(root: Path, manifest: Path) -> list[str]:
    approved, errors = load_approved(manifest)
    errors.extend(check_cargo_lints(root))
    errors.extend(check_lint_flags(root))
    # `needs_manifest` holds the suppressions that did not state a reason
    # inline and therefore fall back to the manifest. `seen` holds every
    # suppression, so a manifest entry whose site is gone reads as stale even
    # when the surviving sites carry inline reasons.
    needs_manifest: collections.Counter[SuppressionKey] = collections.Counter()
    seen: collections.Counter[SuppressionKey] = collections.Counter()
    locations: dict[SuppressionKey, list[int]] = collections.defaultdict(list)

    for path in rust_files(root):
        for suppression in scan_file(path, root):
            key = suppression.key
            locations[key].append(suppression.line)
            seen[key] += 1
            where = f"{key.path}:{suppression.line}"
            if "warnings" in key.lints:
                errors.append(f"{where}: suppressing `warnings` is never permitted")
                continue
            # An inline reason states the exception where the next reader will
            # see it, so it stands on its own. What this does NOT check is that
            # a human agreed with the reason -- nothing mechanical can, since a
            # plausible sentence is exactly what an agent writing its own
            # exception would produce. That rule lives in CLAUDE.md and
            # AGENTS.md; this audit only guarantees every exception is stated
            # and greppable.
            if not suppression.has_reason:
                needs_manifest[key] += 1

    for key, count in sorted((needs_manifest - approved).items()):
        lines = ", ".join(str(line) for line in locations[key])
        errors.append(
            f"{key.path}:{lines}: {key.kind}({', '.join(key.lints)}) before "
            f"`{key.anchor}` needs an inline `reason = \"...\"`, or a manifest "
            f"entry if the attribute position cannot carry one"
        )
    for key, count in sorted((approved - seen).items()):
        errors.append(
            f"{manifest}: stale approval for {key.path} "
            f"{key.kind}({', '.join(key.lints)}) before `{key.anchor}`"
        )
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_root / "ci" / "lint_suppressions.toml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))
    root = args.root.resolve()
    manifest = args.manifest.resolve()
    errors = check(root, manifest)
    if errors:
        print("Rust lint suppression policy failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Rust lint suppression policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
