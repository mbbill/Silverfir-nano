#!/usr/bin/env python3
"""Prepare exact-revision API evidence for the protected review job.

The baseline is extracted from the PR's base revision, not from snapshots
modified in the PR. This program never grants approval. GitHub's protected
public-api-review environment owns that decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from ci.public_api import SNAPSHOTS, api_diff

POLICY_FILES = (
    ".github/workflows/public-api.yml",
    "ci/public_api.py",
    "ci/public_api_review.py",
    "ci/test_public_api.py",
    "ci/test_public_api_review.py",
    "docs/PUBLIC_API_POLICY.md",
)


def revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"expected a full Git revision: {value!r}")
    return value


def changed_files(root: Path, base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z",
         revision(base), revision(head), "--"],
        cwd=root, check=True, stdout=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8").rstrip("\0").split("\0") if result.stdout else []


def bootstrap_required(root: Path, base_sha: str) -> bool:
    result = subprocess.run(
        ["git", "ls-tree", "--name-only", revision(base_sha), "--", "ci/public_api.py"],
        cwd=root, check=True, stdout=subprocess.PIPE,
    )
    return not result.stdout


def verify_checkout(root: Path, head_sha: str) -> None:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if actual != revision(head_sha):
        raise ValueError("API evidence does not match the checked-out PR head")
    subprocess.run(["git", "diff", "--exit-code", "--quiet", head_sha, "--"],
                   cwd=root, check=True)


def compare(base: Path, head: Path) -> dict[str, str]:
    # Missing, partial or undecodable captures are errors, never "unchanged".
    differences = {}
    for name in SNAPSHOTS:
        delta = api_diff(
            (base / name).read_text(encoding="utf-8"),
            (head / name).read_text(encoding="utf-8"), name,
        )
        if delta:
            differences[name] = delta
    return differences


def evidence_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in SNAPSHOTS:
        data = (directory / name).read_bytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def environment_problems(document: dict) -> list[str]:
    problems = []
    if document.get("name") != "public-api-review":
        problems.append("wrong or missing public-api-review environment")
    rules = [rule for rule in document.get("protection_rules", [])
             if rule.get("type") == "required_reviewers"]
    reviewers = [entry for rule in rules for entry in rule.get("reviewers", [])]
    # The project owner is also the usual PR author. Deployment approval
    # deliberately permits that human to review their own account's changes.
    if len(reviewers) != 1 or reviewers[0].get("type") != "User" or \
            reviewers[0].get("reviewer", {}).get("login") != "mbbill":
        # GitHub requires only one listed reviewer, not all of them. An extra
        # bot/team could otherwise approve without the designated human.
        problems.append("mbbill must be the sole required environment reviewer")
    if any(rule.get("prevent_self_review") is not False for rule in rules):
        problems.append("self review must be enabled for the repository owner")
    if document.get("can_admins_bypass") is not False:
        problems.append("environment admin bypass must be disabled")
    return problems


def prepare(base: Path, head: Path, output: Path, base_sha: str, head_sha: str,
            files: list[str], *, bootstrap: bool = False) -> dict:
    differences = {
        name: api_diff("", (head / name).read_text(encoding="utf-8"), name)
        for name in SNAPSHOTS
    } if bootstrap else compare(base, head)
    policy = sorted(set(files).intersection(POLICY_FILES))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "base": revision(base_sha), "head": revision(head_sha),
        "bootstrap": bootstrap,
        "base_digest": None if bootstrap else evidence_digest(base),
        "head_digest": evidence_digest(head),
        "changed_surfaces": sorted(differences), "changed_policy": policy,
        "review_required": bool(bootstrap or differences or policy),
    }
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "public-api.diff").write_text("".join(differences.values()), encoding="utf-8")
    summary = (
        "## Public API review\n\n"
        f"Base: `{base_sha}`\n\nHead: `{head_sha}`\n\n"
        f"Candidate SHA-256: `{report['head_digest']}`\n\n"
    )
    if report["review_required"]:
        if bootstrap:
            summary += (
                "The base revision predates API review tooling. This is the "
                "initial baseline: review the **entire candidate API**. No "
                "historical baseline is claimed as validated or accepted.\n\n"
            )
        summary += (
            "**Human API review required.** Inspect `public-api.diff` and the "
            "captured surfaces in this run's artifact before approving the "
            "`public-api-review` environment. Snapshot edits and PR labels "
            "do not approve this change.\n\n"
            f"Changed surfaces: {', '.join(report['changed_surfaces']) or 'none'}\n\n"
            f"Review policy changes: {', '.join(policy) or 'none'}\n"
        )
    else:
        summary += "Public signatures, trait contracts and Cargo features are unchanged.\n"
    (output / "summary.md").write_text(summary, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    review = commands.add_parser("prepare")
    review.add_argument("--base", type=Path, required=True)
    review.add_argument("--head", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--base-sha", required=True)
    review.add_argument("--head-sha", required=True)
    review.add_argument("--root", type=Path, default=Path.cwd())
    environment = commands.add_parser("environment")
    environment.add_argument("document", type=Path)
    plan = commands.add_parser("plan")
    plan.add_argument("--base-sha", required=True)
    plan.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        if args.mode == "environment":
            problems = environment_problems(json.loads(args.document.read_text(encoding="utf-8")))
            for problem in problems:
                print(problem, file=sys.stderr)
            return int(bool(problems))
        if args.mode == "plan":
            bootstrap = bootstrap_required(args.root, args.base_sha)
            if os.environ.get("GITHUB_OUTPUT"):
                with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
                    stream.write(f"bootstrap={str(bootstrap).lower()}\n")
            print(f"Initial baseline review required: {bootstrap}")
            return 0
        verify_checkout(args.root, args.head_sha)
        report = prepare(
            args.base, args.head, args.output, args.base_sha, args.head_sha,
            changed_files(args.root, args.base_sha, args.head_sha),
            bootstrap=bootstrap_required(args.root, args.base_sha),
        )
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
                stream.write(f"review_required={str(report['review_required']).lower()}\n")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write((args.output / "summary.md").read_text(encoding="utf-8"))
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
