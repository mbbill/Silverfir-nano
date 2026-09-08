"""Prepare release versions and verify Git identity before publishing (Python 3.11+)."""

import argparse
from pathlib import Path
import re
import subprocess

from ci.release_version import check


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def version_tuple(value, historical=False):
    pattern = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    pattern += r"(?:\.(0|[1-9][0-9]*))?" if historical else r"\.(0|[1-9][0-9]*)"
    match = re.fullmatch(pattern, value)
    if not match:
        raise ValueError(f"Invalid release version: {value!r}")
    return tuple(int(part or 0) for part in match.groups())


def clean(root):
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Working tree must be clean, including untracked files")


def remote_refs(root, remote):
    # Query the server, not possibly stale origin/main or local tags. Failure is fatal.
    return {ref: sha for sha, ref in (line.split() for line in
            git(root, "ls-remote", remote, "refs/heads/main", "refs/tags/*").splitlines())}


def releases(root, refs):
    names = set(git(root, "tag", "--list").splitlines())
    names.update(ref.removeprefix("refs/tags/") for ref in refs
                 if ref.startswith("refs/tags/") and not ref.endswith("^{}"))
    result = {}
    for name in names:
        try:
            result[name] = version_tuple(name.removeprefix("v"), historical=True)
        except ValueError:
            continue  # Non-release tags do not have a version ordering.
    return result


def prepare(root, version, refs):
    clean(root)
    current = check(root)
    target = version_tuple(version)
    if target <= version_tuple(current):
        raise ValueError(f"New version must be greater than current {current}")
    if any(target <= old for old in releases(root, refs).values()):
        raise ValueError("New version must be greater than every existing release tag")
    paths = ["Cargo.toml", "sf-nano-core/Cargo.toml", "Cargo.lock", "sf-nano-core/README.md"]
    original = {path: (root / path).read_bytes() for path in paths}
    updated = {path: data.decode() for path, data in original.items()}
    count = updated[paths[0]].count(f'version = "{current}"')
    updated[paths[0]] = updated[paths[0]].replace(
        f'version = "{current}"', f'version = "{version}"')
    if count != 1:
        raise ValueError("Expected exactly one workspace release version")
    updated[paths[1]] = updated[paths[1]].replace(
        f'version = "{current}", path = "../tools/tracked-alloc"',
        f'version = "{version}", path = "../tools/tracked-alloc"')
    for name in ("sf-nano-core", "sf-nano-tracked-alloc"):
        updated[paths[2]] = updated[paths[2]].replace(
            f'name = "{name}"\nversion = "{current}"',
            f'name = "{name}"\nversion = "{version}"')
    series = ".".join(version.split(".")[:2])
    updated[paths[3]] = re.sub(
        r'(sf-nano-core\s*=\s*(?:\{\s*version\s*=\s*)?")[^"]+(")',
        lambda match: match[1] + series + match[2], updated[paths[3]])
    try:
        for path, text in updated.items():
            (root / path).write_text(text)
        check(root, version)
    except BaseException:
        for path, data in original.items():
            (root / path).write_bytes(data)
        raise
    print(f"Prepared {version}. Review the diff, commit and open a PR; no tag or upload created.")


def verify(root, refs, require_tag=True):
    clean(root)
    version = check(root)
    target = version_tuple(version)
    head = git(root, "rev-parse", "HEAD")
    if refs.get("refs/heads/main") != head:
        raise ValueError("HEAD must equal current remote main; merge the reviewed PR first")
    for name, old in releases(root, refs).items():
        if old > target or (old == target and name != version):
            raise ValueError(f"Version must exceed earlier releases: {name}")
    tag_ref = f"refs/tags/{version}"
    local_tags = git(root, "tag", "--list").splitlines()
    if version in local_tags:
        if git(root, "cat-file", "-t", tag_ref) != "tag":
            raise ValueError("Release tag must be annotated")
        if git(root, "rev-parse", f"{tag_ref}^{{commit}}") != head:
            raise ValueError("Local release tag does not point to HEAD")
    elif require_tag:
        raise ValueError(f"Missing local release tag {version}; run the tag command first")
    if tag_ref in refs:
        if refs.get(tag_ref + "^{}") != head:
            raise ValueError("Remote release tag must be annotated and point to HEAD")
        if version not in local_tags or refs[tag_ref] != git(root, "rev-parse", tag_ref):
            raise ValueError("Local and remote tag objects differ; fetch and inspect them")
    elif require_tag:
        raise ValueError(f"Missing remote tag {version}; push the release tag first")
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="origin")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("version")
    sub.add_parser("tag")
    sub.add_parser("check")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        clean(root)
        refs = remote_refs(root, args.remote)
        if args.command == "prepare":
            prepare(root, args.version, refs)
        elif args.command == "tag":
            version = verify(root, refs, require_tag=False)
            if version in git(root, "tag", "--list").splitlines():
                raise ValueError("Tag already exists; it will not be overwritten")
            git(root, "tag", "-a", version, "-m", f"Silverfir-nano {version}")
            print(f"Created local tag {version}. Push it with git push {args.remote} refs/tags/{version}")
        else:
            version = verify(root, refs)
            print(f"Release identity verified: {version}. CI, package review and publication approval are still required.")
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release stopped: {error}\n")


if __name__ == "__main__":
    main()
