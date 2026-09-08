"""Check release identity without compiling or accessing the network."""

import argparse
from pathlib import Path
import re
import tomllib


PUBLISHED = {"sf-nano-core", "sf-nano-tracked-alloc"}


def read_toml(path):
    return tomllib.loads(path.read_text())


def check(root, tag=None):
    workspace = read_toml(root / "Cargo.toml")["workspace"]
    version = workspace["package"]["version"]
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError("Release version must be MAJOR.MINOR.PATCH")
    found = set()
    for member in workspace["members"]:
        package = read_toml(root / member / "Cargo.toml")["package"]
        if package.get("publish") is False:
            continue
        name = package["name"]
        found.add(name)
        if package["version"] != {"workspace": True}:
            raise ValueError(f"{name} must inherit workspace.package.version")
    if found != PUBLISHED:
        raise ValueError(f"Review the set of published packages: {sorted(found)}")
    core = read_toml(root / "sf-nano-core/Cargo.toml")
    dependency = core["dependencies"]["tracked_alloc"]
    if (dependency.get("package"), dependency.get("version"), dependency.get("path")) != (
        "sf-nano-tracked-alloc", version, "../tools/tracked-alloc"
    ):
        raise ValueError("Core's helper dependency must match the release version and path")
    locked = read_toml(root / "Cargo.lock")["package"]
    for name in PUBLISHED:
        entries = [p for p in locked if p["name"] == name]
        if len(entries) != 1 or entries[0]["version"] != version or "source" in entries[0]:
            raise ValueError(f"Cargo.lock must select local {name} {version}")
    readme = (root / "sf-nano-core/README.md").read_text()
    requirements = re.findall(r'sf-nano-core\s*=\s*(?:\{\s*version\s*=\s*)?"([^"]+)"', readme)
    series = ".".join(version.split(".")[:2])
    if not requirements or any(req != series for req in requirements):
        raise ValueError(f"README installation examples must use version {series}")
    if tag is not None and tag != version:
        raise ValueError(f"Release tag {tag!r} must equal {version!r}")
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Exact release tag, e.g. 0.6.0")
    args = parser.parse_args()
    try:
        version = check(Path(__file__).resolve().parents[1], args.tag)
    except (ValueError, KeyError) as error:
        parser.exit(1, f"Release version check failed: {error}\n")
    print(f"Release version verified: {version}")


if __name__ == "__main__":
    main()
