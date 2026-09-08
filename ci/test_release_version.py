"""Release identity must fail closed when any published surface drifts."""

from pathlib import Path
import shutil
import tempfile
import unittest

from ci.release_version import check, read_toml


class ReleaseVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = Path(__file__).resolve().parents[1]
        members = read_toml(source / "Cargo.toml")["workspace"]["members"]
        paths = ["Cargo.toml", "Cargo.lock", "sf-nano-core/README.md"]
        paths += [f"{member}/Cargo.toml" for member in members]
        for path in paths:
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / path, target)
        self.version = check(self.root)

    def replace(self, path, before, after):
        path = self.root / path
        text = path.read_text()
        self.assertIn(before, text)
        path.write_text(text.replace(before, after, 1))

    def test_matching_tag(self):
        self.assertEqual(check(self.root, self.version), self.version)

    def test_wrong_tags(self):
        for tag in ["0.5", "0.6", "v" + self.version, "99.0.0"]:
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "Release tag"):
                check(self.root, tag)

    def test_independent_package_version(self):
        self.replace("tools/tracked-alloc/Cargo.toml", "version.workspace = true", 'version = "99.0.0"')
        with self.assertRaisesRegex(ValueError, "inherit"):
            check(self.root)

    def test_dependency_drift(self):
        self.replace("sf-nano-core/Cargo.toml", f'version = "{self.version}"', 'version = "99.0.0"')
        with self.assertRaisesRegex(ValueError, "helper dependency"):
            check(self.root)

    def test_lockfile_drift(self):
        self.replace("Cargo.lock", f'name = "sf-nano-core"\nversion = "{self.version}"', 'name = "sf-nano-core"\nversion = "99.0.0"')
        with self.assertRaisesRegex(ValueError, "Cargo.lock"):
            check(self.root)

    def test_readme_drift(self):
        series = ".".join(self.version.split(".")[:2])
        self.replace("sf-nano-core/README.md", f'"{series}"', '"99.0"')
        with self.assertRaisesRegex(ValueError, "README"):
            check(self.root)

    def test_accidental_publishable_tool(self):
        self.replace("sf-nano-cli/Cargo.toml", "publish = false", "publish = true")
        with self.assertRaises(ValueError):
            check(self.root)
