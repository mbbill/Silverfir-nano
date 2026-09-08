"""Exercise release safeguards against disposable Git repositories."""

import subprocess
import unittest

from ci.release import git, prepare, remote_refs, verify
from ci.test_release_version import ReleaseVersionTests


class ReleaseTests(ReleaseVersionTests):
    def setUp(self):
        super().setUp()
        git(self.root, "init", "-b", "main")
        git(self.root, "config", "user.email", "test@example.invalid")
        git(self.root, "config", "user.name", "Release test")
        git(self.root, "add", ".")
        git(self.root, "commit", "-m", "fixture")
        self.refs = {"refs/heads/main": git(self.root, "rev-parse", "HEAD")}

    def test_prepare_updates_all_surfaces(self):
        prepare(self.root, "99.0.1", self.refs)
        from ci.release_version import check
        self.assertEqual(check(self.root), "99.0.1")

    def test_non_increasing_version(self):
        for version in [self.version, "0.0.1", "0.6", "v0.7.0"]:
            with self.subTest(version=version), self.assertRaises(ValueError):
                prepare(self.root, version, self.refs)
        self.assertEqual(git(self.root, "status", "--porcelain"), "")

    def test_remote_newer_release(self):
        self.refs["refs/tags/99.1"] = "a" * 40
        with self.assertRaisesRegex(ValueError, "existing release"):
            prepare(self.root, "99.0.1", self.refs)

    def test_dirty_tree(self):
        (self.root / "untracked").write_text("user work")
        with self.assertRaisesRegex(ValueError, "clean"):
            prepare(self.root, "99.0.1", self.refs)

    def test_missing_tag(self):
        with self.assertRaisesRegex(ValueError, "Missing local"):
            verify(self.root, self.refs)

    def test_not_merged(self):
        self.refs["refs/heads/main"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "remote main"):
            verify(self.root, self.refs, False)

    def test_lightweight_tag(self):
        git(self.root, "tag", self.version)
        with self.assertRaisesRegex(ValueError, "annotated"):
            verify(self.root, self.refs)

    def test_tag_not_pushed(self):
        git(self.root, "tag", "-a", self.version, "-m", "release")
        with self.assertRaisesRegex(ValueError, "Missing remote"):
            verify(self.root, self.refs)

    def test_pushed_tag_and_wrong_remote_commit(self):
        git(self.root, "tag", "-a", self.version, "-m", "release")
        refs = remote_refs(self.root, str(self.root))
        self.assertEqual(verify(self.root, refs), self.version)
        refs[f"refs/tags/{self.version}^{{}}"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "point to HEAD"):
            verify(self.root, refs)

    def test_local_tag_wrong_commit(self):
        git(self.root, "tag", "-a", self.version, "-m", "release")
        git(self.root, "commit", "--allow-empty", "-m", "later")
        self.refs["refs/heads/main"] = git(self.root, "rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "Local release tag"):
            verify(self.root, self.refs)

    def test_remote_failure_is_fatal(self):
        with self.assertRaises(subprocess.CalledProcessError):
            remote_refs(self.root, str(self.root / "missing"))
