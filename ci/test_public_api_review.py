import copy
from pathlib import Path
import subprocess
import tempfile
import unittest

from ci.public_api import SUPPORT_SNAPSHOTS
from ci.public_api_review import (
    SNAPSHOTS, bootstrap_required, compare, environment_problems, evidence_digest,
    prepare, revision, verify_checkout,
)


class PublicApiReviewTests(unittest.TestCase):
    def captures(self, root):
        base, head = root / "base", root / "head"
        for directory in (base, head):
            directory.mkdir()
            for name in SNAPSHOTS:
                (directory / name).write_text("pub struct Existing\n", encoding="utf-8")
        return base, head

    def test_snapshot_edit_cannot_approve_an_api_addition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, head = self.captures(root)
            (head / "interp.txt").write_text(
                "pub struct Existing\npub fn added()\n", encoding="utf-8"
            )
            report = prepare(base, head, root / "out", "a" * 40, "b" * 40,
                             ["ci/api/interp.txt"])
            self.assertTrue(report["review_required"])
            self.assertEqual(report["changed_surfaces"], ["interp.txt"])
            self.assertIn("+pub fn added()", (root / "out/public-api.diff").read_text())
            self.assertNotEqual(report["base_digest"], report["head_digest"])

    def test_unchanged_api_passes_but_policy_edits_still_need_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, head = self.captures(root)
            report = prepare(base, head, root / "out", "a" * 40, "b" * 40, [])
            self.assertFalse(report["review_required"])
            report = prepare(base, head, root / "out", "a" * 40, "b" * 40,
                             [".github/workflows/public-api.yml"])
            self.assertTrue(report["review_required"])

    def test_published_support_api_and_features_require_human_review(self):
        for name in SUPPORT_SNAPSHOTS:
            with self.subTest(surface=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base, head = self.captures(root)
                (head / name).write_text("changed support contract\n")
                report = prepare(base, head, root / "out", "a" * 40, "b" * 40, [])
                self.assertTrue(report["review_required"])
                self.assertEqual(report["changed_surfaces"], [name])
                self.assertNotEqual(report["base_digest"], report["head_digest"])
                (base / name).unlink()
                with self.assertRaises(FileNotFoundError):
                    compare(base, head)

    def test_initial_baseline_requires_review_of_the_entire_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, head = self.captures(root)
            report = prepare(base, head, root / "out", "a" * 40, "b" * 40, [],
                             bootstrap=True)
            self.assertTrue(report["review_required"])
            self.assertIsNone(report["base_digest"])
            self.assertEqual(set(report["changed_surfaces"]), set(SNAPSHOTS))

    def test_missing_capture_and_mutated_features_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, head = self.captures(Path(temporary))
            (head / "features.json").write_text("changed features\n", encoding="utf-8")
            self.assertEqual(list(compare(base, head)), ["features.json"])
            (base / "default.txt").unlink()
            with self.assertRaises(FileNotFoundError):
                compare(base, head)
            with self.assertRaises(FileNotFoundError):
                evidence_digest(base)

    def test_review_must_be_pinned_to_full_commits(self):
        for value in ("main", "123abcd", "-a" + "0" * 38, "a" * 40 + "\n"):
            with self.assertRaises(ValueError):
                revision(value)
        self.assertEqual(revision("a" * 40), "a" * 40)

    def test_bootstrap_and_evidence_use_git_revisions_not_worktree_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                return subprocess.run(
                    ["git", *args], cwd=root, check=True, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                ).stdout.strip()

            git("init", "-q")
            (root / "source.rs").write_text("initial\n", encoding="utf-8")
            git("add", ".")
            git("-c", "user.name=API fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-qm", "initial")
            base = git("rev-parse", "HEAD")
            self.assertTrue(bootstrap_required(root, base))
            (root / "ci").mkdir()
            (root / "ci/public_api.py").write_text("tooling\n", encoding="utf-8")
            git("add", ".")
            git("-c", "user.name=API fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-qm", "API tooling")
            head = git("rev-parse", "HEAD")
            self.assertTrue(bootstrap_required(root, base))
            self.assertFalse(bootstrap_required(root, head))
            verify_checkout(root, head)
            with self.assertRaises(ValueError):
                verify_checkout(root, base)
            (root / "source.rs").write_text("dirty candidate\n", encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError):
                verify_checkout(root, head)

    def test_missing_unprotected_or_bypassable_environment_fails_closed(self):
        document = {
            "name": "public-api-review", "can_admins_bypass": False,
            "protection_rules": [{
                "type": "required_reviewers", "prevent_self_review": False,
                "reviewers": [{"type": "User", "reviewer": {"login": "mbbill"}}],
            }],
        }
        self.assertEqual(environment_problems(document), [])
        self.assertTrue(environment_problems({}))
        for change in (
            {"protection_rules": []}, {"can_admins_bypass": True},
            {"can_admins_bypass": None}, {"name": "typo"},
        ):
            broken = dict(document, **change)
            self.assertTrue(environment_problems(broken))
        wrong_reviewer = copy.deepcopy(document)
        wrong_reviewer["protection_rules"][0]["reviewers"][0]["reviewer"]["login"] = "bot"
        self.assertTrue(environment_problems(wrong_reviewer))
        extra_reviewer = copy.deepcopy(document)
        extra_reviewer["protection_rules"][0]["reviewers"].append(
            {"type": "User", "reviewer": {"login": "bot"}}
        )
        self.assertTrue(environment_problems(extra_reviewer))


if __name__ == "__main__":
    unittest.main()
