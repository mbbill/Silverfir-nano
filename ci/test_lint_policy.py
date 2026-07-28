from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ci.lint_policy import check


EMPTY_MANIFEST = "version = 1\n"


class LintPolicyTests(unittest.TestCase):
    def run_policy(self, source: str, manifest: str = EMPTY_MANIFEST) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "fixture"\nversion = "0.0.0"\n'
                'edition = "2021"\n',
                encoding="utf-8",
            )
            source_path = root / "src" / "lib.rs"
            source_path.parent.mkdir()
            source_path.write_text(source, encoding="utf-8")
            manifest_path = root / "lint_suppressions.toml"
            manifest_path.write_text(manifest, encoding="utf-8")
            return check(root, manifest_path)

    def test_clean_source_passes(self) -> None:
        self.assertEqual(self.run_policy("pub fn live() {}\n"), [])

    def test_unapproved_allow_dead_code_fails(self) -> None:
        errors = self.run_policy(
            '#[allow(dead_code, reason = "temporary")]\nfn hidden() {}\n'
        )
        self.assertTrue(any("unapproved allow(dead_code)" in error for error in errors))

    def test_allow_warnings_is_never_permitted(self) -> None:
        manifest = """\
version = 1

[[suppression]]
path = "src/lib.rs"
kind = "allow"
lints = ["warnings"]
anchor = "fn hidden() {}"
reason = "even an approval cannot disable all warnings"
"""
        errors = self.run_policy(
            '#[allow(warnings, reason = "bad")]\nfn hidden() {}\n', manifest
        )
        self.assertTrue(any("never permitted" in error for error in errors))

    def test_exact_approved_exception_passes(self) -> None:
        manifest = """\
version = 1

[[suppression]]
path = "src/lib.rs"
kind = "expect"
lints = ["dead_code"]
anchor = "fn retained_for_ffi() {}"
reason = "called by firmware through a symbol name"
"""
        errors = self.run_policy(
            '#[expect(dead_code, reason = "called through FFI")]\n'
            "fn retained_for_ffi() {}\n",
            manifest,
        )
        self.assertEqual(errors, [])

    def test_suppression_without_inline_reason_fails(self) -> None:
        errors = self.run_policy("#[expect(dead_code)]\nfn hidden() {}\n")
        self.assertTrue(any("require `reason" in error for error in errors))

    def test_stale_approval_fails(self) -> None:
        manifest = """\
version = 1

[[suppression]]
path = "src/lib.rs"
kind = "expect"
lints = ["dead_code"]
anchor = "fn removed() {}"
reason = "old exception"
"""
        errors = self.run_policy("pub fn live() {}\n", manifest)
        self.assertTrue(any("stale approval" in error for error in errors))

    def test_line_comment_is_not_a_suppression(self) -> None:
        self.assertEqual(
            self.run_policy("// never add #[allow(dead_code)] here\npub fn live() {}\n"),
            [],
        )

    def test_cfg_attr_suppression_is_detected(self) -> None:
        errors = self.run_policy(
            '#[cfg_attr(unix, allow(unused_imports, reason = "platform API"))]\n'
            "use core::fmt;\n"
        )
        self.assertTrue(
            any("unapproved allow(unused_imports)" in error for error in errors)
        )

    def test_package_without_lint_override_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "fixture"\nversion = "0.0.0"\n',
                encoding="utf-8",
            )
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text("pub fn live() {}\n")
            manifest = root / "lint_suppressions.toml"
            manifest.write_text(EMPTY_MANIFEST)
            errors = check(root, manifest)
        self.assertEqual(errors, [])

    def test_compiler_allow_flag_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "fixture"\nversion = "0.0.0"\n'
                'edition = "2021"\n',
                encoding="utf-8",
            )
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text("pub fn live() {}\n")
            (root / "build.sh").write_text(
                'RUSTFLAGS="-A dead_code" cargo check\n', encoding="utf-8"
            )
            manifest = root / "lint_suppressions.toml"
            manifest.write_text(EMPTY_MANIFEST)
            errors = check(root, manifest)
        self.assertTrue(any("forbidden compiler lint override" in error for error in errors))

    def test_split_rustflags_allow_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "fixture"\nversion = "0.0.0"\n'
                'edition = "2021"\n',
                encoding="utf-8",
            )
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text("pub fn live() {}\n")
            (root / ".cargo").mkdir()
            (root / ".cargo" / "config.toml").write_text(
                'rustflags = ["-A", "dead_code"]\n', encoding="utf-8"
            )
            manifest = root / "lint_suppressions.toml"
            manifest.write_text(EMPTY_MANIFEST)
            errors = check(root, manifest)
        self.assertTrue(any("forbidden compiler lint override" in error for error in errors))

    def test_cargo_lint_override_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "fixture"\nversion = "0.0.0"\n'
                'edition = "2021"\n\n[lints.rust]\ndead_code = "allow"\n',
                encoding="utf-8",
            )
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text("pub fn live() {}\n")
            manifest = root / "lint_suppressions.toml"
            manifest.write_text(EMPTY_MANIFEST)
            errors = check(root, manifest)
        self.assertTrue(any("must not lower dead_code" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
