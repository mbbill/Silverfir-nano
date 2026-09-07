from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ci.public_api import (
    SUPPORT_SNAPSHOTS, api_diff, capture_support_api, compare_snapshots,
    feature_contract, implementation_leaks,
)


class PublicApiTests(unittest.TestCase):
    def test_additions_and_removals_are_both_reviewable_changes(self):
        delta = api_diff("pub struct A\n", "pub struct B\n", "fixture")
        self.assertIn("-pub struct A", delta)
        self.assertIn("+pub struct B", delta)
        self.assertEqual(api_diff("same\n", "same\n", "fixture"), "")

    def test_alias_leaks_are_detected_even_without_a_parity_difference(self):
        api = "pub fn f() -> tracked_alloc::Vec<u8>\npub fn g() -> alloc::vec::Vec<u8>\n"
        self.assertEqual(implementation_leaks(api), [api.splitlines()[0]])

    def test_raw_engine_values_cannot_reappear_in_embedding_signatures(self):
        api = "pub fn f() -> sf_nano_core::vm::value::Value\npub fn g() -> sf_nano_core::Value\n"
        self.assertEqual(implementation_leaks(api), [api.splitlines()[0]])

    def test_diagnostics_cannot_leak_internal_instruction_types(self):
        api = "pub fn f() -> sf_nano_core::vm::interpreter::instr::Op\n"
        self.assertEqual(implementation_leaks(api), [api.strip()])

    def test_public_constructors_cannot_return_hidden_helper_errors(self):
        api = "pub fn new() -> Result<Self, sf_nano_core::utils::limits::LimitsError>\n"
        self.assertEqual(implementation_leaks(api), [api.strip()])

    def test_missing_baselines_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            self.assertTrue(compare_snapshots(path / "accepted", path / "candidate"))

    def test_published_support_crate_captures_both_feature_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "tools/tracked-alloc/Cargo.toml"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('[package]\nedition = "2021"\nrust-version = "1.94"\n'
                                '[features]\nmemprof = []\n')
            output = root / "evidence"
            output.mkdir()
            env = {"CARGO_TARGET_DIR": str(root / "build")}
            with patch("ci.public_api.run", return_value="pub struct Support\n") as run:
                capture_support_api(root, output, "extractor", env)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 4)
            for command in commands[::2]:
                self.assertIn("sf-nano-tracked-alloc", command)
            self.assertNotIn("--features", commands[0])
            self.assertIn("memprof", commands[2])
            for command in commands[1::2]:
                self.assertTrue(any(arg.endswith("/tracked_alloc.json") for arg in command))
            self.assertTrue(all((output / name).is_file() for name in SUPPORT_SNAPSHOTS))
            self.assertIn('"1.94"', (output / "tracked-alloc-features.json").read_text())

    def test_feature_defaults_and_optional_dependencies_are_reviewed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Cargo.toml"
            path.write_text('[features]\ndefault = ["jit"]\njit = []\n', encoding="utf-8")
            before = feature_contract(path)
            path.write_text(
                '[features]\ndefault = ["interp"]\njit = []\ninterp = []\n'
                '[dependencies]\nfiletime = { version = "0.2", optional = true }\n',
                encoding="utf-8",
            )
            after = feature_contract(path)
            self.assertNotEqual(before, after)
            self.assertIn('"filetime"', after)

    def test_minimum_rust_version_and_edition_are_reviewed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Cargo.toml"
            path.write_text('[package]\nedition = "2021"\nrust-version = "1.89"\n')
            before = feature_contract(path)
            path.write_text('[package]\nedition = "2021"\nrust-version = "1.94"\n')
            self.assertNotEqual(before, feature_contract(path))
            before = feature_contract(path)
            path.write_text('[package]\nedition = "2024"\nrust-version = "1.94"\n')
            self.assertNotEqual(before, feature_contract(path))


if __name__ == "__main__":
    unittest.main()
