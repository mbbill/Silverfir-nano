from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "interp-eager-tier-census.yml"
TOOL = ROOT / "tools" / "eager-tier-census"


class EagerTierCensusTests(unittest.TestCase):
    def test_workflow_is_manual_pinned_and_covers_all_startup_modules(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", text)
        self.assertNotIn("schedule:", text)
        self.assertIn("16a3d7c8fdb05506c116a9451175732d1ac77099", text)
        for module in (
            "bz2",
            "pulldown-cmark",
            "spidermonkey",
            "ffmpeg",
            "coremark",
            "argon2",
            "erc20",
        ):
            self.assertIn(f"{module}=wasmi-benchmarks/", text)

    def test_census_contains_no_wall_clock_measurement(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (TOOL / "src").glob("*.rs")
        )
        for forbidden in ("Instant", "SystemTime", "criterion", "elapsed("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
