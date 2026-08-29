from __future__ import annotations

import unittest

from ci import startup_stage_profile as profile


def sample(case: str) -> dict:
    values = {
        "startup.total": 1_000,
        "drop": 20,
        "parser.total": 100,
        "instance.build.total": 500,
        "instance.setup": 100,
        "instance.memories": 20,
        "instance.globals": 20,
        "instance.tables": 20,
        "instance.stack_deferred": 10,
        "predecode.total": 300,
        "predecode.decode": 50,
        "predecode.scratch": 20,
        "predecode.pinned_census": 30,
        "link.total": 200,
        "link.cells.total": 100,
        "link.handler_selection": 20,
        "link.call_fixup": 30,
        "link.finalize": 40,
        "instance.element_segments": 10,
        "instance.data_segments": 10,
        "instance.lease": 10,
    }
    return {
        "case": case,
        "iteration": 0,
        "stages": {
            name: {"nanos": nanos, "calls": 1}
            for name, nanos in values.items()
        },
    }


class StartupStageProfileTests(unittest.TestCase):
    def test_nested_stages_are_reduced_to_exclusive_buckets(self) -> None:
        exclusive, negative = profile.derive(sample("bz2"))
        self.assertFalse(negative)
        self.assertEqual(exclusive["predecode.lowering_control"], 200)
        self.assertEqual(exclusive["link.cell_transform"], 80)
        self.assertEqual(exclusive["link.other"], 30)
        self.assertEqual(sum(exclusive.values()), 1_000)

    def test_summary_requires_and_ranks_all_seven_workloads(self) -> None:
        samples = [sample(name) for name, _ in profile.CASES]
        summary = profile.summarize(samples)
        self.assertEqual(summary["ranked_stages"][0], "predecode.lowering_control")
        self.assertFalse(summary["negative_residuals"])
        self.assertIn("Per-workload bottleneck", profile.markdown(summary))


if __name__ == "__main__":
    unittest.main()
