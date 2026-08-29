from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ci import wasmi_startup_ranking


class WasmiStartupRankingTests(unittest.TestCase):
    def synthetic_measurements(
        self,
        *,
        faster_peer: str | None = None,
    ) -> dict[str, dict[str, float]]:
        results = {}
        for engine in wasmi_startup_ranking.EXPECTED_INTERPRETERS:
            groups = wasmi_startup_ranking.STARTUP_GROUPS
            if engine == "wasmtime.pulley":
                groups = tuple(group for group in groups if group != "startup/ffmpeg")
            value = 10.0 if engine == wasmi_startup_ranking.NANO_RUNTIME_ID else 20.0
            if engine == faster_peer:
                value = 5.0
            results[engine] = {group: value for group in groups}
        return results

    def test_manifest_is_the_complete_pinned_interpreter_field(self) -> None:
        self.assertEqual(len(wasmi_startup_ranking.STARTUP_GROUPS), 7)
        self.assertEqual(len(wasmi_startup_ranking.EXPECTED_INTERPRETERS), 23)
        lazy = {
            engine
            for engine in wasmi_startup_ranking.EXPECTED_INTERPRETERS
            if wasmi_startup_ranking.is_lazy_runtime(engine)
        }
        self.assertEqual(len(lazy), 7)
        self.assertNotIn("dlr-wasm-interpreter", lazy)
        self.assertIn("wasmi-v2.lazy-translation.checked", lazy)

    def test_nano_gate_covers_every_non_lazy_peer(self) -> None:
        document = wasmi_startup_ranking.build_document(
            self.synthetic_measurements(),
            platform="x64-linux",
            candidate_sha="candidate",
        )

        self.assertTrue(document["all_non_lazy_beaten"])
        self.assertEqual(document["non_lazy_peer_count"], 15)
        self.assertEqual(document["nano_beaten_peer_count"], 15)
        pulley = next(
            row for row in document["rows"] if row["engine"] == "wasmtime.pulley"
        )
        self.assertIsNone(pulley["seven_case_geomean_ns"])
        self.assertEqual(len(pulley["nano_comparison"]["common_workloads"]), 6)

    def test_gate_reports_a_faster_non_lazy_peer(self) -> None:
        document = wasmi_startup_ranking.build_document(
            self.synthetic_measurements(faster_peer="wasmi-v1.eager.checked"),
            platform="arm64-linux",
            candidate_sha="candidate",
        )
        summary = wasmi_startup_ranking.render_summary(document)

        self.assertFalse(document["all_non_lazy_beaten"])
        self.assertIn("not ahead of `wasmi-v1.eager.checked`", summary)
        self.assertIn("wasmtime.pulley", summary)
        self.assertIn("six common workloads", summary)

    def test_collect_rejects_an_unexpected_missing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            measurements = self.synthetic_measurements()
            del measurements["wasm3.eager"]["startup/bz2"]
            for group in wasmi_startup_ranking.STARTUP_GROUPS:
                records = []
                for engine, results in measurements.items():
                    if group not in results:
                        continue
                    records.append(
                        json.dumps(
                            {
                                "reason": "benchmark-complete",
                                "id": f"{group}/{engine}",
                                "typical": {"estimate": results[group]},
                            }
                        )
                    )
                path = raw_dir / f"{group.replace('/', '__')}.jsonl"
                path.write_text("\n".join(records) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "wasm3.eager: missing"):
                wasmi_startup_ranking.collect_measurements(raw_dir)

    def test_workflow_is_dev_only_and_pins_the_suite(self) -> None:
        workflow = (
            Path(__file__).parents[1]
            / ".github"
            / "workflows"
            / "wasmi-startup-ranking.yml"
        ).read_text(encoding="utf-8")

        self.assertIn('branches: ["dev/**"]', workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("workflow_dispatch:", workflow)
        self.assertIn(wasmi_startup_ranking.WASMI_BENCHMARKS_REVISION, workflow)
        self.assertIn("--require-nano-fastest", workflow)
        self.assertIn("ubuntu-latest", workflow)
        self.assertIn("ubuntu-24.04-arm", workflow)


if __name__ == "__main__":
    unittest.main()
