import json
import tempfile
import unittest
from pathlib import Path

from harnessbench.sim.benchmark import run_benchmark, save_benchmark
from harnessbench.sim.visualization import generate_visual_demos


class BenchmarkTests(unittest.TestCase):
    def test_rejects_zero_workers(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers"):
            run_benchmark(episodes=1, workers=0)

    def test_exports_paired_benchmark(self) -> None:
        report = run_benchmark(
            tasks=["insert", "route"],
            policies=["topology", "geometry"],
            difficulties=[0.5],
            episodes=1,
            base_seed=9,
        )
        self.assertEqual(len(report["records"]), 4)
        self.assertEqual(len(report["paired_topology_advantage"]), 2)
        self.assertEqual(len(report["aggregate_by_task_policy"]), 4)
        with tempfile.TemporaryDirectory() as directory:
            exported = save_benchmark(report, Path(directory))
            self.assertTrue(Path(exported["report"]).exists())
            loaded = json.loads(Path(exported["report"]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["benchmark"], "HarnessSim4")

    def test_visual_demo_is_self_contained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = generate_visual_demos(
                Path(directory),
                tasks=["insert"],
                seed=3,
                difficulty=0.4,
                record_every=8,
            )
            dashboard = Path(result["dashboard"])
            content = dashboard.read_text(encoding="utf-8")
            self.assertIn("data:image/jpeg;base64,", content)
            self.assertNotIn("/*__REPLAY_DATA__*/", content)
            self.assertTrue((Path(directory) / "insert_topology.gif").exists())


if __name__ == "__main__":
    unittest.main()
