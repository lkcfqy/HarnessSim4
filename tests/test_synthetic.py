import json
import tempfile
import unittest
from pathlib import Path

from harnessbench.synthetic import SyntheticSpec, generate_synthetic_dataset


class SyntheticDatasetTests(unittest.TestCase):
    def test_generates_expected_images_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic"
            spec = SyntheticSpec(episodes=3, frames_per_episode=3, image_size=96, seed=4)
            info = generate_synthetic_dataset(output, spec)
            self.assertEqual(info["frames"], 9)
            self.assertEqual(len(list((output / "images").glob("*.png"))), 9)
            lines = (output / "labels.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 9)
            record = json.loads(lines[0])
            self.assertIn("target_delta_xy", record)
            self.assertEqual(record["split"], "train")
            cached = generate_synthetic_dataset(output, spec)
            self.assertEqual(cached["status"], "cached")


if __name__ == "__main__":
    unittest.main()
