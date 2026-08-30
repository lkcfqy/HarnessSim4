import tempfile
import unittest
from pathlib import Path

import numpy as np

from harnessbench.lerobot import dataset_summary, load_trajectories


class LeRobotCacheTests(unittest.TestCase):
    def test_loads_portable_cache_and_sorts_episode_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            cache_dir = data_dir / "data"
            cache_dir.mkdir()
            state = np.asarray([[20.0], [10.0], [11.0], [21.0]], dtype=np.float32)
            np.savez_compressed(
                cache_dir / "harnessbench_rows.npz",
                state=state,
                action=state + 0.5,
                episode=np.asarray([2, 1, 1, 2]),
                frame=np.asarray([0, 0, 1, 1]),
                timestamp=np.asarray([0.0, 0.0, 0.1, 0.1]),
            )
            table = load_trajectories(data_dir)
            np.testing.assert_array_equal(table.episode, [1, 1, 2, 2])
            np.testing.assert_allclose(table.state[:, 0], [10.0, 11.0, 20.0, 21.0])
            summary = dataset_summary(table)
            self.assertEqual(summary["episodes"], 2)
            self.assertTrue(summary["finite"])


if __name__ == "__main__":
    unittest.main()
