import unittest

import numpy as np

from harnessbench.features import build_features, split_episode_ids
from harnessbench.lerobot import TrajectoryTable


class FeatureTests(unittest.TestCase):
    def test_episode_split_is_disjoint_and_deterministic(self) -> None:
        episodes = np.arange(10)
        first = split_episode_ids(episodes, train_fraction=0.8, seed=7)
        second = split_episode_ids(episodes, train_fraction=0.8, seed=7)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 2)
        self.assertEqual(np.intersect1d(*first).size, 0)

    def test_velocity_resets_at_episode_boundary(self) -> None:
        state = np.asarray([[0.0], [1.0], [10.0], [12.0]])
        table = TrajectoryTable(
            state=state,
            action=state + 0.5,
            episode=np.asarray([0, 0, 1, 1]),
            frame=np.asarray([0, 1, 0, 1]),
            timestamp=np.asarray([0.0, 0.1, 0.0, 0.1]),
        )
        features = build_features(table)
        # state dim is one, so velocity is the second feature.
        np.testing.assert_allclose(features.x[:, 1], [0.0, 1.0, 0.0, 2.0])
        np.testing.assert_allclose(features.x[:, 2], [0.0, 1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
