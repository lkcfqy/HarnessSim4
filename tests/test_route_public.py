import unittest

import numpy as np
from PIL import Image

from harnessbench.learning.route_public import (
    _episode_split,
    _features,
    _paired_episode_comparison,
    _previous_action_baseline,
    _previous_action_feature,
)
from harnessbench.learning.route_public_visuals import compose_multiview
from harnessbench.lerobot import TrajectoryTable
from harnessbench.route_public_data import validate_route_table


def _table() -> TrajectoryTable:
    episode = np.repeat(np.arange(12, dtype=np.int64), 3)
    frame = np.tile(np.arange(3, dtype=np.int64), 12)
    state = np.arange(len(episode) * 8, dtype=np.float64).reshape(len(episode), 8) / 100.0
    action = np.arange(len(episode) * 7, dtype=np.float64).reshape(len(episode), 7) / 50.0
    return TrajectoryTable(
        state=state,
        action=action,
        episode=episode,
        frame=frame,
        timestamp=frame.astype(np.float64) / 10.0,
    )


class RoutePublicDataTests(unittest.TestCase):
    def test_validation_and_feature_construction(self) -> None:
        table = _table()
        report = validate_route_table(table, strict_size=False)
        self.assertEqual(report["episodes"], 12)
        self.assertTrue(report["episode_frame_keys_unique"])
        features = _features(table)
        self.assertEqual(features.shape, (36, 18))
        np.testing.assert_allclose(features[0, 8:16], 0.0)
        np.testing.assert_allclose(features[3, 8:16], 0.0)

    def test_whole_episode_split_is_deterministic_and_disjoint(self) -> None:
        ids = np.arange(100)
        first = _episode_split(ids, seed=17)
        second = _episode_split(ids, seed=17)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        train, validation, test = first
        self.assertEqual(len(np.intersect1d(train, validation)), 0)
        self.assertEqual(len(np.intersect1d(train, test)), 0)
        self.assertEqual(len(np.intersect1d(validation, test)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate(first)), ids)

    def test_previous_action_resets_at_episode_boundaries(self) -> None:
        action = np.asarray([[1.0], [2.0], [9.0], [10.0]])
        episode = np.asarray([0, 0, 1, 1])
        predicted = _previous_action_baseline(action, episode, np.asarray([5.0]))
        np.testing.assert_allclose(predicted[:, 0], [5.0, 1.0, 5.0, 9.0])
        feature = _previous_action_feature(action, episode)
        np.testing.assert_allclose(feature[:, 0], [0.0, 1.0, 0.0, 9.0])

    def test_paired_comparison_uses_whole_episodes(self) -> None:
        true = np.zeros((8, 1))
        reference = np.ones((8, 1))
        candidate = np.full((8, 1), 0.5)
        episode = np.repeat(np.arange(4), 2)
        result = _paired_episode_comparison(
            true,
            reference,
            candidate,
            episode,
            np.asarray([0]),
            seed=5,
            bootstrap_samples=100,
        )
        self.assertEqual(result["candidate_wins"], 4)
        self.assertLess(result["candidate_minus_reference_mean"], 0)

    def test_multiview_composition_has_four_panels(self) -> None:
        frames = [
            (f"view {index}", Image.new("RGB", (16, 16), (index * 30, 20, 40)))
            for index in range(4)
        ]
        sheet = compose_multiview(frames, footer="source", panel_size=32)
        self.assertGreater(sheet.width, 64)
        self.assertGreater(sheet.height, 64)


if __name__ == "__main__":
    unittest.main()
