import unittest

import numpy as np

from harnessbench.sim.core import CableGraph


class CableGraphTests(unittest.TestCase):
    def test_anchor_moves_chain_without_unbounded_stretch(self) -> None:
        positions = np.column_stack((np.linspace(0.1, 0.5, 12), np.full(12, 0.5)))
        cable = CableGraph.chain(positions, self_collision=False)
        for _ in range(10):
            cable.step({11: np.asarray([0.62, 0.58])}, dt=0.025, substeps=2, iterations=20)
        np.testing.assert_allclose(cable.positions[11], [0.62, 0.58])
        self.assertLess(cable.stretch_error(), 0.10)

    def test_counts_nonadjacent_segment_crossing(self) -> None:
        positions = np.asarray([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])
        cable = CableGraph(
            positions,
            np.asarray([[0, 1], [2, 3]]),
            self_collision=False,
        )
        self.assertEqual(cable.crossing_count(), 1)


if __name__ == "__main__":
    unittest.main()
