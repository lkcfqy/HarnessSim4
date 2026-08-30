import tempfile
import unittest
from pathlib import Path

import numpy as np

from harnessbench.ridge import RidgePolicy


class RidgePolicyTests(unittest.TestCase):
    def test_learns_and_round_trips_linear_mapping(self) -> None:
        rng = np.random.default_rng(12)
        x = rng.normal(size=(300, 4))
        mapping = rng.normal(size=(4, 2))
        y = x @ mapping + np.asarray([0.2, -0.4])
        policy = RidgePolicy(alpha=1e-8).fit(x, y)
        np.testing.assert_allclose(policy.predict_delta(x), y, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            policy.save(path)
            restored = RidgePolicy.load(path)
            np.testing.assert_allclose(restored.predict_delta(x), y, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
