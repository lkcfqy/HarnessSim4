import unittest

from harnessbench.sim import TopologyPolicy, make_env


class FourRobotEnvironmentTests(unittest.TestCase):
    def _run(self, task: str):
        env = make_env(task)
        env.reset(seed=0, difficulty=0.5)
        policy = TopologyPolicy(seed=0)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(policy.act(env))
        return env, terminated

    def test_topology_policy_completes_all_four_tasks(self) -> None:
        for task in ("inspect", "insert", "route", "branch"):
            with self.subTest(task=task):
                env, terminated = self._run(task)
                self.assertTrue(terminated)
                self.assertTrue(env.success())

    def test_insert_is_deterministic_for_fixed_seed(self) -> None:
        first, _ = self._run("insert")
        second, _ = self._run("insert")
        self.assertEqual(first.step_count, second.step_count)
        self.assertEqual(first.metrics(), second.metrics())

    def test_route_requires_explicit_endpoint_placement(self) -> None:
        env = make_env("route")
        env.reset(seed=0, difficulty=0.5)
        env.bindings = {index: int(particle) for index, particle in enumerate(env.target_indices)}
        env.cable.positions[-1] = env.finish.copy()
        env.endpoint_placed = False
        self.assertFalse(env.success())
        env.endpoint_placed = True
        self.assertTrue(env.success())


if __name__ == "__main__":
    unittest.main()
