"""Print a compact learned-policy rollout trace for diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.learning.policy import LearnedGraphPolicy
from harnessbench.sim.envs import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("inspect", "insert", "route", "branch"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=31_000_000)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()

    env = make_env(args.task)
    env.reset(seed=args.seed, difficulty=args.difficulty)
    policy = LearnedGraphPolicy(
        args.checkpoint,
        args.task,
        name="debug_learned",
    )
    for _ in range(args.steps):
        action = policy.act(env)
        _, reward, terminated, truncated, info = env.step(action)
        if env.step_count == 1 or env.step_count % args.every == 0 or terminated or truncated:
            print(
                json.dumps(
                    {
                        "step": env.step_count,
                        "reward": reward,
                        "debug": policy.last_debug,
                        "info": info,
                    },
                    ensure_ascii=False,
                )
            )
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
