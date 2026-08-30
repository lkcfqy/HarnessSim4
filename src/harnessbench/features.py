"""Leakage-safe feature construction for an intentionally small baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from harnessbench.lerobot import TrajectoryTable


@dataclass(frozen=True)
class SupervisedData:
    x: np.ndarray
    y_delta: np.ndarray
    state: np.ndarray
    action: np.ndarray
    episode: np.ndarray
    frame: np.ndarray


def split_episode_ids(
    episode_ids: np.ndarray, train_fraction: float = 0.8, seed: int = 202609
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole episodes, never neighboring frames from one episode."""

    unique = np.unique(np.asarray(episode_ids, dtype=np.int64))
    if unique.size < 2:
        raise ValueError("At least two episodes are required for a held-out evaluation")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    cut = round(shuffled.size * train_fraction)
    cut = min(max(cut, 1), shuffled.size - 1)
    return np.sort(shuffled[:cut]), np.sort(shuffled[cut:])


def build_features(table: TrajectoryTable) -> SupervisedData:
    """Use current state, within-episode velocity and normalized episode phase."""

    state = table.state
    velocity = np.zeros_like(state)
    same_episode = table.episode[1:] == table.episode[:-1]
    velocity[1:][same_episode] = state[1:][same_episode] - state[:-1][same_episode]

    phase = np.zeros((len(state), 1), dtype=np.float64)
    for episode_id in table.episodes:
        mask = table.episode == episode_id
        frames = table.frame[mask].astype(np.float64)
        denominator = max(float(frames.max() - frames.min()), 1.0)
        phase[mask, 0] = (frames - frames.min()) / denominator

    x = np.concatenate((state, velocity, phase, phase**2), axis=1)
    return SupervisedData(
        x=x,
        y_delta=table.action - state,
        state=state,
        action=table.action,
        episode=table.episode,
        frame=table.frame,
    )


def select_episodes(data: SupervisedData, episode_ids: np.ndarray) -> SupervisedData:
    mask = np.isin(data.episode, episode_ids)
    return SupervisedData(
        x=data.x[mask],
        y_delta=data.y_delta[mask],
        state=data.state[mask],
        action=data.action[mask],
        episode=data.episode[mask],
        frame=data.frame[mask],
    )
