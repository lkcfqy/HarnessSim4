"""Small LeRobot-v3 parquet reader for the columns used by this project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrajectoryTable:
    state: np.ndarray
    action: np.ndarray
    episode: np.ndarray
    frame: np.ndarray
    timestamp: np.ndarray

    def __post_init__(self) -> None:
        rows = self.state.shape[0]
        if self.state.ndim != 2 or self.action.ndim != 2:
            raise ValueError("state and action must both be rank-2 arrays")
        if self.action.shape[0] != rows:
            raise ValueError("state and action row counts differ")
        for name in ("episode", "frame", "timestamp"):
            if getattr(self, name).shape != (rows,):
                raise ValueError(f"{name} must have shape ({rows},)")

    @property
    def episodes(self) -> np.ndarray:
        return np.unique(self.episode)


def find_data_files(data_dir: Path) -> list[Path]:
    files = sorted((data_dir / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No LeRobot parquet files found below {data_dir / 'data'}. "
            "Run `python -m harnessbench download-public` first."
        )
    return files


def load_trajectories(data_dir: Path) -> TrajectoryTable:
    cache_path = data_dir / "data" / "harnessbench_rows.npz"
    if cache_path.exists():
        with np.load(cache_path) as arrays:
            table = TrajectoryTable(
                state=np.asarray(arrays["state"], dtype=np.float64),
                action=np.asarray(arrays["action"], dtype=np.float64),
                episode=np.asarray(arrays["episode"], dtype=np.int64),
                frame=np.asarray(arrays["frame"], dtype=np.int64),
                timestamp=np.asarray(arrays["timestamp"], dtype=np.float64),
            )
        order = np.lexsort((table.frame, table.episode))
        return TrajectoryTable(
            state=table.state[order],
            action=table.action[order],
            episode=table.episode[order],
            frame=table.frame[order],
            timestamp=table.timestamp[order],
        )

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # Import may be blocked by endpoint application-control policy.
        raise RuntimeError(
            "Neither the portable NumPy cache nor a working PyArrow installation is available. "
            "Run `python -m harnessbench download-public` to build the API cache."
        ) from exc

    columns = ["observation.state", "action", "episode_index", "frame_index", "timestamp"]
    chunks: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    for path in find_data_files(data_dir):
        table = pq.read_table(path, columns=columns)
        values = table.to_pydict()
        chunks["observation.state"].append(
            np.asarray(values["observation.state"], dtype=np.float64)
        )
        chunks["action"].append(np.asarray(values["action"], dtype=np.float64))
        chunks["episode_index"].append(np.asarray(values["episode_index"], dtype=np.int64))
        chunks["frame_index"].append(np.asarray(values["frame_index"], dtype=np.int64))
        chunks["timestamp"].append(np.asarray(values["timestamp"], dtype=np.float64))

    state = np.concatenate(chunks["observation.state"])
    action = np.concatenate(chunks["action"])
    episode = np.concatenate(chunks["episode_index"])
    frame = np.concatenate(chunks["frame_index"])
    timestamp = np.concatenate(chunks["timestamp"])

    order = np.lexsort((frame, episode))
    return TrajectoryTable(
        state=state[order],
        action=action[order],
        episode=episode[order],
        frame=frame[order],
        timestamp=timestamp[order],
    )


def dataset_summary(table: TrajectoryTable) -> dict:
    episode_ids, counts = np.unique(table.episode, return_counts=True)
    return {
        "frames": int(table.state.shape[0]),
        "episodes": len(episode_ids),
        "state_dim": int(table.state.shape[1]),
        "action_dim": int(table.action.shape[1]),
        "episode_frame_min": int(counts.min()),
        "episode_frame_max": int(counts.max()),
        "timestamp_min": float(table.timestamp.min()),
        "timestamp_max": float(table.timestamp.max()),
        "finite": bool(np.isfinite(table.state).all() and np.isfinite(table.action).all()),
    }
