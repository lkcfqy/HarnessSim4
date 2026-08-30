"""Generate leakage-safe expert demonstrations from the four simulators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np

from harnessbench.learning.graph import (
    TASK_ORDER,
    action_mask,
    binary_action_mask,
    encode_environment,
    normalize_action,
)
from harnessbench.sim.envs import make_env
from harnessbench.sim.envs.route import ROUTE_TARGET_ASSIGNMENTS

SPLIT_TO_ID = {"train": 0, "validation": 1, "test": 2}
ID_TO_SPLIT = {value: key for key, value in SPLIT_TO_ID.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_expert_dataset(
    output_path: Path,
    *,
    tasks: Iterable[str] = TASK_ORDER,
    train_episodes: int = 12,
    validation_episodes: int = 4,
    test_episodes: int = 8,
    difficulties: Iterable[float] = (0.2, 0.5, 0.8),
    record_every: int = 2,
    base_seed: int = 430_000,
    route_target_assignments: Iterable[tuple[int, int, int]] = ROUTE_TARGET_ASSIGNMENTS,
) -> dict:
    """Collect observable graph states paired with heuristic expert actions.

    Splits are made by whole episode seed before collecting any frame. Adjacent
    states from one trajectory can therefore never leak across splits.
    """

    task_list = list(tasks)
    unknown = set(task_list) - set(TASK_ORDER)
    if unknown:
        raise KeyError(f"unknown tasks: {sorted(unknown)}")
    counts = {
        "train": int(train_episodes),
        "validation": int(validation_episodes),
        "test": int(test_episodes),
    }
    if any(value < 1 for value in counts.values()):
        raise ValueError("every split needs at least one episode per task")
    difficulty_values = [float(value) for value in difficulties]
    if not difficulty_values or any(not 0.0 <= value <= 1.0 for value in difficulty_values):
        raise ValueError("difficulties must be a non-empty subset of [0, 1]")
    if record_every < 1:
        raise ValueError("record_every must be at least one")
    route_assignments = [tuple(int(index) for index in value) for value in route_target_assignments]
    if not route_assignments:
        raise ValueError("route_target_assignments must be non-empty")
    if any(
        len(value) != 3
        or any(left >= right for left, right in pairwise(value))
        or value[0] < 1
        or value[-1] >= 28
        for value in route_assignments
    ):
        raise ValueError("every RouteBot target assignment needs three increasing internal indices")

    arrays: dict[str, list[np.ndarray | float | int]] = {
        "node_features": [],
        "physical_adjacency": [],
        "semantic_adjacency": [],
        "node_mask": [],
        "global_features": [],
        "task_id": [],
        "action": [],
        "action_mask": [],
        "binary_action_mask": [],
        "pointer_target": [],
        "pointer_mask": [],
        "split": [],
        "episode_seed": [],
        "physical_seed": [],
        "episode_step": [],
        "difficulty": [],
    }
    episode_manifest: list[dict] = []
    split_seed_sets: dict[str, set[int]] = {name: set() for name in counts}
    split_physical_seed_sets: dict[str, set[int]] = {name: set() for name in counts}
    split_offsets = {"train": 0, "validation": 10_000_000, "test": 20_000_000}

    for task_index, task in enumerate(task_list):
        for split_name, episode_count in counts.items():
            for episode_index in range(episode_count):
                split_base = base_seed + task_index * 100_000 + split_offsets[split_name]
                episode_seed = split_base + episode_index
                physical_episode_index = episode_index
                if task == "route":
                    physical_episode_index = episode_index // len(route_assignments)
                elif task == "branch":
                    physical_episode_index = episode_index // 2
                physical_seed = split_base + physical_episode_index
                difficulty = difficulty_values[
                    (physical_episode_index + task_index) % len(difficulty_values)
                ]
                split_seed_sets[split_name].add(episode_seed)
                split_physical_seed_sets[split_name].add(physical_seed)
                env = make_env(task)
                if task == "branch":
                    # Balance the semantic counterfactual exactly within every
                    # split; the physical seed is unchanged and never encoded.
                    env.semantic_swap_override = bool(episode_index % 2)
                if task == "route":
                    # Repeat one frozen physical scene across every mapping;
                    # episode_seed remains unique so action chunks cannot cross
                    # semantic variants, while physical_seed records pairing.
                    assignment_index = episode_index % len(route_assignments)
                    env.target_indices_override = route_assignments[assignment_index]
                env.reset(seed=physical_seed, difficulty=difficulty)
                expert_rng = np.random.default_rng(physical_seed + 100_003)
                terminated = truncated = False
                samples_before = len(arrays["task_id"])
                while not (terminated or truncated):
                    action = env.policy_action("topology", expert_rng)
                    if env.step_count % record_every == 0:
                        graph = encode_environment(task, env)
                        arrays["node_features"].append(graph.node_features)
                        arrays["physical_adjacency"].append(graph.physical_adjacency)
                        arrays["semantic_adjacency"].append(graph.semantic_adjacency)
                        arrays["node_mask"].append(graph.node_mask)
                        arrays["global_features"].append(graph.global_features)
                        arrays["task_id"].append(graph.task_id)
                        arrays["action"].append(normalize_action(task, action))
                        arrays["action_mask"].append(action_mask(task))
                        arrays["binary_action_mask"].append(binary_action_mask(task))
                        valid_nodes = np.flatnonzero(graph.node_mask > 0.0)
                        pointer_target = np.zeros(2, dtype=np.int64)
                        first_distances = np.linalg.norm(
                            graph.node_features[valid_nodes, :2] - np.asarray(action[:2]),
                            axis=1,
                        )
                        pointer_target[0] = int(valid_nodes[int(np.argmin(first_distances))])
                        if task == "branch":
                            second_distances = np.linalg.norm(
                                graph.node_features[valid_nodes, :2] - np.asarray(action[3:5]),
                                axis=1,
                            )
                            pointer_target[1] = int(valid_nodes[int(np.argmin(second_distances))])
                        arrays["pointer_target"].append(pointer_target)
                        arrays["pointer_mask"].append(
                            np.asarray([1.0, float(task == "branch")], dtype=np.float32)
                        )
                        arrays["split"].append(SPLIT_TO_ID[split_name])
                        arrays["episode_seed"].append(episode_seed)
                        arrays["physical_seed"].append(physical_seed)
                        arrays["episode_step"].append(env.step_count)
                        arrays["difficulty"].append(difficulty)
                    _, _, terminated, truncated, _ = env.step(action)
                episode_manifest.append(
                    {
                        "task": task,
                        "split": split_name,
                        "seed": episode_seed,
                        "physical_seed": physical_seed,
                        "difficulty": difficulty,
                        "success": bool(terminated and env.success()),
                        "steps": env.step_count,
                        "samples": len(arrays["task_id"]) - samples_before,
                        "semantic_target_swap": (
                            bool(env.semantic_target_swap) if task == "branch" else None
                        ),
                        "route_target_assignment": (
                            [int(value) for value in env.target_indices]
                            if task == "route"
                            else None
                        ),
                    }
                )

    if any(
        split_seed_sets[left] & split_seed_sets[right]
        for left in split_seed_sets
        for right in split_seed_sets
        if left < right
    ):
        raise AssertionError("episode seed leakage across splits")
    if any(
        split_physical_seed_sets[left] & split_physical_seed_sets[right]
        for left in split_physical_seed_sets
        for right in split_physical_seed_sets
        if left < right
    ):
        raise AssertionError("physical seed leakage across splits")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = {
        key: np.stack(value).astype(np.float32)
        if key
        in {
            "node_features",
            "physical_adjacency",
            "semantic_adjacency",
            "node_mask",
            "global_features",
            "action",
            "action_mask",
            "binary_action_mask",
            "pointer_mask",
        }
        else np.asarray(value, dtype=np.int64 if key != "difficulty" else np.float32)
        for key, value in arrays.items()
    }
    np.savez_compressed(output_path, **stacked)
    digest = _sha256(output_path)

    split_samples = {
        name: int(np.sum(stacked["split"] == split_id)) for name, split_id in SPLIT_TO_ID.items()
    }
    success_by_split = {
        name: float(
            np.mean(
                [episode["success"] for episode in episode_manifest if episode["split"] == name]
            )
        )
        for name in SPLIT_TO_ID
    }
    branch_swap_counts = {
        name: {
            str(swap).lower(): sum(
                episode["task"] == "branch"
                and episode["split"] == name
                and episode["semantic_target_swap"] is swap
                for episode in episode_manifest
            )
            for swap in (False, True)
        }
        for name in SPLIT_TO_ID
    }
    route_assignment_counts = {
        name: {
            "-".join(str(value) for value in assignment): sum(
                episode["task"] == "route"
                and episode["split"] == name
                and episode["route_target_assignment"] == list(assignment)
                for episode in episode_manifest
            )
            for assignment in route_assignments
        }
        for name in SPLIT_TO_ID
    }
    metadata = {
        "name": "HarnessSim4 observable expert graph demonstrations",
        "schema_version": "0.3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(output_path.resolve()),
        "sha256": digest,
        "tasks": task_list,
        "difficulties": difficulty_values,
        "record_every": record_every,
        "base_seed": base_seed,
        "episodes_per_task": counts,
        "split_samples": split_samples,
        "episode_success_rate": success_by_split,
        "branch_semantic_swap_counts": branch_swap_counts,
        "route_target_assignments": [list(value) for value in route_assignments],
        "route_assignment_counts": route_assignment_counts,
        "route_primary_success_definition": (
            "all assigned cable particles latched to their ordered clips and endpoint placed; "
            "2-D crossing-free completion is a separate strict metric"
        ),
        "split_seeds": {name: sorted(values) for name, values in split_seed_sets.items()},
        "split_physical_seeds": {
            name: sorted(values) for name, values in split_physical_seed_sets.items()
        },
        "leakage_check": "passed: disjoint whole-episode seeds",
        "physical_pairing": (
            "RouteBot repeats each physical seed across every configured target assignment; "
            "BranchBot repeats each physical seed across both semantic swaps; physical seeds "
            "are disjoint across splits"
        ),
        "observation_scope": "observable state only; no ground-truth defect labels",
        "episodes": episode_manifest,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "dataset": str(output_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "sha256": digest,
        "samples": len(stacked["task_id"]),
        "split_samples": split_samples,
        "episode_success_rate": success_by_split,
    }


def load_expert_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def filter_successful_expert_dataset(
    source_path: Path,
    output_path: Path,
    *,
    train_episodes: int,
    validation_episodes: int,
    test_episodes: int,
) -> dict:
    """Build a difficulty-balanced subset containing only successful episodes."""

    requested = {
        "train": int(train_episodes),
        "validation": int(validation_episodes),
        "test": int(test_episodes),
    }
    if any(value < 1 for value in requested.values()):
        raise ValueError("every split needs at least one successful episode per task")
    metadata_path = source_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = load_expert_dataset(source_path)
    difficulties = [float(value) for value in source_metadata["difficulties"]]
    selected_episodes: list[dict] = []
    selection_counts: dict[str, dict[str, dict[str, int]]] = {}

    def stratum_key(task: str, episode: dict) -> tuple:
        key: list[object] = [float(episode["difficulty"])]
        if task == "route" and episode.get("route_target_assignment") is not None:
            key.append(tuple(int(value) for value in episode["route_target_assignment"]))
        if task == "branch" and episode.get("semantic_target_swap") is not None:
            key.append(bool(episode["semantic_target_swap"]))
        return tuple(key)

    def stratum_label(task: str, key: tuple) -> str:
        label = f"difficulty={float(key[0]):g}"
        if task == "route" and len(key) > 1:
            label += "|assignment=" + "-".join(str(value) for value in key[1])
        if task == "branch" and len(key) > 1:
            label += f"|semantic_swap={str(bool(key[1])).lower()}"
        return label

    for task in source_metadata["tasks"]:
        selection_counts[task] = {}
        for split_name, episode_count in requested.items():
            source_split_episodes = [
                episode
                for episode in source_metadata["episodes"]
                if episode["task"] == task and episode["split"] == split_name
            ]
            strata = sorted({stratum_key(task, episode) for episode in source_split_episodes})
            if not strata:
                raise ValueError(f"source has no episodes for {task}/{split_name}")
            quota = {
                stratum: episode_count // len(strata)
                + int(index < episode_count % len(strata))
                for index, stratum in enumerate(strata)
            }
            split_selection: list[dict] = []
            for stratum in strata:
                candidates = sorted(
                    (
                        episode
                        for episode in source_split_episodes
                        if bool(episode["success"])
                        and stratum_key(task, episode) == stratum
                    ),
                    key=lambda episode: int(episode["seed"]),
                )
                required = quota[stratum]
                if len(candidates) < required:
                    raise ValueError(
                        f"need {required} successful {task}/{split_name}/"
                        f"{stratum_label(task, stratum)} "
                        f"episodes, found {len(candidates)}; generate a larger source dataset"
                    )
                split_selection.extend(candidates[:required])
            selected_episodes.extend(split_selection)
            selection_counts[task][split_name] = {
                stratum_label(task, stratum): sum(
                    stratum_key(task, episode) == stratum
                    for episode in split_selection
                )
                for stratum in strata
            }

    selected_seeds = {int(episode["seed"]) for episode in selected_episodes}
    mask = np.isin(arrays["episode_seed"], np.asarray(sorted(selected_seeds), dtype=np.int64))
    filtered = {key: value[mask] for key, value in arrays.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **filtered)
    digest = _sha256(output_path)
    split_samples = {
        name: int(np.sum(filtered["split"] == split_id))
        for name, split_id in SPLIT_TO_ID.items()
    }
    split_seeds = {
        name: sorted(
            int(episode["seed"])
            for episode in selected_episodes
            if episode["split"] == name
        )
        for name in SPLIT_TO_ID
    }
    split_physical_seeds = {
        name: sorted(
            {
                int(episode.get("physical_seed", episode["seed"]))
                for episode in selected_episodes
                if episode["split"] == name
            }
        )
        for name in SPLIT_TO_ID
    }
    metadata = {
        "name": "HarnessSim4 successful-only observable expert graph demonstrations",
        "schema_version": "0.4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(output_path.resolve()),
        "sha256": digest,
        "source_dataset": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "tasks": source_metadata["tasks"],
        "difficulties": difficulties,
        "record_every": source_metadata["record_every"],
        "episodes_per_task": requested,
        "selection_counts": selection_counts,
        "route_target_assignments": source_metadata.get("route_target_assignments", []),
        "route_primary_success_definition": source_metadata.get(
            "route_primary_success_definition"
        ),
        "split_samples": split_samples,
        "episode_success_rate": {name: 1.0 for name in SPLIT_TO_ID},
        "split_seeds": split_seeds,
        "split_physical_seeds": split_physical_seeds,
        "leakage_check": (
            "passed: disjoint whole-episode and physical seeds inherited from source"
        ),
        "physical_pairing": source_metadata.get("physical_pairing"),
        "observation_scope": source_metadata["observation_scope"],
        "selection_rule": (
            "successful complete episodes only; deterministic lowest-seed selection balanced "
            "across difficulty and available task-specific semantic-assignment strata"
        ),
        "source_episode_count": len(source_metadata["episodes"]),
        "selected_episode_count": len(selected_episodes),
        "episodes": selected_episodes,
    }
    output_metadata_path = output_path.with_suffix(".json")
    output_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "dataset": str(output_path.resolve()),
        "metadata": str(output_metadata_path.resolve()),
        "sha256": digest,
        "samples": int(mask.sum()),
        "selected_episode_count": len(selected_episodes),
        "split_samples": split_samples,
    }
