"""Convert the four heterogeneous environments into one typed graph schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from harnessbench.sim.envs.base import HarnessEnv
from harnessbench.sim.envs.branch import BRANCH_PHASE_BOUNDARIES

TASK_ORDER = ("inspect", "insert", "route", "branch")
TASK_TO_ID = {task: index for index, task in enumerate(TASK_ORDER)}
ACTION_DIMS = {"inspect": 3, "insert": 4, "route": 3, "branch": 6}
MAX_ACTION_DIM = 6
MAX_NODES = 48
GLOBAL_DIM = 12

X = 0
Y = 1
ROLE_CABLE = 2
ROLE_ENDPOINT = 3
ROLE_JUNCTION = 4
ROLE_SITE = 5
ROLE_TARGET = 6
ROLE_ARM = 7
ROLE_CLIP = 8
ROLE_CONNECTOR = 9
SEMANTIC_A = 10
SEMANTIC_B = 11
ORDER = 12
STATE_INSPECTED = 13
STATE_DETECTED = 14
STATE_ACTIVE = 15
CONFIDENCE = 16
DEGREE = 17
PROGRESS = 18
DIFFICULTY = 19
FEATURE_DIM = 20

# The no-topology ablation keeps positions, generic object types and observable
# state, but removes graph connectivity, endpoint/junction identity, semantic
# branch identity and process order.
TOPOLOGY_FEATURE_COLUMNS = (
    ROLE_ENDPOINT,
    ROLE_JUNCTION,
    SEMANTIC_A,
    SEMANTIC_B,
    ORDER,
)


@dataclass(frozen=True)
class EncodedGraph:
    node_features: np.ndarray
    physical_adjacency: np.ndarray
    semantic_adjacency: np.ndarray
    node_mask: np.ndarray
    global_features: np.ndarray
    task_id: int


class _GraphBuilder:
    def __init__(self, progress: float, difficulty: float) -> None:
        self.progress = float(progress)
        self.difficulty = float(difficulty)
        self.nodes: list[np.ndarray] = []
        self.physical_edges: set[tuple[int, int]] = set()
        self.semantic_edges: set[tuple[int, int]] = set()

    def add_node(
        self,
        position: np.ndarray | tuple[float, float],
        *,
        roles: tuple[int, ...] = (),
        semantic_a: float = 0.0,
        semantic_b: float = 0.0,
        order: float = 0.0,
        inspected: float = 0.0,
        detected: float = 0.0,
        active: float = 0.0,
        confidence: float = 0.0,
        degree: float = 0.0,
    ) -> int:
        feature = np.zeros(FEATURE_DIM, dtype=np.float32)
        feature[X : Y + 1] = np.asarray(position, dtype=np.float32)
        for role in roles:
            feature[role] = 1.0
        feature[SEMANTIC_A] = semantic_a
        feature[SEMANTIC_B] = semantic_b
        feature[ORDER] = order
        feature[STATE_INSPECTED] = inspected
        feature[STATE_DETECTED] = detected
        feature[STATE_ACTIVE] = active
        feature[CONFIDENCE] = confidence
        feature[DEGREE] = degree
        feature[PROGRESS] = self.progress
        feature[DIFFICULTY] = self.difficulty
        self.nodes.append(feature)
        if len(self.nodes) > MAX_NODES:
            raise ValueError(f"graph exceeds MAX_NODES={MAX_NODES}")
        return len(self.nodes) - 1

    @staticmethod
    def _edge(edge_set: set[tuple[int, int]], source: int, target: int) -> None:
        if source == target:
            return
        edge_set.add((min(source, target), max(source, target)))

    def physical(self, source: int, target: int) -> None:
        self._edge(self.physical_edges, source, target)

    def semantic(self, source: int, target: int) -> None:
        self._edge(self.semantic_edges, source, target)

    def finish(self, global_features: np.ndarray, task_id: int) -> EncodedGraph:
        node_features = np.zeros((MAX_NODES, FEATURE_DIM), dtype=np.float32)
        node_mask = np.zeros(MAX_NODES, dtype=np.float32)
        count = len(self.nodes)
        node_features[:count] = np.asarray(self.nodes, dtype=np.float32)
        node_mask[:count] = 1.0
        physical = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        semantic = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        for source, target in self.physical_edges:
            physical[source, target] = physical[target, source] = 1.0
        for source, target in self.semantic_edges:
            semantic[source, target] = semantic[target, source] = 1.0
        return EncodedGraph(
            node_features=node_features,
            physical_adjacency=physical,
            semantic_adjacency=semantic,
            node_mask=node_mask,
            global_features=np.asarray(global_features, dtype=np.float32),
            task_id=task_id,
        )


def _base_graph(task: str, env: HarnessEnv, observation: dict) -> tuple[_GraphBuilder, np.ndarray]:
    progress = env.step_count / max(env.max_steps, 1)
    builder = _GraphBuilder(progress, env.difficulty)
    positions = np.asarray(observation["cable_positions"], dtype=np.float64)
    edges = np.asarray(observation["cable_edges"], dtype=np.int64)
    degree = np.zeros(len(positions), dtype=np.int64)
    for source, target in edges:
        degree[int(source)] += 1
        degree[int(target)] += 1
    for index, position in enumerate(positions):
        roles = [ROLE_CABLE]
        if degree[index] == 1:
            roles.append(ROLE_ENDPOINT)
        if degree[index] >= 3:
            roles.append(ROLE_JUNCTION)
        builder.add_node(position, roles=tuple(roles), degree=min(degree[index] / 3.0, 1.0))
    for source, target in edges:
        builder.physical(int(source), int(target))

    global_features = np.zeros(GLOBAL_DIM, dtype=np.float32)
    global_features[TASK_TO_ID[task]] = 1.0
    global_features[4] = progress
    global_features[5] = env.difficulty
    return builder, global_features


def _nearest_node(positions: np.ndarray, point: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(positions - point, axis=1)))


def encode_environment(task: str, env: HarnessEnv) -> EncodedGraph:
    """Encode only observable state plus task/difficulty/progress metadata."""

    if task not in TASK_TO_ID:
        raise KeyError(task)
    observation = env.observation()
    positions = np.asarray(observation["cable_positions"], dtype=np.float64)
    builder, global_features = _base_graph(task, env, observation)

    if task == "inspect":
        inspected = {int(index) for index in observation["inspected"]}
        detections = {int(index) for index in observation["detections"]}
        attempts = np.asarray(observation.get("scan_attempts", np.zeros(7)), dtype=float)
        scores = np.asarray(observation.get("site_scores", np.full(7, 0.5)), dtype=float)
        site_nodes = []
        sites = np.asarray(observation["sites"], dtype=float)
        denominator = max(len(sites) - 1, 1)
        for index, site in enumerate(sites):
            node = builder.add_node(
                site,
                roles=(ROLE_SITE, ROLE_TARGET),
                order=index / denominator,
                inspected=float(index in inspected),
                detected=float(index in detections),
                active=min(float(attempts[index]) / 4.0, 1.0),
                confidence=float(scores[index]),
            )
            site_nodes.append(node)
            builder.semantic(node, _nearest_node(positions, site))
        camera_node = builder.add_node(observation["camera"], roles=(ROLE_ARM,))
        for site_node in site_nodes:
            builder.semantic(camera_node, site_node)
        global_features[8] = len(inspected) / max(len(sites), 1)
        global_features[9] = len(detections) / max(len(sites), 1)
        global_features[10] = float(scores.mean())

    elif task == "insert":
        endpoint = len(positions) - 1
        builder.nodes[endpoint][SEMANTIC_A] = 1.0
        socket = np.asarray(observation["socket"], dtype=float)
        axis = np.asarray(observation["socket_axis"], dtype=float)
        preinsert = socket - axis * (0.13 + 0.035 * env.difficulty)
        socket_node = builder.add_node(
            socket, roles=(ROLE_TARGET, ROLE_CONNECTOR), semantic_a=1.0, order=1.0
        )
        preinsert_node = builder.add_node(
            preinsert, roles=(ROLE_TARGET,), semantic_a=1.0, order=0.5
        )
        arm_node = builder.add_node(observation["arm_ee"], roles=(ROLE_ARM,), semantic_a=1.0)
        builder.semantic(endpoint, socket_node)
        builder.semantic(endpoint, preinsert_node)
        builder.semantic(endpoint, arm_node)
        global_features[6:8] = axis
        global_features[8] = float(observation["terminal_angle"]) / np.pi
        global_features[9] = float(np.linalg.norm(positions[endpoint] - socket))

    elif task == "route":
        target_indices = np.asarray(observation["target_particle_indices"], dtype=np.int64)
        bindings = {int(key): int(value) for key, value in observation["bindings"].items()}
        clips = np.asarray(observation["clips"], dtype=float)
        for clip_index, (clip, particle) in enumerate(zip(clips, target_indices)):
            order = (clip_index + 1) / max(len(clips), 1)
            clip_node = builder.add_node(
                clip,
                roles=(ROLE_CLIP, ROLE_TARGET),
                semantic_a=1.0,
                order=order,
                active=float(clip_index in bindings),
            )
            builder.semantic(int(particle), clip_node)
        finish_node = builder.add_node(
            observation["finish"], roles=(ROLE_TARGET,), order=1.0, semantic_b=1.0
        )
        builder.semantic(len(positions) - 1, finish_node)
        grasp_idx = observation["grasp_idx"]
        arm_node = builder.add_node(
            observation["arm_ee"], roles=(ROLE_ARM,), active=float(grasp_idx is not None)
        )
        builder.semantic(
            arm_node,
            int(grasp_idx)
            if grasp_idx is not None
            else _nearest_node(positions, np.asarray(observation["arm_ee"])),
        )
        global_features[8] = len(bindings) / max(len(clips), 1)
        global_features[9] = float(grasp_idx is not None)
        global_features[10] = float(observation["endpoint_placed"])

    elif task == "branch":
        endpoints = observation["semantic_endpoints"]
        endpoint_a, endpoint_b = int(endpoints["A"]), int(endpoints["B"])
        builder.nodes[endpoint_a][SEMANTIC_A] = 1.0
        builder.nodes[endpoint_b][SEMANTIC_B] = 1.0
        semantic_waypoints = observation["semantic_waypoints"]
        waypoint_nodes_a = [
            builder.add_node(
                waypoint,
                roles=(ROLE_TARGET,),
                semantic_a=1.0,
                order=boundary,
            )
            for waypoint, boundary in zip(semantic_waypoints["A"], BRANCH_PHASE_BOUNDARIES)
        ]
        waypoint_nodes_b = [
            builder.add_node(
                waypoint,
                roles=(ROLE_TARGET,),
                semantic_b=1.0,
                order=boundary,
            )
            for waypoint, boundary in zip(semantic_waypoints["B"], BRANCH_PHASE_BOUNDARIES)
        ]
        target_a = builder.add_node(
            observation["semantic_targets"]["A"],
            roles=(ROLE_TARGET,),
            semantic_a=1.0,
            order=1.0,
        )
        target_b = builder.add_node(
            observation["semantic_targets"]["B"],
            roles=(ROLE_TARGET,),
            semantic_b=1.0,
            order=1.0,
        )
        arm_a = builder.add_node(observation["arm_a_ee"], roles=(ROLE_ARM,), semantic_a=1.0)
        arm_b = builder.add_node(observation["arm_b_ee"], roles=(ROLE_ARM,), semantic_b=1.0)
        builder.semantic(endpoint_a, target_a)
        builder.semantic(endpoint_b, target_b)
        for waypoint_node in waypoint_nodes_a:
            builder.semantic(endpoint_a, waypoint_node)
        for waypoint_node in waypoint_nodes_b:
            builder.semantic(endpoint_b, waypoint_node)
        builder.semantic(endpoint_a, arm_a)
        builder.semantic(endpoint_b, arm_b)
        endpoint_positions = positions[[endpoint_a, endpoint_b]]
        target_positions = np.asarray(
            [
                observation["semantic_targets"]["A"],
                observation["semantic_targets"]["B"],
            ]
        )
        distance_matrix = np.linalg.norm(
            endpoint_positions[:, None, :] - target_positions[None, :, :],
            axis=2,
        )
        nearest_distances = distance_matrix.min(axis=1)
        # Permutation-invariant geometry progress. Semantic A/B assignment is
        # represented only by typed nodes and edges, never leaked globally.
        global_features[8] = float(nearest_distances.mean())
        global_features[9] = float(nearest_distances.max())
        global_features[10] = min(float(observation["crossings"]) / 4.0, 1.0)

    return builder.finish(global_features, TASK_TO_ID[task])


def normalize_action(task: str, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    output = np.zeros(MAX_ACTION_DIM, dtype=np.float32)
    output[: ACTION_DIMS[task]] = action
    if task == "insert":
        output[2] = np.clip((output[2] / np.pi + 1.0) * 0.5, 0.0, 1.0)
    return np.clip(output, 0.0, 1.0)


def denormalize_action(task: str, action: np.ndarray) -> np.ndarray:
    output = np.asarray(action, dtype=np.float64).copy()[: ACTION_DIMS[task]]
    if task == "insert":
        output[2] = (output[2] * 2.0 - 1.0) * np.pi
    return output


def action_mask(task: str) -> np.ndarray:
    mask = np.zeros(MAX_ACTION_DIM, dtype=np.float32)
    mask[: ACTION_DIMS[task]] = 1.0
    return mask


def binary_action_mask(task: str) -> np.ndarray:
    mask = np.zeros(MAX_ACTION_DIM, dtype=np.float32)
    for index in {
        "inspect": (2,),
        "insert": (3,),
        "route": (2,),
        "branch": (2, 5),
    }[task]:
        mask[index] = 1.0
    return mask
