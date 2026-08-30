"""State-based ACT-style action-chunking baseline for HarnessSim4.

This is an independent, compact implementation of the central ACT recipe:
a conditional variational encoder, Transformer action queries, action chunks,
and temporal ensembling at inference. It consumes the same observable typed
node features as TopoHarnessNet but deliberately does not use graph edges or
message passing. It is not code from, nor an exact reproduction of, the
official ACT repository.

Reference: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
Low-Cost Hardware", RSS 2023, arXiv:2304.13705.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from harnessbench.learning.dataset import SPLIT_TO_ID, load_expert_dataset
from harnessbench.learning.graph import (
    FEATURE_DIM,
    GLOBAL_DIM,
    MAX_ACTION_DIM,
    MAX_NODES,
    ROLE_CABLE,
    ROLE_CLIP,
    ROLE_TARGET,
    SEMANTIC_A,
    SEMANTIC_B,
    STATE_ACTIVE,
    TASK_ORDER,
    TASK_TO_ID,
    binary_action_mask,
    denormalize_action,
    encode_environment,
)
from harnessbench.sim.envs.base import HarnessEnv

ACT_PAPER_URL = "https://arxiv.org/abs/2304.13705"
ACT_CODE_URL = "https://github.com/tonyzhaozh/act"


@dataclass(frozen=True)
class ActChunkConfig:
    hidden_dim: int = 96
    num_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    action_encoder_layers: int = 2
    chunk_size: int = 16
    latent_dim: int = 16
    dropout: float = 0.05
    use_relational_pooling: bool = False


class ActionChunkDataset(Dataset):
    """Create future action chunks without crossing whole-episode boundaries."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        *,
        tasks: tuple[str, ...],
        chunk_size: int,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least one")
        unknown = set(tasks) - set(TASK_ORDER)
        if unknown:
            raise KeyError(f"unknown tasks: {sorted(unknown)}")
        self.arrays = arrays
        self.tasks = tasks
        self.chunk_size = int(chunk_size)
        task_ids = {TASK_TO_ID[task] for task in tasks}
        self.record_indices = np.asarray(
            [index for index, task_id in enumerate(arrays["task_id"]) if int(task_id) in task_ids],
            dtype=np.int64,
        )
        if len(self.record_indices) == 0:
            raise ValueError("selected tasks have no samples")

        grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for record_index in self.record_indices:
            key = (
                int(arrays["task_id"][record_index]),
                int(arrays["episode_seed"][record_index]),
                int(arrays["split"][record_index]),
            )
            grouped[key].append(int(record_index))
        for records in grouped.values():
            records.sort(key=lambda index: int(arrays["episode_step"][index]))

        self.future_indices = np.zeros((len(self.record_indices), self.chunk_size), dtype=np.int64)
        self.future_valid = np.zeros((len(self.record_indices), self.chunk_size), dtype=np.float32)
        local_by_record = {
            int(record_index): local_index
            for local_index, record_index in enumerate(self.record_indices)
        }
        for records in grouped.values():
            for position, record_index in enumerate(records):
                available = records[position : position + self.chunk_size]
                padded = available + [available[-1]] * (self.chunk_size - len(available))
                local_index = local_by_record[record_index]
                self.future_indices[local_index] = np.asarray(padded, dtype=np.int64)
                self.future_valid[local_index, : len(available)] = 1.0

        self.splits = arrays["split"][self.record_indices].astype(np.int64)
        self.task_ids = arrays["task_id"][self.record_indices].astype(np.int64)

    def __len__(self) -> int:
        return len(self.record_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index = int(self.record_indices[index])
        future = self.future_indices[index]
        return {
            "node_features": torch.from_numpy(self.arrays["node_features"][record_index]),
            "semantic_adjacency": torch.from_numpy(
                self.arrays["semantic_adjacency"][record_index]
            ),
            "node_mask": torch.from_numpy(self.arrays["node_mask"][record_index]),
            "global_features": torch.from_numpy(self.arrays["global_features"][record_index]),
            "task_id": torch.as_tensor(self.arrays["task_id"][record_index]),
            "action_mask": torch.from_numpy(self.arrays["action_mask"][record_index]),
            "binary_action_mask": torch.from_numpy(self.arrays["binary_action_mask"][record_index]),
            "action_chunk": torch.from_numpy(self.arrays["action"][future]),
            "pointer_chunk": torch.from_numpy(self.arrays["pointer_target"][future]),
            "pointer_mask_chunk": torch.from_numpy(self.arrays["pointer_mask"][future]),
            "chunk_valid": torch.from_numpy(self.future_valid[index]),
            "episode_seed": torch.as_tensor(self.arrays["episode_seed"][record_index]),
            "episode_step": torch.as_tensor(self.arrays["episode_step"][record_index]),
        }


class ActChunkNet(nn.Module):
    """CVAE Transformer that predicts a fixed-length normalized action chunk."""

    def __init__(self, config: ActChunkConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActChunkConfig()
        hidden = self.config.hidden_dim
        if hidden % self.config.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.node_encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.relation_encoder = (
            nn.Sequential(
                nn.Linear(FEATURE_DIM, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            if self.config.use_relational_pooling
            else None
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.task_embedding = nn.Embedding(len(TASK_ORDER), hidden)
        self.observation_positions = nn.Parameter(torch.randn(MAX_NODES + 1, hidden) * 0.02)
        observation_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.config.num_heads,
            dim_feedforward=hidden * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.observation_encoder = nn.TransformerEncoder(
            observation_layer,
            num_layers=self.config.encoder_layers,
            norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )

        self.action_encoder = nn.Linear(MAX_ACTION_DIM, hidden)
        self.action_positions = nn.Parameter(torch.randn(self.config.chunk_size, hidden) * 0.02)
        self.latent_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        latent_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.config.num_heads,
            dim_feedforward=hidden * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(
            latent_layer,
            num_layers=self.config.action_encoder_layers,
            norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )
        self.latent_stats = nn.Linear(hidden, self.config.latent_dim * 2)
        self.latent_projection = nn.Linear(self.config.latent_dim, hidden)

        self.action_queries = nn.Parameter(torch.randn(self.config.chunk_size, hidden) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=self.config.num_heads,
            dim_feedforward=hidden * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.config.decoder_layers,
            norm=nn.LayerNorm(hidden),
        )
        self.action_head = nn.Linear(hidden, MAX_ACTION_DIM)
        self.pointer_query = nn.Linear(hidden, hidden * 2)

    def _encode_observation(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        global_features: torch.Tensor,
        task_id: torch.Tensor,
        semantic_adjacency: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.global_encoder(global_features) + self.task_embedding(task_id)
        context_token = context.unsqueeze(1)
        node_tokens = self.node_encoder(node_features) + context_token
        if self.relation_encoder is not None:
            if semantic_adjacency is None:
                raise ValueError("semantic_adjacency is required for relational ACT")
            degree = semantic_adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            neighbor_features = torch.bmm(semantic_adjacency, node_features) / degree
            node_tokens = node_tokens + self.relation_encoder(neighbor_features)
        tokens = torch.cat((context_token, node_tokens), dim=1)
        tokens = tokens + self.observation_positions.unsqueeze(0)
        padding = torch.cat(
            (
                torch.zeros(
                    (node_mask.shape[0], 1),
                    dtype=torch.bool,
                    device=node_mask.device,
                ),
                node_mask <= 0.0,
            ),
            dim=1,
        )
        memory = self.observation_encoder(tokens, src_key_padding_mask=padding)
        return memory, padding, context

    def _encode_latent(
        self,
        action_chunk: torch.Tensor,
        chunk_valid: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = action_chunk.shape[0]
        latent_token = self.latent_token.expand(batch_size, -1, -1) + context.unsqueeze(1)
        action_tokens = (
            self.action_encoder(action_chunk)
            + self.action_positions.unsqueeze(0)
            + context.unsqueeze(1)
        )
        tokens = torch.cat((latent_token, action_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros(
                    (batch_size, 1),
                    dtype=torch.bool,
                    device=chunk_valid.device,
                ),
                chunk_valid <= 0.0,
            ),
            dim=1,
        )
        encoded = self.latent_encoder(tokens, src_key_padding_mask=padding)
        mu, log_variance = self.latent_stats(encoded[:, 0]).chunk(2, dim=-1)
        if self.training:
            standard_deviation = torch.exp(0.5 * log_variance)
            latent = mu + standard_deviation * torch.randn_like(standard_deviation)
        else:
            latent = mu
        return latent, mu, log_variance

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        global_features: torch.Tensor,
        task_id: torch.Tensor,
        *,
        semantic_adjacency: torch.Tensor | None = None,
        action_chunk: torch.Tensor | None = None,
        chunk_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, memory_padding, context = self._encode_observation(
            node_features,
            node_mask,
            global_features,
            task_id,
            semantic_adjacency,
        )
        if action_chunk is None:
            latent = torch.zeros(
                (node_features.shape[0], self.config.latent_dim),
                dtype=node_features.dtype,
                device=node_features.device,
            )
            mu = torch.zeros_like(latent)
            log_variance = torch.zeros_like(latent)
        else:
            if chunk_valid is None:
                chunk_valid = torch.ones(
                    action_chunk.shape[:2],
                    dtype=action_chunk.dtype,
                    device=action_chunk.device,
                )
            latent, mu, log_variance = self._encode_latent(
                action_chunk,
                chunk_valid,
                context,
            )
        queries = self.action_queries.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        queries = queries + context.unsqueeze(1) + self.latent_projection(latent).unsqueeze(1)
        decoded = self.decoder(
            queries,
            memory,
            memory_key_padding_mask=memory_padding,
        )
        pointer_queries = self.pointer_query(decoded).reshape(
            node_features.shape[0],
            self.config.chunk_size,
            2,
            self.config.hidden_dim,
        )
        pointer_logits = torch.einsum(
            "bksd,bnd->bksn",
            pointer_queries,
            memory[:, 1:],
        ) / np.sqrt(self.config.hidden_dim)
        pointer_logits = pointer_logits.masked_fill(
            node_mask[:, None, None, :] <= 0.0,
            -1e9,
        )
        return self.action_head(decoded), pointer_logits, mu, log_variance

    def model_metadata(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "config": asdict(self.config),
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "observation": (
                "typed node set + global state + fixed one-hop semantic neighbor pooling"
                if self.config.use_relational_pooling
                else "typed node set + global state; no adjacency/message passing"
            ),
            "reference": ACT_PAPER_URL,
        }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    integer_keys = {"task_id", "pointer_chunk", "episode_seed", "episode_step"}
    return {
        key: value.to(
            device=device,
            dtype=torch.long if key in integer_keys else torch.float32,
        )
        for key, value in batch.items()
    }


def _act_loss(
    logits: torch.Tensor,
    pointer_logits: torch.Tensor,
    mu: torch.Tensor,
    log_variance: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    kl_weight: float,
    pointer_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = torch.sigmoid(logits)
    action_mask = batch["action_mask"].unsqueeze(1)
    binary_mask = batch["binary_action_mask"].unsqueeze(1)
    continuous_mask = torch.clamp(action_mask - binary_mask, 0.0, 1.0)
    valid = batch["chunk_valid"].unsqueeze(-1)
    continuous_weight = valid * continuous_mask
    binary_weight = valid * binary_mask
    l1 = torch.sum(torch.abs(prediction - batch["action_chunk"]) * continuous_weight)
    l1 = l1 / continuous_weight.sum().clamp_min(1.0)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        batch["action_chunk"],
        reduction="none",
    )
    bce = torch.sum(bce * binary_weight) / binary_weight.sum().clamp_min(1.0)
    pointer_values = nn.functional.cross_entropy(
        pointer_logits.reshape(-1, pointer_logits.shape[-1]),
        batch["pointer_chunk"].reshape(-1),
        reduction="none",
    ).reshape(pointer_logits.shape[:3])
    pointer_weight = batch["chunk_valid"].unsqueeze(-1) * batch["pointer_mask_chunk"]
    pointer_loss = torch.sum(pointer_values * pointer_weight)
    pointer_loss = pointer_loss / pointer_weight.sum().clamp_min(1.0)
    kl = -0.5 * torch.mean(1.0 + log_variance - mu.square() - log_variance.exp())
    loss = l1 + bce + float(pointer_loss_weight) * pointer_loss + float(kl_weight) * kl
    return loss, {
        "continuous_l1": float(l1.detach()),
        "binary_bce": float(bce.detach()),
        "pointer_cross_entropy": float(pointer_loss.detach()),
        "kl": float(kl.detach()),
    }


@torch.no_grad()
def _evaluate_loader(
    model: ActChunkNet,
    loader: DataLoader,
    device: torch.device,
    *,
    kl_weight: float,
    pointer_loss_weight: float,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    absolute_error = 0.0
    active_dimensions = 0.0
    pointer_correct = 0.0
    pointer_events = 0.0
    samples = 0
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        logits, pointer_logits, mu, log_variance = model(
            batch["node_features"],
            batch["node_mask"],
            batch["global_features"],
            batch["task_id"],
            semantic_adjacency=batch["semantic_adjacency"],
            action_chunk=batch["action_chunk"],
            chunk_valid=batch["chunk_valid"],
        )
        loss, _ = _act_loss(
            logits,
            pointer_logits,
            mu,
            log_variance,
            batch,
            kl_weight=kl_weight,
            pointer_loss_weight=pointer_loss_weight,
        )
        mask = batch["chunk_valid"].unsqueeze(-1) * batch["action_mask"].unsqueeze(1)
        absolute_error += float(
            torch.sum(torch.abs(torch.sigmoid(logits) - batch["action_chunk"]) * mask)
        )
        active_dimensions += float(mask.sum())
        pointer_prediction = pointer_logits.argmax(dim=-1)
        pointer_weight = batch["chunk_valid"].unsqueeze(-1) * batch["pointer_mask_chunk"]
        pointer_correct += float(
            torch.sum((pointer_prediction == batch["pointer_chunk"]).float() * pointer_weight)
        )
        pointer_events += float(pointer_weight.sum())
        batch_size = len(batch["task_id"])
        loss_sum += float(loss) * batch_size
        samples += batch_size
    return {
        "samples": samples,
        "loss": loss_sum / max(samples, 1),
        "action_mae": absolute_error / max(active_dimensions, 1.0),
        "pointer_accuracy": pointer_correct / max(pointer_events, 1.0),
    }


@torch.no_grad()
def _calibrate_thresholds(
    model: ActChunkNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[list[float]], dict[str, dict[str, float | int]]]:
    probabilities: dict[tuple[int, int], list[float]] = defaultdict(list)
    targets: dict[tuple[int, int], list[int]] = defaultdict(list)
    model.eval()
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        logits, _, _, _ = model(
            batch["node_features"],
            batch["node_mask"],
            batch["global_features"],
            batch["task_id"],
            semantic_adjacency=batch["semantic_adjacency"],
        )
        prediction = torch.sigmoid(logits[:, 0]).cpu().numpy()
        labels = batch["action_chunk"][:, 0].cpu().numpy()
        masks = batch["binary_action_mask"].cpu().numpy()
        task_ids = batch["task_id"].cpu().numpy()
        for row, task_id in enumerate(task_ids):
            for dimension in np.flatnonzero(masks[row] > 0.0):
                key = (int(task_id), int(dimension))
                probabilities[key].append(float(prediction[row, dimension]))
                targets[key].append(int(labels[row, dimension] >= 0.5))

    thresholds = np.full((len(TASK_ORDER), MAX_ACTION_DIM), 0.5, dtype=np.float32)
    report: dict[str, dict[str, float | int]] = {}
    for key, values_list in probabilities.items():
        values = np.asarray(values_list, dtype=np.float64)
        labels = np.asarray(targets[key], dtype=np.int64)
        best_threshold = 0.5
        best_score = -1.0
        if labels.min() != labels.max():
            for threshold in np.linspace(0.05, 0.95, 181):
                predicted = values >= threshold
                positive_recall = float(np.mean(predicted[labels == 1]))
                negative_recall = float(np.mean(~predicted[labels == 0]))
                score = 0.5 * (positive_recall + negative_recall)
                if score > best_score + 1e-12:
                    best_score = score
                    best_threshold = float(threshold)
        thresholds[key] = best_threshold
        report[f"task_{key[0]}_dim_{key[1]}"] = {
            "threshold": best_threshold,
            "samples": len(values),
            "positive_rate": float(labels.mean()),
            "balanced_accuracy": best_score if best_score >= 0.0 else 1.0,
        }
    return thresholds.tolist(), report


def train_act_chunk_policy(
    dataset_path: Path,
    output_path: Path,
    *,
    tasks: tuple[str, ...] = TASK_ORDER,
    seed: int = 202_610,
    epochs: int = 120,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 96,
    num_heads: int = 4,
    layers: int = 2,
    chunk_size: int = 16,
    latent_dim: int = 16,
    kl_weight: float = 0.1,
    pointer_loss_weight: float = 1.0,
    use_relations: bool = False,
    patience: int = 18,
    device_name: str = "cpu",
) -> dict[str, Any]:
    arrays = load_expert_dataset(dataset_path)
    dataset_metadata_path = dataset_path.with_suffix(".json")
    dataset_metadata = (
        json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
        if dataset_metadata_path.is_file()
        else {}
    )
    control_stride = int(dataset_metadata.get("record_every", 1))
    if control_stride < 1:
        raise ValueError("dataset record_every/control stride must be at least one")
    dataset = ActionChunkDataset(arrays, tasks=tasks, chunk_size=chunk_size)
    indices = {
        name: np.flatnonzero(dataset.splits == split_id).tolist()
        for name, split_id in SPLIT_TO_ID.items()
    }
    if any(not values for values in indices.values()):
        raise ValueError("dataset is missing at least one selected-task split")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    generator = torch.Generator().manual_seed(seed)
    train_task_ids = dataset.task_ids[indices["train"]]
    task_counts = np.bincount(train_task_ids, minlength=len(TASK_ORDER))
    sample_weights = np.asarray(
        [1.0 / max(task_counts[task_id], 1) for task_id in train_task_ids],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights),
        num_samples=len(indices["train"]),
        replacement=True,
        generator=generator,
    )
    loaders = {
        "train": DataLoader(
            Subset(dataset, indices["train"]),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        ),
        "validation": DataLoader(
            Subset(dataset, indices["validation"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            Subset(dataset, indices["test"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
    config = ActChunkConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        encoder_layers=layers,
        decoder_layers=layers,
        action_encoder_layers=layers,
        chunk_size=chunk_size,
        latent_dim=latent_dim,
        use_relational_pooling=use_relations,
    )
    model = ActChunkNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_validation = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        samples = 0
        last_components: dict[str, float] = {}
        for raw_batch in loaders["train"]:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, pointer_logits, mu, log_variance = model(
                batch["node_features"],
                batch["node_mask"],
                batch["global_features"],
                batch["task_id"],
                semantic_adjacency=batch["semantic_adjacency"],
                action_chunk=batch["action_chunk"],
                chunk_valid=batch["chunk_valid"],
            )
            loss, last_components = _act_loss(
                logits,
                pointer_logits,
                mu,
                log_variance,
                batch,
                kl_weight=kl_weight,
                pointer_loss_weight=pointer_loss_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_samples = len(batch["task_id"])
            loss_sum += float(loss.detach()) * batch_samples
            samples += batch_samples
        validation = _evaluate_loader(
            model,
            loaders["validation"],
            device,
            kl_weight=kl_weight,
            pointer_loss_weight=pointer_loss_weight,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(samples, 1),
                "validation_loss": float(validation["loss"]),
                **last_components,
            }
        )
        if float(validation["loss"]) < best_validation - 1e-6:
            best_validation = float(validation["loss"])
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("ACT training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics = _evaluate_loader(
        model,
        loaders["validation"],
        device,
        kl_weight=kl_weight,
        pointer_loss_weight=pointer_loss_weight,
    )
    test_metrics = _evaluate_loader(
        model,
        loaders["test"],
        device,
        kl_weight=kl_weight,
        pointer_loss_weight=pointer_loss_weight,
    )
    binary_thresholds, calibration = _calibrate_thresholds(
        model,
        loaders["validation"],
        device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "0.1",
        "baseline": (
            "ACT-style state action chunking with fixed semantic-relation pooling"
            if use_relations
            else "ACT-style state action chunking"
        ),
        "model_config": asdict(config),
        "state_dict": best_state,
        "tasks": list(tasks),
        "dataset_sha256": _file_hash(dataset_path),
        "seed": seed,
        "best_epoch": best_epoch,
        "binary_thresholds": binary_thresholds,
        "temporal_ensemble_decay": 0.05,
        "control_stride": control_stride,
    }
    torch.save(checkpoint, output_path)
    report = {
        "name": "HarnessSim4 ACT-style action-chunking baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(output_path.resolve()),
        "model_metadata": model.model_metadata(),
        "tasks": list(tasks),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": checkpoint["dataset_sha256"],
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
        "kl_weight": kl_weight,
        "pointer_loss_weight": pointer_loss_weight,
        "use_relations": use_relations,
        "control_stride": control_stride,
        "binary_threshold_calibration": calibration,
        "split_samples": {name: len(values) for name, values in indices.items()},
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "reference": {
            "paper": ACT_PAPER_URL,
            "official_code": ACT_CODE_URL,
            "implementation_scope": (
                "independent state-based adaptation; no images, joint proprioception, "
                "or upstream ACT source code"
            ),
        },
        "claim_scope": "offline imitation metrics only; closed-loop evaluation required",
        "torch_version": torch.__version__,
        "device": str(device),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": str(output_path.resolve()),
        "report": str(report_path.resolve()),
        "parameters": report["model_metadata"]["parameters"],
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
    }


def load_act_chunk_model(
    path: Path,
    *,
    device_name: str = "cpu",
) -> tuple[ActChunkNet, dict[str, Any]]:
    device = torch.device(device_name)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch before weights_only was added
        checkpoint = torch.load(path, map_location=device)
    model = ActChunkNet(ActChunkConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


class ActChunkPolicy:
    """Closed-loop temporal-ensemble adapter for an ACT-style checkpoint."""

    def __init__(
        self,
        checkpoint: Path,
        task: str,
        *,
        name: str = "act_chunk",
        device_name: str = "cpu",
        typed_grounding: bool = False,
    ) -> None:
        self.model, metadata = load_act_chunk_model(checkpoint, device_name=device_name)
        if task not in metadata["tasks"]:
            raise ValueError(f"checkpoint was not trained for {task!r}")
        self.task = task
        self.name = name
        self.typed_grounding = bool(typed_grounding)
        self.device = next(self.model.parameters()).device
        self.thresholds = np.asarray(metadata["binary_thresholds"][TASK_TO_ID[task]])
        self.temporal_decay = float(metadata.get("temporal_ensemble_decay", 0.05))
        # Historical HarnessSim4 ACT checkpoints were all collected with
        # record_every=2 before this field became explicit.
        self.control_stride = int(metadata.get("control_stride", 2))
        self.predictions: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.time_index = 0
        self.environment_step_index = 0
        self.seed = 0
        self.last_debug: dict[str, Any] = {}

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.predictions = []
        self.time_index = 0
        self.environment_step_index = 0
        self.last_debug = {}

    def _ground_route_action(
        self,
        env: HarnessEnv,
        graph,
        normalized: np.ndarray,
        *,
        pointer_scores: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation = env.observation()
        features = graph.node_features
        candidates = graph.node_mask > 0.0
        if observation["grasp_idx"] is None:
            candidates &= features[:, ROLE_CABLE] > 0.5
            for particle in observation["bindings"].values():
                candidates[int(particle)] = False
            phase = "typed_approach_cable"
        else:
            target = features[:, ROLE_TARGET] > 0.5
            cable = features[:, ROLE_CABLE] > 0.5
            active_clip = (features[:, ROLE_CLIP] > 0.5) & (features[:, STATE_ACTIVE] > 0.5)
            candidates &= target & ~cable & ~active_clip
            phase = "typed_transport_to_fixture"
        if not np.any(candidates):
            candidates = graph.node_mask > 0.0
        candidate_indices = np.flatnonzero(candidates)
        masked_scores = np.where(candidates, pointer_scores, -np.inf)
        pointer_index = int(np.argmax(masked_scores))
        selected_prediction_distance = float(
            np.linalg.norm(features[pointer_index, :2] - normalized[:2])
        )
        normalized[:2] = features[pointer_index, :2]
        if observation["grasp_idx"] is None:
            arm_distance = float(np.linalg.norm(np.asarray(observation["arm_ee"]) - normalized[:2]))
            normalized[2] = float(arm_distance <= 0.032)
        else:
            arm_distance = float(np.linalg.norm(np.asarray(observation["arm_ee"]) - normalized[:2]))
            normalized[2] = 1.0
        return normalized, {
            "phase": phase,
            "pointer_index": pointer_index,
            "candidate_count": len(candidate_indices),
            "prediction_to_pointer_distance": selected_prediction_distance,
            "pointer_margin": float(
                np.max(masked_scores) - np.partition(masked_scores[candidates], -2)[-2]
                if len(candidate_indices) > 1
                else 0.0
            ),
            "arm_pointer_distance": arm_distance,
        }

    def _ground_branch_action(
        self,
        graph,
        normalized: np.ndarray,
        *,
        pointer_scores: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Ground the two ACT pointer slots on observable A/B target nodes.

        BranchBot exposes two semantic arms and two semantic endpoint-to-target
        relations.  The grounding layer only enforces the same observable type
        constraints used by the graph policy; it does not choose a waypoint or
        inject a scripted phase.  The ACT pointer scores still determine which
        A/B target or waypoint is selected at every control step.
        """

        features = graph.node_features
        valid = graph.node_mask > 0.0
        target_nodes = valid & (features[:, ROLE_TARGET] > 0.5)
        pointer_indices: list[int] = []
        candidate_counts: list[int] = []
        pointer_margins: list[float] = []
        for pointer_slot, semantic_column in enumerate((SEMANTIC_A, SEMANTIC_B)):
            candidates = target_nodes & (features[:, semantic_column] > 0.5)
            if not np.any(candidates):
                candidates = target_nodes if np.any(target_nodes) else valid
            scores = np.where(candidates, pointer_scores[pointer_slot], -np.inf)
            candidate_indices = np.flatnonzero(candidates)
            pointer_index = int(np.argmax(scores))
            pointer_indices.append(pointer_index)
            candidate_counts.append(len(candidate_indices))
            pointer_margins.append(
                float(
                    np.max(scores)
                    - np.partition(scores[candidates], -2)[-2]
                    if len(candidate_indices) > 1
                    else 0.0
                )
            )

        normalized[:2] = features[pointer_indices[0], :2]
        normalized[3:5] = features[pointer_indices[1], :2]
        normalized[2] = normalized[5] = 1.0
        return normalized, {
            "phase": "typed_dual_arm_pointer",
            "pointer_indices": pointer_indices,
            "candidate_counts": candidate_counts,
            "pointer_margins": pointer_margins,
        }

    @torch.no_grad()
    def act(self, env: HarnessEnv) -> np.ndarray:
        graph = encode_environment(self.task, env)

        def tensor(value: np.ndarray, *, integer: bool = False) -> torch.Tensor:
            return torch.as_tensor(
                value,
                dtype=torch.long if integer else torch.float32,
                device=self.device,
            ).unsqueeze(0)

        if self.environment_step_index % self.control_stride == 0:
            logits, pointer_logits, _, _ = self.model(
                tensor(graph.node_features),
                tensor(graph.node_mask),
                tensor(graph.global_features),
                tensor(np.asarray(graph.task_id), integer=True),
                semantic_adjacency=tensor(graph.semantic_adjacency),
            )
            new_chunk = torch.sigmoid(logits[0]).cpu().numpy()
            new_pointer_chunk = pointer_logits[0].cpu().numpy()
            self.predictions.append((self.time_index, new_chunk, new_pointer_chunk))
        self.predictions = [
            item
            for item in self.predictions
            if self.time_index - item[0] < self.model.config.chunk_size
        ]
        candidates: list[np.ndarray] = []
        pointer_candidates: list[np.ndarray] = []
        weights: list[float] = []
        for start, chunk, pointer_chunk in self.predictions:
            offset = self.time_index - start
            if 0 <= offset < len(chunk):
                candidates.append(chunk[offset])
                pointer_candidates.append(pointer_chunk[offset])
                weights.append(float(np.exp(-self.temporal_decay * offset)))
        normalized = np.average(
            np.asarray(candidates, dtype=np.float64),
            axis=0,
            weights=np.asarray(weights, dtype=np.float64),
        )
        for dimension in np.flatnonzero(binary_action_mask(self.task) > 0.0):
            normalized[dimension] = float(normalized[dimension] >= self.thresholds[dimension])
        grounding = None
        if self.typed_grounding:
            if self.task not in {"route", "branch"}:
                raise NotImplementedError(
                    "typed ACT grounding is currently validated for route and branch only"
                )
            pointer_scores = np.average(
                np.asarray(pointer_candidates, dtype=np.float64),
                axis=0,
                weights=np.asarray(weights, dtype=np.float64),
            )
            if self.task == "route":
                normalized, grounding = self._ground_route_action(
                    env,
                    graph,
                    normalized,
                    pointer_scores=pointer_scores[0],
                )
            else:
                normalized, grounding = self._ground_branch_action(
                    graph,
                    normalized,
                    pointer_scores=pointer_scores,
                )
        normalized = np.clip(normalized, 0.0, 1.0)
        action = denormalize_action(self.task, normalized)
        self.last_debug = {
            "time_index": self.time_index,
            "environment_step_index": self.environment_step_index,
            "control_stride": self.control_stride,
            "ensemble_predictions": len(candidates),
            "normalized_action": normalized.tolist(),
            "action": action.tolist(),
            "typed_grounding": grounding,
        }
        self.environment_step_index += 1
        if self.environment_step_index % self.control_stride == 0:
            self.time_index += 1
        return action
