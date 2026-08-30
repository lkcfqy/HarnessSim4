"""Shared graph policy with topology/no-topology ablation support."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from harnessbench.learning.graph import (
    FEATURE_DIM,
    GLOBAL_DIM,
    MAX_ACTION_DIM,
    TOPOLOGY_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 96
    message_passing_layers: int = 3
    dropout: float = 0.05
    use_topology: bool = True
    adjacency_mode: str = "combined"
    use_topology_features: bool = True


class MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.message_linear = nn.Linear(hidden_dim, hidden_dim)
        self.task_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        task_context: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        messages = torch.bmm(adjacency, hidden) / degree
        update = torch.nn.functional.gelu(
            self.self_linear(hidden)
            + self.message_linear(messages)
            + self.task_linear(task_context).unsqueeze(1)
        )
        hidden = self.norm(hidden + self.dropout(update))
        return hidden * node_mask.unsqueeze(-1)


class TopoHarnessNet(nn.Module):
    """One graph encoder shared by all tasks with four lightweight action heads."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.adjacency_mode not in {"combined", "physical", "semantic", "none"}:
            raise ValueError(f"unknown adjacency_mode={self.config.adjacency_mode!r}")
        hidden = self.config.hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.task_embedding = nn.Embedding(4, hidden)
        self.layers = nn.ModuleList(
            [
                MessageLayer(hidden, self.config.dropout)
                for _ in range(self.config.message_passing_layers)
            ]
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 3, hidden),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(hidden, MAX_ACTION_DIM),
                )
                for _ in range(4)
            ]
        )
        self.pointer_query = nn.Linear(hidden * 3, hidden * 2)

    def forward(
        self,
        node_features: torch.Tensor,
        physical_adjacency: torch.Tensor,
        semantic_adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        global_features: torch.Tensor,
        task_id: torch.Tensor,
        *,
        use_topology: bool | None = None,
    ) -> torch.Tensor:
        action_logits, _ = self.forward_with_pointers(
            node_features,
            physical_adjacency,
            semantic_adjacency,
            node_mask,
            global_features,
            task_id,
            use_topology=use_topology,
        )
        return action_logits

    def forward_with_pointers(
        self,
        node_features: torch.Tensor,
        physical_adjacency: torch.Tensor,
        semantic_adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        global_features: torch.Tensor,
        task_id: torch.Tensor,
        *,
        use_topology: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topology = self.config.use_topology if use_topology is None else use_topology
        if not topology or not self.config.use_topology_features:
            node_features = node_features.clone()
            node_features[:, :, list(TOPOLOGY_FEATURE_COLUMNS)] = 0.0
        if not topology:
            adjacency = torch.zeros_like(physical_adjacency)
        elif self.config.adjacency_mode == "combined":
            adjacency = torch.clamp(physical_adjacency + semantic_adjacency, 0.0, 1.0)
        elif self.config.adjacency_mode == "physical":
            adjacency = physical_adjacency
        elif self.config.adjacency_mode == "semantic":
            adjacency = semantic_adjacency
        else:
            adjacency = torch.zeros_like(physical_adjacency)

        # Always keep a self-loop for valid nodes. The geometry ablation then
        # becomes a permutation-invariant DeepSets-style encoder of positions
        # and generic object state with exactly the same parameter count.
        identity = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0)
        adjacency = torch.clamp(adjacency + identity * node_mask.unsqueeze(1), 0.0, 1.0)
        hidden = self.node_encoder(node_features) * node_mask.unsqueeze(-1)
        task_context = self.task_embedding(task_id)
        for layer in self.layers:
            hidden = layer(hidden, adjacency, task_context, node_mask)

        attention_logits = self.attention(hidden).squeeze(-1)
        attention_logits = attention_logits.masked_fill(node_mask <= 0.0, -1e9)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        pooled = torch.sum(hidden * attention_weights.unsqueeze(-1), dim=1)
        global_hidden = self.global_encoder(global_features)
        context = torch.cat((pooled, global_hidden, task_context), dim=-1)

        output = torch.zeros(
            (node_features.shape[0], MAX_ACTION_DIM),
            device=node_features.device,
            dtype=node_features.dtype,
        )
        for index, head in enumerate(self.heads):
            selected = task_id == index
            if torch.any(selected):
                output[selected] = head(context[selected])
        queries = self.pointer_query(context).reshape(
            node_features.shape[0], 2, self.config.hidden_dim
        )
        pointer_logits = torch.einsum("bkh,bnh->bkn", queries, hidden) / math.sqrt(
            self.config.hidden_dim
        )
        pointer_logits = pointer_logits.masked_fill(node_mask.unsqueeze(1) <= 0.0, -1e9)
        return output, pointer_logits

    def model_metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "config": asdict(self.config),
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }
