"""Neural network architectures and serialization helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """Feed-forward network with policy and value heads."""

    def __init__(self, input_size: int = 28, hidden_size: int = 256, num_actions: int = 4) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.act1 = nn.ReLU()

        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.act2 = nn.ReLU()

        reduced = hidden_size // 2
        self.hidden_layer_2 = nn.Linear(hidden_size, reduced)
        self.layer_norm3 = nn.LayerNorm(reduced)
        self.act3 = nn.ReLU()

        self.policy_head = nn.Linear(reduced, num_actions)
        self.value_head = nn.Linear(reduced, 1)

        self.training_count: int = 0
        self.last_training_at: datetime = datetime.now(timezone.utc)
        self.to(DEVICE)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(inputs)
        x = self.layer_norm1(x)
        x = self.act1(x)

        x = self.hidden_layer(x)
        x = self.layer_norm2(x)
        x = self.act2(x)

        x = self.hidden_layer_2(x)
        x = self.layer_norm3(x)
        x = self.act3(x)

        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        state = super().state_dict(*args, **kwargs)
        state["_training_count"] = self.training_count
        state["_last_training_at"] = self.last_training_at.isoformat()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        self.training_count = state_dict.pop("_training_count", 0)
        last_training_raw = state_dict.pop("_last_training_at", None)
        if isinstance(last_training_raw, (int, float)):
            self.last_training_at = datetime.fromtimestamp(last_training_raw, tz=timezone.utc)
        elif isinstance(last_training_raw, str):
            try:
                parsed = datetime.fromisoformat(last_training_raw)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                self.last_training_at = parsed
            except ValueError:
                self.last_training_at = datetime.now(timezone.utc)
        else:
            self.last_training_at = datetime.now(timezone.utc)
        super().load_state_dict(state_dict, strict)


@dataclass
class ModelPaths:
    startup_path: Optional[str] = None
    checkpoint_dir: str = "saved_models"


def load_model(model: ActorCritic, paths: ModelPaths) -> None:
    """Load weights from disk if available."""

    if paths.startup_path and os.path.exists(paths.startup_path):
        state = torch.load(paths.startup_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        return

    if not os.path.exists(paths.checkpoint_dir):
        return

    checkpoints = [
        os.path.join(paths.checkpoint_dir, fname)
        for fname in os.listdir(paths.checkpoint_dir)
        if fname.endswith(".pt")
    ]
    if not checkpoints:
        return

    checkpoints.sort(key=os.path.getmtime, reverse=True)
    state = torch.load(checkpoints[0], map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)


def save_model(model: ActorCritic, paths: ModelPaths) -> str:
    """Persist the actor-critic weights on disk."""

    os.makedirs(paths.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(paths.checkpoint_dir, f"autopilot_{timestamp}.pt")
    torch.save(model.state_dict(), path)
    return path
