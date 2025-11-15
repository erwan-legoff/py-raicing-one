"""Transition/episode data structures and buffers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch


@dataclass
class Transition:
    obs: torch.Tensor
    action_idx: int
    action_mask: list[int]
    log_prob: float
    reward: float
    done: bool
    value: float
    tags: list[str]
    frame_id: int


@dataclass
class Episode:
    transitions: list[Transition]
    total_reward: float
    length: int


class RolloutBuffer:
    """Per-session buffer accumulating transitions until an episode ends."""

    def __init__(self) -> None:
        self._transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def clear(self) -> None:
        self._transitions.clear()

    def pop_episode(self) -> Episode:
        total_reward = sum(t.reward for t in self._transitions)
        episode = Episode(list(self._transitions), total_reward, len(self._transitions))
        self.clear()
        return episode

    def __len__(self) -> int:
        return len(self._transitions)


class ReplayBuffer:
    """Optional FIFO replay buffer (not heavily used yet)."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._storage: list[Transition] = []

    def extend(self, transitions: Iterable[Transition]) -> None:
        for t in transitions:
            self._storage.append(t)
            if len(self._storage) > self.capacity:
                self._storage.pop(0)

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size >= len(self._storage):
            return list(self._storage)
        indices = torch.randperm(len(self._storage))[:batch_size].tolist()
        return [self._storage[i] for i in indices]

    def __len__(self) -> int:
        return len(self._storage)
