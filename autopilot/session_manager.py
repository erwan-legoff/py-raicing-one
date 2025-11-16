"""Session management for concurrent simulator clients."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from .storage import RolloutBuffer


@dataclass
class Session:
    client_id: str
    rollout_buffer: RolloutBuffer = field(default_factory=RolloutBuffer)
    last_obs: Optional[torch.Tensor] = None
    last_payload: Optional[dict] = None
    last_action_idx: Optional[int] = None
    last_action_mask: Optional[list[int]] = None
    last_log_prob: float = 0.0
    last_value: float = 0.0
    last_reward: float = 0.0
    last_tags: list[str] = field(default_factory=list)
    episode_steps: int = 0
    episode_return: float = 0.0
    completed_episodes: int = 0
    evaluation_mode: bool = False
    low_speed_steps: int = 0
    stagnation_steps: int = 0
    forced_reset_streak: int = 0
    reverse_steps: int = 0
    time_in_warning_left: float = 0.0
    time_in_warning_right: float = 0.0
    time_in_danger_left: float = 0.0
    time_in_danger_right: float = 0.0
    time_in_slow: float = 0.0
    center_drift_integral: float = 0.0

    def reset_episode(self) -> None:
        self.rollout_buffer.clear()
        self.episode_steps = 0
        self.episode_return = 0.0

    def clear_last_transition(self) -> None:
        self.last_obs = None
        self.last_payload = None
        self.last_action_idx = None
        self.last_action_mask = None
        self.last_log_prob = 0.0
        self.last_value = 0.0

    def reset_stall_counters(self) -> None:
        self.low_speed_steps = 0
        self.stagnation_steps = 0
        self.reverse_steps = 0

    def register_forced_reset(self) -> None:
        self.forced_reset_streak += 1

    def clear_forced_reset_streak(self) -> None:
        self.forced_reset_streak = 0

    def decay_forced_reset_streak(self) -> None:
        if self.forced_reset_streak > 0:
            self.forced_reset_streak -= 1

    def reset_reward_counters(self) -> None:
        self.time_in_warning_left = 0.0
        self.time_in_warning_right = 0.0
        self.time_in_danger_left = 0.0
        self.time_in_danger_right = 0.0
        self.time_in_slow = 0.0
        self.center_drift_integral = 0.0


class SessionManager:
    """Registry of active WebSocket sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def get(self, client_id: str) -> Session:
        if client_id not in self._sessions:
            self._sessions[client_id] = Session(client_id=client_id)
        return self._sessions[client_id]

    def remove(self, client_id: str) -> None:
        self._sessions.pop(client_id, None)
