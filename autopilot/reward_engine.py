"""Isolated reward computation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class RewardResult:
    reward: float
    tags: list[str]
    done: bool


class RewardEngine:
    """Domain-specific heuristics to score transitions."""

    def __init__(self) -> None:
        self.progress_scale = 5.0
        self.forward_bonus = 1.0
        self.center_bonus = 0.5
        self.side_penalty = -1.5
        self.offroad_penalty = -25.0
        self.crash_penalty = -100.0

    def compute_reward(
        self,
        prev_payload: dict,
        current_payload: dict,
        action_mask: Iterable[int],
        frame_idx: int,
        road_size: dict | None,
        car_size: dict | None,
    ) -> RewardResult:
        if not prev_payload or not current_payload:
            return RewardResult(0.0, ["BOOTSTRAP"], False)

        tags: list[str] = []
        done = False

        prev_car = (prev_payload.get("positions") or {}).get("car", {})
        curr_car = (current_payload.get("positions") or {}).get("car", {})
        prev_z = float(prev_car.get("z", 0.0))
        curr_z = float(curr_car.get("z", 0.0))
        curr_x = float(curr_car.get("x", 0.0))
        curr_y = float(curr_car.get("y", 0.0))

        road = road_size or {}
        car = car_size or {}
        lane_half = max((road.get("width", 10.0) - car.get("width", 1.0)) / 2.0, 0.5)
        center_zone = lane_half * 0.3
        danger_zone = lane_half * 0.85

        forward_progress = prev_z - curr_z  # axis inverted in game
        reward = forward_progress * self.progress_scale
        if forward_progress > 0:
            tags.append("FORWARD_PROGRESS")
        else:
            reward -= 1.0
            tags.append("BACKTRACKING")

        abs_x = abs(curr_x)
        if abs_x <= center_zone:
            reward += self.center_bonus
            tags.append("CENTERED")
        elif abs_x >= danger_zone:
            reward += self.side_penalty
            tags.append("DANGEROUS_SIDE")

        # Encourage a bit of steering when action indicates lateral move
        mask = list(action_mask)
        if mask and (mask[0] or mask[2]):
            reward += 0.1
            tags.append("LATERAL_CONTROL")

        # Penalize low speed
        curr_speed = (current_payload.get("speeds") or {}).get("z", 0.0)
        forward_speed = -float(curr_speed)
        if forward_speed < 0.1:
            reward -= 0.5
            tags.append("LOW_FORWARD_SPEED")
        else:
            reward += min(forward_speed, 3.0) * self.forward_bonus

        # Offroad / crash detection via vertical drop or x beyond lane
        if curr_y < (road.get("y", 0.0) - 0.5):
            reward += self.crash_penalty
            tags.append("FALLING_OFF_ROAD")
            done = True
        elif abs_x >= lane_half:
            reward += self.offroad_penalty * (abs_x / max(lane_half, 1e-3))
            tags.append("NEW_OFFROAD")
            done = True

        # Finish line detection
        road_depth = float(road.get("depth", 200.0))
        finish_line = -road_depth / 2.0 + 10.0
        if curr_z <= finish_line:
            reward += 100.0
            tags.append("FINISH_LINE")
            done = True

        reward = float(torch.clamp(torch.tensor(reward), -10.0, 10.0).item())
        return RewardResult(reward=reward, tags=tags, done=done)
