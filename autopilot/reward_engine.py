"""Isolated reward computation utilities inspired by the original heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .session_manager import Session


@dataclass
class RewardResult:
    reward: float
    tags: list[str]
    done: bool


class RewardEngine:
    """Domain-specific heuristics to score transitions."""

    def __init__(self, warmup_frames: int = 200, physics_fps: float = 60.0) -> None:
        self.warmup_frames = warmup_frames
        self.physics_fps = max(physics_fps, 1.0)
        # Lateral thresholds
        self.z_warn = 0.30
        self.z_danger = 0.70
        self.warn_alpha = 0.15
        self.warn_beta = 0.6
        self.danger_alpha = 0.4
        self.danger_beta = 0.8
        self.k_warn = 2.0
        self.k_danger = 6.0
        self.lateral_penalty_cap = 25.0
        self.lateral_slow_horizon = 6.0
        self.lateral_fast_horizon = 2.0
        self.lateral_speed_fast_threshold = 0.8
        # Speed thresholds
        self.speed_warn = 0.40
        self.speed_danger = 0.15
        self.k_fast = 1.5
        self.fast_gamma = 0.4
        self.k_slow = 3.0
        self.slow_alpha = 0.25
        self.slow_beta = 0.4
        self.speed_penalty_cap = 20.0
        self.recovery_rate = 0.75
        self.slow_time_horizon = 6.0

    def compute_reward(
        self,
        prev_payload: dict,
        current_payload: dict,
        action_mask: Iterable[int],
        frame_idx: int,
        road_size: dict | None,
        car_size: dict | None,
        session: Session,
    ) -> RewardResult:
        if not prev_payload or not current_payload:
            return RewardResult(0.0, ["BOOTSTRAP"], False)

        tags: list[str] = []
        done = False

        prev_positions = prev_payload.get("positions") or {}
        prev_car = prev_positions.get("car") or {}
        curr_positions = current_payload.get("positions") or {}
        curr_car = curr_positions.get("car") or {}
        if not prev_car or not curr_car:
            return RewardResult(0.0, ["INVALID_POSITIONS"], False)

        current_frame = int(frame_idx or 0)
        if current_frame <= self.warmup_frames:
            return RewardResult(0.0, ["EPISODE_WARMUP"], False)

        prev_frame = int(prev_payload.get("frameId", current_frame - 1) or (current_frame - 1))
        frame_delta = max(1.0, float(current_frame - prev_frame))
        dt = frame_delta / self.physics_fps

        road = road_size or {}
        car = car_size or {}
        road_width = float(road.get("width", 0.0))
        car_width = float(car.get("width", 0.0))
        x_max = max((road_width / 2.0) - (car_width / 2.0), 0.0)

        curr_x = float(curr_car.get("x", 0.0))
        prev_x = float(prev_car.get("x", 0.0))
        curr_z = -float(curr_car.get("z", 0.0))
        prev_z = -float(prev_car.get("z", 0.0))
        curr_y = float(curr_car.get("y", 0.0))
        road_y = float((current_payload.get("positions") or {}).get("road", {}).get("y", road.get("y", 0.0)))

        # Finish line and fatal conditions
        road_depth = float(road.get("depth", 0.0))
        road_length = max((road_depth / 2.0) - 10.0, 0.0)
        if road_length and abs(curr_z) > road_length:
            tags.append("FINISH_LINE")
            session.reset_reward_counters()
            return RewardResult(150.0, tags, True)

        if curr_y < road_y - 0.1:
            tags.append("FALLING_OFF_ROAD")
            session.reset_reward_counters()
            return RewardResult(-150.0, tags, True)

        if x_max and abs(curr_x) >= x_max * 1.02:
            tags.append("NEW_OFFROAD")
            session.reset_reward_counters()
            return RewardResult(-100.0, tags, True)

        # Base reward: forward progress
        forward_progress = curr_z - prev_z
        base_reward = forward_progress * 2.0

        speeds_prev = prev_payload.get("speeds") or {}
        speeds_curr = current_payload.get("speeds") or {}
        forward_speed_prev = -float(speeds_prev.get("z", 0.0))
        forward_speed_curr = -float(speeds_curr.get("z", 0.0))
        speed_tendency = forward_speed_curr - forward_speed_prev
        lateral_speed = float(speeds_curr.get("x", 0.0))

        pen_lateral = self._compute_lateral_penalty(
            session,
            x_max,
            curr_x,
            lateral_speed,
            dt,
            tags,
        )
        reward_speed = self._compute_speed_reward(
            session,
            forward_speed_curr,
            speed_tendency,
            dt,
            tags,
        )

        total_reward = base_reward + pen_lateral + reward_speed
        total_reward = float(torch.clamp(torch.tensor(total_reward), -200.0, 200.0).item())
        return RewardResult(total_reward, tags, done)

    def _compute_lateral_penalty(
        self,
        session: Session,
        x_max: float,
        curr_x: float,
        lateral_speed: float,
        dt: float,
        tags: list[str],
    ) -> float:
        if x_max <= 1e-3:
            session.reset_reward_counters()
            return 0.0

        x_norm = min(abs(curr_x) / max(x_max, 1e-6), 1.0)
        side = -1 if curr_x < 0 else 1
        side_name = "left" if side < 0 else "right"
        v_lat = lateral_speed * side

        if x_norm < self.z_warn:
            self._decay_lateral_counters(session, dt)
            return 0.0

        horizon_warning = self._lateral_horizon(abs(lateral_speed))

        if self.z_warn <= x_norm < self.z_danger:
            counter = f"time_in_warning_{side_name}"
            current_value = getattr(session, counter)
            current_value += dt
            setattr(session, counter, current_value)
            self._decay_other_counters(session, side_name, warning=True, dt=dt)
            severity = (x_norm - self.z_warn) / max(self.z_danger - self.z_warn, 1e-6)
            severity = max(0.0, min(severity, 1.0))
            persistence = 1.0 + self.warn_alpha * current_value
            tendency = 1.0 + self.warn_beta * max(0.0, v_lat)
            time_progress = min(current_value / max(horizon_warning, 1e-6), 1.0)
            target = self.k_warn * severity * persistence * tendency
            penalty = -target * time_progress
            tags.append(f"LATERAL_WARNING_{side_name.upper()}")
        else:
            counter = f"time_in_danger_{side_name}"
            current_value = getattr(session, counter)
            current_value += dt
            setattr(session, counter, current_value)
            self._decay_other_counters(session, side_name, warning=False, dt=dt)
            severity = (x_norm - self.z_danger) / max(1.0 - self.z_danger, 1e-6)
            severity = max(0.0, min(severity, 1.0))
            persistence = 1.0 + self.danger_alpha * current_value
            tendency = 1.0 + self.danger_beta * max(0.0, v_lat)
            horizon_danger = self.lateral_fast_horizon if abs(lateral_speed) >= self.lateral_speed_fast_threshold else self.lateral_slow_horizon
            time_progress = min(current_value / max(horizon_danger, 1e-6), 1.0)
            target = self.k_danger * severity * persistence * tendency
            penalty = -target * time_progress
            tags.append(f"LATERAL_DANGER_{side_name.upper()}")

        penalty = max(-self.lateral_penalty_cap, min(0.0, penalty))
        return penalty

    def _compute_speed_reward(
        self,
        session: Session,
        speed_curr: float,
        speed_tendency: float,
        dt: float,
        tags: list[str],
    ) -> float:
        reward = 0.0
        if speed_curr >= self.speed_warn:
            session.time_in_slow = max(0.0, session.time_in_slow - dt * self.recovery_rate)
            reward = self.k_fast * (speed_curr - self.speed_warn)
            if speed_tendency > 0:
                reward *= 1.0 + self.fast_gamma * speed_tendency
            tags.append("SPEED_GOOD")
            return reward

        session.time_in_slow += dt
        severity = (self.speed_warn - speed_curr) / max(self.speed_warn - self.speed_danger, 1e-6)
        severity = max(0.0, min(severity, 1.0))
        persistence = 1.0 + self.slow_alpha * session.time_in_slow
        tendency = 1.0 + self.slow_beta * max(0.0, -speed_tendency)
        time_progress = min(1.0, session.time_in_slow / max(self.slow_time_horizon, 1e-6))
        target = self.k_slow * severity * persistence * tendency
        penalty = -time_progress * target
        penalty = max(-self.speed_penalty_cap, min(0.0, penalty))

        if speed_curr <= self.speed_danger:
            tags.append("SPEED_DANGER")
        else:
            tags.append("SPEED_SLOW")
        return penalty

    def _decay_lateral_counters(self, session: Session, dt: float) -> None:
        session.time_in_warning_left = self._decay(session.time_in_warning_left, dt)
        session.time_in_warning_right = self._decay(session.time_in_warning_right, dt)
        session.time_in_danger_left = self._decay(session.time_in_danger_left, dt)
        session.time_in_danger_right = self._decay(session.time_in_danger_right, dt)

    def _decay_other_counters(self, session: Session, side_name: str, warning: bool, dt: float) -> None:
        if warning:
            other = "left" if side_name == "right" else "right"
            attr = f"time_in_warning_{other}"
            setattr(session, attr, self._decay(getattr(session, attr), dt))
            setattr(session, f"time_in_danger_{side_name}", self._decay(getattr(session, f"time_in_danger_{side_name}"), dt))
            setattr(session, f"time_in_danger_{other}", self._decay(getattr(session, f"time_in_danger_{other}"), dt))
        else:
            other = "left" if side_name == "right" else "right"
            setattr(session, f"time_in_danger_{other}", self._decay(getattr(session, f"time_in_danger_{other}"), dt))
            setattr(session, f"time_in_warning_{other}", self._decay(getattr(session, f"time_in_warning_{other}", 0.0), dt))

    @staticmethod
    def _decay(value: float, dt: float) -> float:
        return max(0.0, value - dt)

    def _lateral_horizon(self, lateral_speed_abs: float) -> float:
        if lateral_speed_abs < self.lateral_speed_fast_threshold:
            return self.lateral_slow_horizon
        return self.lateral_fast_horizon
