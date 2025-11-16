"""Bridge between the WebSocket payloads and the RL core."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import List

import torch

from datetime import datetime, timezone

from .models import ActorCritic, DEVICE
from .reward_engine import RewardEngine
from .session_manager import Session, SessionManager
from .storage import Transition
from .trainer import Trainer

SENSOR_ORDER = [
    "left45Ray",
    "left22Ray",
    "leftNarrowRay",
    "frontRay",
    "rightNarrowRay",
    "right22Ray",
    "right45Ray",
]
DRIVING_INPUTS = {0: "LEFT", 1: "FORWARD", 2: "RIGHT", 3: "BACKWARD"}
NUM_FEATURES = 28
LOGGER = logging.getLogger(__name__)


class FeatureNormalizer:
    def __init__(self, size: int, device: torch.device = DEVICE) -> None:
        self.size = size
        self.device = device
        self.mean = torch.zeros(size, dtype=torch.float32, device=device)
        self.m2 = torch.zeros(size, dtype=torch.float32, device=device)
        self.count = 1e-4

    def observe(self, x: torch.Tensor) -> None:
        x = x.to(self.device)
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        self.observe(x)
        variance = self.m2 / max(self.count - 1.0, 1.0)
        std = torch.sqrt(torch.clamp(variance, min=1e-6))
        return torch.clamp((x - self.mean) / std, -5.0, 5.0)


class ObservationBuilder:
    def __init__(self, normalizer: FeatureNormalizer) -> None:
        self.normalizer = normalizer

    def build(self, payload: dict, session: Session) -> torch.Tensor:
        sensors = payload.get("sensors", {})
        speeds = payload.get("speeds", {})
        accels = payload.get("accelerations", {})
        car_pos = (payload.get("positions") or {}).get("car", {})
        road_size = payload.get("roadSize") or {}
        car_size = payload.get("carSize") or {}

        sensor_values = [float(sensors.get(name, 0.0)) for name in SENSOR_ORDER]
        speed_values = [float(speeds.get(axis, 0.0)) for axis in ("x", "y", "z")]
        accel_values = [float(accels.get(axis, 0.0)) for axis in ("x", "y", "z")]
        car_positions = [float(car_pos.get(axis, 0.0)) for axis in ("x", "y", "z")]
        road_values = [float(road_size.get(axis, 0.0)) for axis in ("width", "height", "depth")]
        car_values = [float(car_size.get(axis, 0.0)) for axis in ("width", "height", "depth")]
        last_mask = session.last_action_mask or [0, 0, 0, 0]
        frame_norm = float(payload.get("frameId", 0)) / 10_000.0
        last_reward = session.last_reward

        features = (
            sensor_values
            + speed_values
            + accel_values
            + car_positions
            + road_values
            + car_values
            + last_mask
            + [frame_norm, last_reward]
        )
        if len(features) != NUM_FEATURES:
            raise ValueError(f"Observation has {len(features)} features, expected {NUM_FEATURES}")
        tensor = torch.tensor(features, dtype=torch.float32)
        return self.normalizer.normalize(tensor)


class RewardNormalizer:
    """Online normalization for rewards to keep magnitudes stable."""

    def __init__(self, clip_value: float = 5.0) -> None:
        self.clip_value = clip_value
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 1e-4

    def normalize(self, value: float) -> float:
        self.count += 1.0
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        variance = self.m2 / max(self.count - 1.0, 1.0)
        std = max(variance, 1e-6) ** 0.5
        normalized = (value - self.mean) / std
        return float(max(-self.clip_value, min(self.clip_value, normalized)))


@dataclass
class ActionDecision:
    action_idx: int
    action_mask: list[int]
    log_prob: float


class ActionSelector:
    def __init__(self, epsilon_start: float = 0.1, epsilon_end: float = 0.02, decay: float = 50_000.0) -> None:
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay = decay
        self.total_steps = 0

    def _epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-self.total_steps / self.decay)

    def select(self, logits: torch.Tensor, evaluation: bool = False) -> ActionDecision:
        probs = torch.softmax(logits, dim=-1)
        epsilon = self._epsilon()
        self.total_steps += 1

        if evaluation:
            action_idx = int(torch.argmax(probs).item())
        else:
            if torch.rand(1).item() < epsilon:
                action_idx = int(torch.randint(0, probs.size(-1), (1,)).item())
            else:
                dist = torch.distributions.Categorical(probs=probs)
                action_idx = int(dist.sample().item())
        dist = torch.distributions.Categorical(probs=probs)
        log_prob = float(dist.log_prob(torch.tensor(action_idx, device=logits.device)).item())
        action_mask = [1 if i == action_idx else 0 for i in range(probs.size(-1))]
        return ActionDecision(action_idx=action_idx, action_mask=action_mask, log_prob=log_prob)


class EnvironmentBridge:
    def __init__(
        self,
        model: ActorCritic,
        trainer: Trainer,
        reward_engine: RewardEngine,
        session_manager: SessionManager,
    ) -> None:
        self.model = model
        self.trainer = trainer
        self.reward_engine = reward_engine
        self.sessions = session_manager
        self.normalizer = FeatureNormalizer(NUM_FEATURES)
        self.obs_builder = ObservationBuilder(self.normalizer)
        self.action_selector = ActionSelector()
        self.reward_normalizer = RewardNormalizer()
        log_dir = os.path.join("sessions", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.session_log_path = os.path.join(
            log_dir, f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.low_speed_threshold = 0.2
        self.low_speed_limit = 96
        self.stagnation_delta = 0.2
        self.stagnation_limit = 160
        self.reverse_limit = 64
        self.forced_reset_penalty = -250.0

    def training_metadata(self) -> dict[str, object]:
        last_training_at = getattr(self.model, "last_training_at", None)
        if last_training_at is None:
            last_training_at = datetime.now(timezone.utc)
        return {
            "count": getattr(self.model, "training_count", 0),
            "last": last_training_at.isoformat(),
        }

    def process_step(self, client_id: str, payload: dict) -> dict:
        session = self.sessions.get(client_id)
        obs = self.obs_builder.build(payload, session)
        reward = 0.0
        tags: List[str] = []
        done = False

        forced_reset = False
        normalized_reward = 0.0
        display_reward = 0.0
        if session.last_obs is not None and session.last_payload is not None and session.last_action_mask is not None:
            result = self.reward_engine.compute_reward(
                session.last_payload,
                payload,
                session.last_action_mask,
                payload.get("frameId", 0),
                payload.get("roadSize"),
                payload.get("carSize"),
                session,
            )
            reward = result.reward
            display_reward = reward
            normalized_reward = self.reward_normalizer.normalize(reward)
            tags = result.tags
            done = result.done
            reset_tag = None
            if not done:
                forced_reset, reset_tag, penalty = self._check_stall_reset(session, payload)
                if forced_reset:
                    done = True
                    reward += penalty
                    display_reward = reward
                    normalized_reward = self.reward_normalizer.normalize(reward)
                    tags = list(tags)
                    if reset_tag:
                        tags.append(reset_tag)

            transition = Transition(
                obs=session.last_obs.detach(),
                action_idx=session.last_action_idx or 0,
                action_mask=session.last_action_mask,
                log_prob=session.last_log_prob,
                reward=normalized_reward,
                done=done,
                value=session.last_value,
                tags=tags,
                frame_id=int(payload.get("frameId", 0)),
            )
            session.rollout_buffer.add(transition)
            session.episode_steps += 1
            session.episode_return += reward
            session.last_reward = display_reward
            session.last_tags = tags
            if done:
                if len(session.rollout_buffer) > 0:
                    episode = session.rollout_buffer.pop_episode()
                    self.trainer.enqueue_episode(episode)
                    session.completed_episodes += 1
                    session.evaluation_mode = (session.completed_episodes % 10) == 0
                session.reset_episode()
                session.clear_last_transition()
                session.reset_stall_counters()
                session.reset_reward_counters()
                if not forced_reset:
                    session.clear_forced_reset_streak()
                session.last_reward = 0.0
                session.last_tags = []
                self._log_session_step(client_id, payload, display_reward, tags, done=True)
                return self._build_message(["RESET"], reward, tags)
        else:
            session.last_reward = 0.0
            session.last_tags = []
            session.reset_stall_counters()
            session.reset_reward_counters()
            if not forced_reset:
                session.clear_forced_reset_streak()
        self._log_session_step(client_id, payload, display_reward, tags, done=False)

        with torch.no_grad():
            logits, value = self.model(obs.unsqueeze(0).to(DEVICE))
        decision = self.action_selector.select(logits[0], evaluation=session.evaluation_mode)

        session.last_obs = obs.detach()
        session.last_payload = payload
        session.last_action_idx = decision.action_idx
        session.last_action_mask = decision.action_mask
        session.last_log_prob = decision.log_prob
        session.last_value = float(value.squeeze().item())

        driving_inputs = [DRIVING_INPUTS[idx] for idx, v in enumerate(decision.action_mask) if v]
        if not driving_inputs:
            driving_inputs = ["FORWARD"]
            session.last_action_mask = [0, 1, 0, 0]
            session.last_action_idx = 1

        return self._build_message(driving_inputs, session.last_reward, session.last_tags)

    def _build_message(self, driving_inputs: list[str], reward: float, tags: list[str]) -> dict:
        return {
            "driving_inputs": driving_inputs,
            "reward": reward,
            "tags": tags,
            "training": self.training_metadata(),
        }

    def _log_session_step(self, client_id: str, payload: dict, reward: float, tags: list[str], done: bool) -> None:
        frame = payload.get("frameId")
        pos = (payload.get("positions") or {}).get("car", {})
        with open(self.session_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"{datetime.now(timezone.utc).isoformat()} client={client_id} frame={frame} "
                f"pos=({pos.get('x'):.2f},{pos.get('z'):.2f}) reward={reward:.3f} "
                f"tags={tags} done={done}\n"
            )

    def _check_stall_reset(self, session: Session, current_payload: dict) -> tuple[bool, str | None, float]:
        prev_payload = session.last_payload
        if not prev_payload:
            session.reset_stall_counters()
            return False, None, 0.0

        prev_car = (prev_payload.get("positions") or {}).get("car") or {}
        curr_car = (current_payload.get("positions") or {}).get("car") or {}
        if not prev_car or not curr_car:
            return False, None, 0.0

        prev_z = float(prev_car.get("z", 0.0))
        curr_z = float(curr_car.get("z", 0.0))
        progress = prev_z - curr_z

        if abs(progress) < self.stagnation_delta:
            session.stagnation_steps += 1
        else:
            session.stagnation_steps = 0

        if progress < -self.stagnation_delta:
            session.reverse_steps += 1
        else:
            session.reverse_steps = 0

        curr_z_speed = -float((current_payload.get("speeds") or {}).get("z", 0.0))
        if curr_z_speed < self.low_speed_threshold:
            session.low_speed_steps += 1
        else:
            session.low_speed_steps = 0

        if session.low_speed_steps >= self.low_speed_limit:
            session.reset_stall_counters()
            session.register_forced_reset()
            multiplier = 1.0 + 0.5 * (session.forced_reset_streak - 1)
            penalty = self.forced_reset_penalty * multiplier
            LOGGER.info(
                "🔁 Forced RESET: low forward speed detected for %d frames (streak=%d, penalty=%.1f)",
                self.low_speed_limit,
                session.forced_reset_streak,
                penalty,
            )
            return True, "FORCED_RESET_LOW_SPEED", penalty

        if session.reverse_steps >= self.reverse_limit:
            session.reset_stall_counters()
            session.register_forced_reset()
            multiplier = 1.0 + 0.5 * (session.forced_reset_streak - 1)
            penalty = self.forced_reset_penalty * multiplier
            LOGGER.info(
                "🔁 Forced RESET: sustained backward motion detected for %d frames (streak=%d, penalty=%.1f)",
                self.reverse_limit,
                session.forced_reset_streak,
                penalty,
            )
            return True, "FORCED_RESET_REVERSING", penalty

        if session.stagnation_steps >= self.stagnation_limit:
            session.reset_stall_counters()
            session.register_forced_reset()
            multiplier = 1.0 + 0.5 * (session.forced_reset_streak - 1)
            penalty = self.forced_reset_penalty * multiplier
            LOGGER.info(
                "🔁 Forced RESET: no forward progress detected for %d frames (streak=%d, penalty=%.1f)",
                self.stagnation_limit,
                session.forced_reset_streak,
                penalty,
            )
            return True, "FORCED_RESET_STALLED_Z", penalty

        if progress > self.stagnation_delta * 2 or curr_z_speed > self.low_speed_threshold * 2:
            session.decay_forced_reset_streak()

        return False, None, 0.0
