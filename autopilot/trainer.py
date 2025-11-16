"""Background trainer consuming finished episodes."""
from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import List

import torch

from .algo.ppo import PPOConfig, ppo_update
from .data_cleaner import EpisodeCleaner
from .models import ActorCritic, DEVICE
from .storage import Episode

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: ActorCritic, config: PPOConfig | None = None) -> None:
        self.model = model
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.queue: "queue.Queue[Episode]" = queue.Queue(maxsize=256)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.pending: List[Episode] = []
        self._stop_event = threading.Event()
        self.cleaner = EpisodeCleaner()
        self.last_update_at = time.monotonic()
        self.idle_timeout = 30.0
        self.min_timeout_transitions = 512
        self.priority_fraction = 0.6
        self.priority_tag_bonus = 5.0
        self.priority_std_weight = 0.5
        self.thread.start()

    def enqueue_episode(self, episode: Episode) -> None:
        if episode.length == 0:
            return
        try:
            self.queue.put_nowait(episode)
        except queue.Full:
            logger.warning("Trainer queue is full; dropping episode")

    def stop(self) -> None:
        self._stop_event.set()
        self.thread.join(timeout=1)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                episode = self.queue.get(timeout=0.5)
            except queue.Empty:
                if (
                    self.pending
                    and sum(ep.length for ep in self.pending) >= self.min_timeout_transitions
                    and (time.monotonic() - self.last_update_at) > self.idle_timeout
                ):
                    self._train(self.pending)
                    self.pending = []
                continue
            self.pending.append(episode)
            transitions = sum(ep.length for ep in self.pending)
            if transitions >= self.config.batch_size:
                self._train(self.pending)
                self.pending = []

    def _train(self, episodes: List[Episode]) -> None:
        cleaned: List[Episode] = []
        for episode in episodes:
            cleaned_episode = self.cleaner.clean_episode(episode)
            if cleaned_episode.length > 0:
                cleaned.append(cleaned_episode)
        if not cleaned:
            logger.warning("⚠️ Episode cleaning removed all transitions, skipping PPO step")
            return

        prioritized = self._prioritize(cleaned)
        if not prioritized:
            logger.warning("⚠️ Episode prioritization filtered everything, skipping PPO step")
            return

        total_transitions = sum(ep.length for ep in prioritized)
        logger.info(
            "🎯 PPO update on %d prioritized episodes (%d transitions)",
            len(prioritized),
            total_transitions,
        )
        stats = ppo_update(self.model, self.optimizer, prioritized, DEVICE, self.config)
        self.model.training_count += 1
        self.model.last_training_at = datetime.now(timezone.utc)
        self.last_update_at = time.monotonic()
        logger.info(
            "✅ PPO done | policy_loss=%.4f value_loss=%.4f entropy=%.4f", stats["policy_loss"], stats["value_loss"], stats["entropy"]
        )

    def _prioritize(self, episodes: List[Episode]) -> List[Episode]:
        scored: list[tuple[float, Episode]] = []
        for episode in episodes:
            rewards = [abs(t.reward) for t in episode.transitions]
            if not rewards:
                continue
            mean_abs = sum(rewards) / len(rewards)
            variance = sum((r - mean_abs) ** 2 for r in rewards) / len(rewards)
            std = variance ** 0.5
            tag_bonus = 0.0
            for transition in episode.transitions:
                tags = set(transition.tags or [])
                if tags & {"DANGEROUS_SIDE", "NEW_OFFROAD", "FORCED_RESET_LOW_SPEED", "FORCED_RESET_STALLED_Z", "FORCED_RESET_REVERSING"}:
                    tag_bonus = self.priority_tag_bonus
                    break
            priority = mean_abs + self.priority_std_weight * std + tag_bonus
            scored.append((priority, episode))
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        top_k = max(1, int(len(scored) * self.priority_fraction))
        selected = [episode for _, episode in scored[:top_k]]
        if len(selected) < len(episodes):
            logger.info(
                "✨ Prioritization kept %d/%d episodes (top %.0f%%)",
                len(selected),
                len(episodes),
                self.priority_fraction * 100,
            )
        return selected
