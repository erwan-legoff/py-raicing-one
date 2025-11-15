"""Background trainer consuming finished episodes."""
from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timezone
from typing import List

import torch

from .algo.ppo import PPOConfig, ppo_update
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
                continue
            self.pending.append(episode)
            transitions = sum(ep.length for ep in self.pending)
            if transitions >= self.config.batch_size:
                self._train(self.pending)
                self.pending = []

    def _train(self, episodes: List[Episode]) -> None:
        logger.info("🎯 PPO update on %d episodes (%d transitions)", len(episodes), sum(ep.length for ep in episodes))
        stats = ppo_update(self.model, self.optimizer, episodes, DEVICE, self.config)
        self.model.training_count += 1
        self.model.last_training_at = datetime.now(timezone.utc)
        logger.info(
            "✅ PPO done | policy_loss=%.4f value_loss=%.4f entropy=%.4f", stats["policy_loss"], stats["value_loss"], stats["entropy"]
        )
