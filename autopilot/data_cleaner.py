"""Utilities to probabilistically drop low-value transitions before training."""
from __future__ import annotations

import logging
import math
import random
from typing import Iterable

from .storage import Episode, Transition

logger = logging.getLogger(__name__)

IDX_LEFT, IDX_FORWARD, IDX_RIGHT, IDX_BACKWARD = 0, 1, 2, 3


def _classify_rewards(transitions: Iterable[Transition]) -> dict[str, int]:
    categories = {
        "neg_extreme": 0,
        "neg_high": 0,
        "neg_neutral": 0,
        "neutral": 0,
        "pos_neutral": 0,
        "pos_high": 0,
        "pos_extreme": 0,
    }
    for transition in transitions:
        reward = float(getattr(transition, "reward", 0.0) or 0.0)
        if reward <= -75:
            categories["neg_extreme"] += 1
        elif reward <= -15:
            categories["neg_high"] += 1
        elif reward < 0:
            categories["neg_neutral"] += 1
        elif reward <= 15:
            categories["neutral"] += 1
        elif reward < 75:
            categories["pos_neutral"] += 1
        elif reward < 150:
            categories["pos_high"] += 1
        else:
            categories["pos_extreme"] += 1
    return categories


def _log_reward_stats(label: str, total: int, stats: dict[str, int]) -> None:
    if total <= 0:
        logger.info("🧮 Stats rewards %s: aucun point", label)
        return
    pct = {key: (value / total) * 100 for key, value in stats.items()}
    logger.info(
        "🧮 Stats rewards %s: neutral=%.1f%% | pos[+]=%.1f%% | pos[++]=%.1f%% | pos[+++]=%.1f%% | "
        "neg[-]=%.1f%% | neg[--]=%.1f%% | neg[---]=%.1f%%",
        label,
        pct["neutral"],
        pct["pos_neutral"],
        pct["pos_high"],
        pct["pos_extreme"],
        pct["neg_neutral"],
        pct["neg_high"],
        pct["neg_extreme"],
    )


class EpisodeCleaner:
    """Apply the historical clean_history sampling to PPO episodes."""

    def __init__(
        self,
        tag_probabilities: dict[str, float] | None = None,
        min_keep_ratios: dict[str, float] | None = None,
        high_reward_prob: float = 0.02,
        rng: random.Random | None = None,
    ) -> None:
        self.rng = rng or random.Random()
        self.tag_probabilities = {
            "CENTERED": 0.25,
            "BORING_CENTERED": 0.75,
            "FINISH_LINE": 0.05,
            "SAVING_DANGEROUS_SIDE": 0.05,
            "PANIC_RECENTERING": 0.05,
            "PROGRESSING_CENTER": 0.33,
            "STALLING_CENTERED": 0.5,
            "LOW_FORWARD_SPEED": 0.20,
        }
        if tag_probabilities:
            self.tag_probabilities.update(tag_probabilities)
        self.min_keep_ratios = {
            "CENTERED": 0.15,
            "BORING_CENTERED": 0.05,
        }
        if min_keep_ratios:
            self.min_keep_ratios.update(min_keep_ratios)
        self.high_reward_prob = high_reward_prob
        self.protected_tags = {"DANGEROUS_SIDE", "NEW_OFFROAD", "STILL_OFFROAD"}

    def _random(self) -> float:
        return self.rng.random()

    def clean_episode(self, episode: Episode) -> Episode:
        transitions = list(episode.transitions)
        if not transitions:
            return Episode([], 0.0, 0)

        total = len(transitions)
        center_total = sum(1 for t in transitions if "CENTERED" in (t.tags or []))
        boring_total = sum(1 for t in transitions if "BORING_CENTERED" in (t.tags or []))
        reward_stats_before = _classify_rewards(transitions)
        _log_reward_stats("avant", total, reward_stats_before)

        min_center_keep = math.ceil(total * self.min_keep_ratios.get("CENTERED", 0.0))
        min_boring_keep = math.ceil(total * self.min_keep_ratios.get("BORING_CENTERED", 0.0))
        center_remaining = center_total
        boring_remaining = boring_total

        kept: list[Transition] = []
        removed = 0

        for transition in transitions:
            tags = set(transition.tags or [])
            is_center = "CENTERED" in tags
            is_boring = "BORING_CENTERED" in tags
            mask = transition.action_mask or []
            left_active = bool(len(mask) > IDX_LEFT and mask[IDX_LEFT] > 0)
            right_active = bool(len(mask) > IDX_RIGHT and mask[IDX_RIGHT] > 0)
            forward_active = bool(len(mask) > IDX_FORWARD and mask[IDX_FORWARD] > 0)
            backward_active = bool(len(mask) > IDX_BACKWARD and mask[IDX_BACKWARD] > 0)
            lateral_active = left_active or right_active

            if lateral_active:
                kept.append(transition)
                continue

            if tags & self.protected_tags:
                kept.append(transition)
                continue

            if is_center and not is_boring and center_remaining <= min_center_keep:
                kept.append(transition)
                continue

            if is_boring and boring_remaining <= min_boring_keep:
                kept.append(transition)
                continue

            only_forward = forward_active and not lateral_active and not backward_active

            if only_forward and transition.reward < 150 and self._random() < 0.05:
                removed += 1
                if is_center:
                    center_remaining -= 1
                if is_boring:
                    boring_remaining -= 1
                continue

            if not (lateral_active or forward_active or backward_active) and self._random() < 0.01:
                removed += 1
                if is_center:
                    center_remaining -= 1
                if is_boring:
                    boring_remaining -= 1
                continue

            if (
                transition.reward > 100
                and "CENTERED" in tags
                and "SKIDDING_SIDEWAYS" in tags
            ):
                removed += 1
                if is_center:
                    center_remaining -= 1
                if is_boring:
                    boring_remaining -= 1
                continue

            if transition.reward > 50 and self._random() < self.high_reward_prob:
                removed += 1
                if is_center:
                    center_remaining -= 1
                if is_boring:
                    boring_remaining -= 1
                continue

            removal_probability = 0.0
            for tag in tags:
                prob = self.tag_probabilities.get(tag)
                if prob is None:
                    continue
                if tag == "LOW_FORWARD_SPEED" and "CENTERED" in tags:
                    continue
                removal_probability = max(removal_probability, prob)

            if removal_probability > 0 and self._random() < removal_probability:
                removed += 1
                if is_center:
                    center_remaining -= 1
                if is_boring:
                    boring_remaining -= 1
                continue

            kept.append(transition)

        new_total = len(kept)
        reward_stats_after = _classify_rewards(kept)
        if new_total:
            _log_reward_stats("après", new_total, reward_stats_after)

        if removed:
            logger.info(
                "🧹 Episode cleaner: removed %d/%d transitions (%.1f%%)",
                removed,
                total,
                (removed / total) * 100,
            )

        return Episode(
            transitions=kept,
            total_reward=sum(t.reward for t in kept),
            length=new_total,
        )
