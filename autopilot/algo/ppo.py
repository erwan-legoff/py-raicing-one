"""Implementation of PPO with GAE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from ..storage import Episode


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 1.5e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    batch_size: int = 1024
    epochs: int = 6
    max_grad_norm: float = 0.5


def _build_tensors(episodes: Iterable[Episode], device: torch.device):
    obs = []
    actions = []
    log_probs = []
    for episode in episodes:
        for transition in episode.transitions:
            obs.append(transition.obs)
            actions.append(transition.action_idx)
            log_probs.append(transition.log_prob)
    return (
        torch.stack(obs).to(device),
        torch.tensor(actions, dtype=torch.long, device=device),
        torch.tensor(log_probs, dtype=torch.float32, device=device),
    )


def _compute_returns_and_advantages(
    episodes: Iterable[Episode],
    config: PPOConfig,
    device: torch.device,
):
    advantages: list[float] = []
    returns: list[float] = []
    for episode in episodes:
        adv = 0.0
        next_value = 0.0
        ep_advs: list[float] = []
        ep_returns: list[float] = []
        for transition in reversed(episode.transitions):
            mask = 0.0 if transition.done else 1.0
            delta = transition.reward + config.gamma * next_value * mask - transition.value
            adv = delta + config.gamma * config.gae_lambda * mask * adv
            ep_advs.insert(0, adv)
            ep_returns.insert(0, adv + transition.value)
            next_value = transition.value
        advantages.extend(ep_advs)
        returns.extend(ep_returns)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
        advantages_tensor.std(unbiased=False) + 1e-8
    )
    return advantages_tensor, returns_tensor


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: Iterable[Episode],
    device: torch.device,
    config: PPOConfig,
) -> dict[str, float]:
    obs, actions, old_log_probs = _build_tensors(episodes, device)
    advantages, returns = _compute_returns_and_advantages(episodes, config, device)

    num_samples = obs.size(0)
    batch_size = min(config.batch_size, num_samples)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    for _ in range(config.epochs):
        indices = torch.randperm(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            idx = indices[start:end]

            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            logits, values = model(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - batch_old_log_probs).exp()
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = F.mse_loss(values, batch_returns)

            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            stats["policy_loss"] = float(policy_loss.detach())
            stats["value_loss"] = float(value_loss.detach())
            stats["entropy"] = float(entropy.detach())

    return stats
