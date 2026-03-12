"""
Multi-Agent PPO (MAPPO) trainer for Project SIGNAL.

Implements the PPO clipped surrogate objective with:
    - Centralized value function (global state → V(s))
    - Decentralized policies (local obs → action)
    - Entropy bonus to encourage exploration
    - ICM curiosity loss (optional)
    - Gradient clipping for stability
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.networks import SignalAgent
from src.agents.curiosity import IntrinsicCuriosityModule
from src.environment.spaces import compute_action_log_prob
from src.training.buffer import RolloutBuffer


class MAPPOTrainer:
    """
    Trains the shared SignalAgent using PPO with centralized critic.

    Usage:
        trainer = MAPPOTrainer(agent, icm, config)
        for each episode:
            # ... collect rollout into buffer ...
            metrics = trainer.update(buffer)
    """

    def __init__(
        self,
        agent: SignalAgent,
        icm: IntrinsicCuriosityModule | None = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_icm: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        icm_loss_coef: float = 0.1,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
    ):
        self.agent = agent
        self.icm = icm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.icm_loss_coef = icm_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Separate optimizers for actor and critic (different learning rates)
        self.actor_optimizer = optim.Adam(
            agent.policy.parameters(), lr=lr_actor,
        )
        self.critic_optimizer = optim.Adam(
            agent.critic.parameters(), lr=lr_critic,
        )
        self.icm_optimizer = (
            optim.Adam(icm.parameters(), lr=lr_icm) if icm else None
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """
        Run PPO update using data from the rollout buffer.

        Args:
            buffer: Filled RolloutBuffer with computed returns and advantages.

        Returns:
            Dict of training metrics.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_icm_loss = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            batches = buffer.generate_batches(self.mini_batch_size)

            for batch in batches:
                metrics = self._update_batch(batch)
                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_entropy += metrics["entropy"]
                total_icm_loss += metrics["icm_loss"]
                total_clip_fraction += metrics["clip_fraction"]
                num_updates += 1

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "icm_loss": total_icm_loss / n,
            "clip_fraction": total_clip_fraction / n,
        }

    def _update_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run a single PPO update step on one mini-batch."""

        # ---- Policy forward pass ----
        move_logits, task_logits, msg_logits = self.agent.policy(
            batch["grids"], batch["states"], batch["messages_in"],
        )

        # Compute new log probs and entropy for the stored actions
        new_log_probs, entropy = compute_action_log_prob(
            move_logits, task_logits, msg_logits,
            batch["moves"], batch["tasks"], batch["msg_tokens"],
        )

        # ---- PPO clipped surrogate ----
        old_log_probs = batch["log_probs"]
        ratio = torch.exp(new_log_probs - old_log_probs)

        advantages = batch["advantages"]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()

        # ---- Value function loss ----
        new_values = self.agent.critic(
            batch["global_grids"],
            batch["all_agent_states"],
            batch["all_agent_messages"],
        ).squeeze(-1)

        returns = batch["returns"]
        value_loss = nn.functional.mse_loss(new_values, returns)

        # ---- Actor update ----
        actor_total = policy_loss + self.entropy_coef * entropy_loss
        self.actor_optimizer.zero_grad()
        actor_total.backward()
        nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # ---- Critic update ----
        critic_total = self.value_loss_coef * value_loss
        self.critic_optimizer.zero_grad()
        critic_total.backward()
        nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # ---- ICM update (optional) ----
        icm_loss_val = 0.0
        if self.icm is not None and self.icm_optimizer is not None:
            # Flatten received messages for ICM (take mean across senders)
            msg_for_icm = batch["messages_in"].mean(dim=1)  # (B, msg_flat_dim)
            # Trim or pad to expected 24 dims
            msg_for_icm = msg_for_icm[:, :24]

            icm_loss = self.icm.get_loss(
                batch["grids"], batch["states"],
                batch["next_grids"], batch["next_states"],
                msg_for_icm,
            )
            icm_total = self.icm_loss_coef * icm_loss

            self.icm_optimizer.zero_grad()
            icm_total.backward()
            nn.utils.clip_grad_norm_(self.icm.parameters(), self.max_grad_norm)
            self.icm_optimizer.step()
            icm_loss_val = icm_loss.item()

        # ---- Metrics ----
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.ppo_clip).float().mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "icm_loss": icm_loss_val,
            "clip_fraction": clip_fraction,
        }
