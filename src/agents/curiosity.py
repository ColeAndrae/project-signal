"""
Intrinsic Curiosity Module (ICM) for Project SIGNAL.

Bootstraps meaningful communication by rewarding agents for sending messages
that help recipients predict future observations. The key insight: if agent A's
message helps agent B predict what B will see in the next timestep, that message
carried useful information — and should be reinforced.

Architecture:
    ┌─────────────────────────────────────────────┐
    │            Forward Dynamics Model            │
    │                                              │
    │   ┌──────────┐   ┌──────────┐               │
    │   │ Current   │   │ Received │               │
    │   │ Features  │   │ Message  │               │
    │   └────┬─────┘   └────┬─────┘               │
    │        └──────┬───────┘                      │
    │          ┌────▼────┐                         │
    │          │ Forward  │                        │
    │          │ MLP      │                        │
    │          └────┬────┘                         │
    │          ┌────▼────┐     ┌──────────┐        │
    │          │Predicted │ vs │ Actual    │        │
    │          │Features  │    │ Features  │        │
    │          └─────────┘     └──────────┘        │
    │               prediction error               │
    │                    ↓                         │
    │            intrinsic reward                   │
    └─────────────────────────────────────────────┘

The curiosity reward β-anneals over training so it bootstraps communication
early without distorting the final triage policy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """
    Encode raw observation components into a compact feature vector.

    Shares the same architecture as the policy's encoders but with
    independent weights — this prevents the curiosity gradient from
    destabilizing the policy representations.

    Input:  grid (B, C, H, W) + state (B, 7)
    Output: (B, feature_dim)
    """

    def __init__(self, grid_channels: int = 6, feature_dim: int = 64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        # Input 7x7 → conv1 7x7 → conv2 5x5 → flat = 32 * 5 * 5 = 800
        self._flat_size = 32 * 5 * 5

        self.fc = nn.Sequential(
            nn.Linear(self._flat_size + 8, 128),  # +7 for state vector
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, grid: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid:  (B, C, H, W)
            state: (B, 7)
        Returns:
            (B, feature_dim)
        """
        x = self.conv(grid)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, state], dim=-1)
        return self.fc(x)


class ForwardDynamicsModel(nn.Module):
    """
    Predicts the next observation's features given current features + received message.

    This is the core of ICM: if the prediction improves when the message is included,
    then the message is carrying useful information about the world.
    """

    def __init__(self, feature_dim: int = 64, message_dim: int = 24, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self, features: torch.Tensor, message: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim) — current observation features.
            message:  (B, message_dim) — flattened one-hot received message.
        Returns:
            (B, feature_dim) — predicted next features.
        """
        x = torch.cat([features, message], dim=-1)
        return self.net(x)


class IntrinsicCuriosityModule(nn.Module):
    """
    Complete ICM with feature encoding, forward prediction, and reward computation.

    Usage during training:
        1. At each timestep, call compute_intrinsic_reward() with current + next
           observations and the received message.
        2. Add the scaled reward to the extrinsic reward: r_total = r_ext + β * r_int
        3. Call get_loss() to get the forward model's training loss.
        4. β anneals over training via update_beta().

    The intrinsic reward is the L2 prediction error of the forward model:
        r_intrinsic = ||φ(s_{t+1}) - f(φ(s_t), m_t)||²

    where φ is the feature encoder, f is the forward model, and m_t is the
    message received at time t.
    """

    def __init__(
        self,
        grid_channels: int = 6,
        feature_dim: int = 64,
        message_dim: int = 24,     # message_length * vocab_size = 3 * 8
        hidden_dim: int = 128,
        beta_start: float = 0.5,
        beta_end: float = 0.01,
        anneal_episodes: int = 3000,
    ):
        super().__init__()

        self.feature_encoder = FeatureEncoder(
            grid_channels=grid_channels, feature_dim=feature_dim,
        )
        self.forward_model = ForwardDynamicsModel(
            feature_dim=feature_dim, message_dim=message_dim, hidden_dim=hidden_dim,
        )

        self.feature_dim = feature_dim
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_episodes = anneal_episodes

        # Running statistics for reward normalization
        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

    def encode(self, grid: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Encode observations into feature space (detached from policy)."""
        return self.feature_encoder(grid, state)

    def compute_intrinsic_reward(
        self,
        grid_t: torch.Tensor,
        state_t: torch.Tensor,
        grid_tp1: torch.Tensor,
        state_tp1: torch.Tensor,
        message_received: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute intrinsic curiosity reward for a batch of transitions.

        Args:
            grid_t:            (B, C, H, W) — current grid observation.
            state_t:           (B, 7) — current state vector.
            grid_tp1:          (B, C, H, W) — next grid observation.
            state_tp1:         (B, 7) — next state vector.
            message_received:  (B, message_dim) — one-hot flattened message received.
            normalize:         Whether to normalize the reward.

        Returns:
            (B,) — intrinsic reward per sample (already scaled by β).
        """
        # Encode current and next observations
        with torch.no_grad():
            features_t = self.feature_encoder(grid_t, state_t)
            features_tp1 = self.feature_encoder(grid_tp1, state_tp1)

        # Predict next features from current features + message
        predicted_tp1 = self.forward_model(features_t.detach(), message_received)

        # Prediction error = intrinsic reward
        reward = 0.5 * (predicted_tp1 - features_tp1.detach()).pow(2).sum(dim=-1)

        if normalize:
            reward = self._normalize_reward(reward)

        return self.beta * reward

    def get_loss(
        self,
        grid_t: torch.Tensor,
        state_t: torch.Tensor,
        grid_tp1: torch.Tensor,
        state_tp1: torch.Tensor,
        message_received: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the forward model training loss.

        This is called during the PPO update to train the forward dynamics model
        alongside the policy and critic.

        Returns:
            Scalar loss (mean over batch).
        """
        # Encode (with gradient for the feature encoder)
        features_t = self.feature_encoder(grid_t, state_t)
        features_tp1 = self.feature_encoder(grid_tp1, state_tp1)

        # Predict
        predicted_tp1 = self.forward_model(features_t, message_received)

        # MSE loss against actual next features
        loss = F.mse_loss(predicted_tp1, features_tp1.detach())

        return loss

    def update_beta(self, episode: int) -> float:
        """
        Anneal the curiosity coefficient β linearly from beta_start to beta_end.

        Args:
            episode: Current training episode number.

        Returns:
            Updated β value.
        """
        if episode >= self.anneal_episodes:
            self.beta = self.beta_end
        else:
            progress = episode / self.anneal_episodes
            self.beta = self.beta_start + progress * (self.beta_end - self.beta_start)
        return self.beta

    def _normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """
        Normalize intrinsic reward using running statistics.

        Prevents the curiosity signal from dominating early in training
        when the forward model has high prediction error on everything.
        """
        batch_mean = reward.mean().item()
        batch_var = reward.var().item() if reward.numel() > 1 else 1.0
        self._reward_count += 1

        # Exponential moving average
        alpha = min(1.0, 1.0 / self._reward_count)
        self._reward_running_mean = (
            (1 - alpha) * self._reward_running_mean + alpha * batch_mean
        )
        self._reward_running_var = (
            (1 - alpha) * self._reward_running_var + alpha * batch_var
        )

        std = max(self._reward_running_var ** 0.5, 1e-8)
        return (reward - self._reward_running_mean) / std

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts for each submodule."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "feature_encoder": _count(self.feature_encoder),
            "forward_model": _count(self.forward_model),
            "total": _count(self),
        }
