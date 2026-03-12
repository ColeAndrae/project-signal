"""
Neural network architectures for Project SIGNAL agents.

Architecture overview:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Grid (CNN)  │     │  State (MLP) │     │  Msgs (GRU)  │
    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
           │                    │                    │
           └────────────┬───────┴────────────────────┘
                        │
                  ┌─────▼──────┐
                  │  Core MLP  │
                  │  (256→256) │
                  └──┬───┬───┬─┘
                     │   │   │
              ┌──────▼┐ ┌▼────┐ ┌▼──────────┐
              │ Move  │ │Task │ │ Message    │
              │ (5)   │ │(6)  │ │ (L × V)   │
              └───────┘ └─────┘ └────────────┘

Components:
    - GridEncoder:      Small CNN that processes the padded local grid observation.
    - MessageEncoder:   GRU that processes one-hot encoded incoming messages.
    - PolicyNetwork:    Full actor with 3 factored output heads.
    - CentralizedCritic: Value function that sees global state (CTDE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.environment.spaces import (
    MAX_VISION_SIZE,
    NUM_CHANNELS,
    STATE_DIM,
)


# ============================================================
# Grid Encoder (CNN)
# ============================================================

class GridEncoder(nn.Module):
    """
    Small CNN that encodes the local grid observation.

    Input:  (B, C=6, H=7, W=7)
    Output: (B, feature_dim)
    """

    def __init__(self, in_channels: int = NUM_CHANNELS, feature_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        # Compute the flattened size after convolutions
        # Input: (7, 7) → conv1 (7, 7) → conv2 (7, 7) → conv3 (5, 5)
        self._flat_size = 64 * 5 * 5  # 1600

        self.fc = nn.Linear(self._flat_size, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) grid tensor, channels-first.
        Returns:
            (B, feature_dim) encoded grid features.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))
        return x


# ============================================================
# Message Encoder (GRU)
# ============================================================

class MessageEncoder(nn.Module):
    """
    GRU-based encoder for incoming messages from other agents.

    Processes each agent's message independently, then aggregates via mean pooling.
    This is permutation-invariant over sender identity — the network learns
    *what* was said, not *who* said it (role info is in the grid observation).

    Input:  (B, num_other_agents, message_length * vocab_size) — one-hot flattened
    Output: (B, hidden_dim)
    """

    def __init__(
        self,
        num_other_agents: int = 3,
        message_length: int = 3,
        vocab_size: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_other_agents = num_other_agents
        self.message_length = message_length
        self.vocab_size = vocab_size
        self.token_dim = vocab_size  # each token is a one-hot vector of this size

        # GRU processes one message as a sequence of tokens
        self.gru = nn.GRU(
            input_size=vocab_size,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.hidden_dim = hidden_dim

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """
        Args:
            messages: (B, num_other_agents, message_length * vocab_size)
        Returns:
            (B, hidden_dim) — aggregated message representation.
        """
        B, N, flat_dim = messages.shape

        # Reshape to sequence of tokens: (B, N, L*V) → (B*N, L, V)
        msgs = messages.reshape(B * N, self.message_length, self.vocab_size)

        # Run GRU over token sequence
        _, hidden = self.gru(msgs)  # hidden: (1, B*N, hidden_dim)
        hidden = hidden.squeeze(0)  # (B*N, hidden_dim)

        # Reshape back and mean-pool over senders
        hidden = hidden.reshape(B, N, self.hidden_dim)  # (B, N, H)
        aggregated = hidden.mean(dim=1)  # (B, H) — permutation invariant

        return aggregated


# ============================================================
# Policy Network (Actor)
# ============================================================

class PolicyNetwork(nn.Module):
    """
    Full actor network with factored action heads.

    Encodes grid + state + messages into a shared representation,
    then outputs logits for three independent action distributions:
        - Movement (5 actions)
        - Task (6 actions)
        - Message tokens (message_length × vocab_size)
    """

    def __init__(
        self,
        grid_feature_dim: int = 128,
        message_hidden_dim: int = 64,
        core_hidden_dim: int = 256,
        move_dim: int = 5,
        task_dim: int = 8,
        message_length: int = 3,
        vocab_size: int = 8,
        num_other_agents: int = 3,
    ):
        super().__init__()

        self.message_length = message_length
        self.vocab_size = vocab_size

        # Encoders
        self.grid_encoder = GridEncoder(
            in_channels=NUM_CHANNELS, feature_dim=grid_feature_dim,
        )
        self.message_encoder = MessageEncoder(
            num_other_agents=num_other_agents,
            message_length=message_length,
            vocab_size=vocab_size,
            hidden_dim=message_hidden_dim,
        )

        # Core MLP — combines all encoded features
        combined_dim = grid_feature_dim + STATE_DIM + message_hidden_dim
        self.core = nn.Sequential(
            nn.Linear(combined_dim, core_hidden_dim),
            nn.ReLU(),
            nn.Linear(core_hidden_dim, core_hidden_dim),
            nn.ReLU(),
        )

        # Action heads
        self.move_head = nn.Linear(core_hidden_dim, move_dim)
        self.task_head = nn.Linear(core_hidden_dim, task_dim)
        self.message_head = nn.Linear(core_hidden_dim, message_length * vocab_size)

    def forward(
        self,
        grid: torch.Tensor,
        state: torch.Tensor,
        messages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass producing logits for all three action heads.

        Args:
            grid:     (B, C, H, W) — padded local grid observation.
            state:    (B, STATE_DIM) — agent state vector.
            messages: (B, num_other_agents, message_length * vocab_size)

        Returns:
            move_logits:    (B, 5)
            task_logits:    (B, 6)
            message_logits: (B, message_length, vocab_size)
        """
        # Encode each modality
        grid_features = self.grid_encoder(grid)          # (B, 128)
        msg_features = self.message_encoder(messages)    # (B, 64)

        # Combine
        combined = torch.cat([grid_features, state, msg_features], dim=-1)
        core_out = self.core(combined)  # (B, 256)

        # Action heads
        move_logits = self.move_head(core_out)       # (B, 5)
        task_logits = self.task_head(core_out)       # (B, 6)
        msg_flat = self.message_head(core_out)       # (B, L * V)
        message_logits = msg_flat.reshape(
            -1, self.message_length, self.vocab_size,
        )  # (B, L, V)

        return move_logits, task_logits, message_logits

    def get_features(
        self,
        grid: torch.Tensor,
        state: torch.Tensor,
        messages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the shared feature representation (useful for curiosity module).

        Returns:
            (B, core_hidden_dim) — core features before action heads.
        """
        grid_features = self.grid_encoder(grid)
        msg_features = self.message_encoder(messages)
        combined = torch.cat([grid_features, state, msg_features], dim=-1)
        return self.core(combined)


# ============================================================
# Centralized Critic (Value Network)
# ============================================================

class CentralizedCritic(nn.Module):
    """
    Value network that observes the GLOBAL state during training (CTDE paradigm).

    This critic sees the full grid (not just a local window) plus all agents'
    states and messages. It outputs a single scalar V(s) used for GAE
    advantage estimation.

    Input: Full global state tensor from env.get_global_state()
           + all agents' state vectors + all messages
    Output: Scalar value estimate
    """

    def __init__(
        self,
        grid_size: int = 16,
        num_grid_channels: int = 10,  # NUM_CHANNELS (6) + num_agents (4)
        num_agents: int = 4,
        state_dim: int = STATE_DIM,
        message_length: int = 3,
        vocab_size: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.message_length = message_length
        self.vocab_size = vocab_size

        # CNN for global grid (larger than local — 16x16)
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(num_grid_channels, 32, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # 2x2
            nn.ReLU(),
        )
        cnn_flat = 64 * 2 * 2  # 256

        # All agents' states concatenated
        all_states_dim = num_agents * state_dim

        # All messages flattened
        all_msgs_dim = num_agents * message_length * vocab_size

        combined_dim = cnn_flat + all_states_dim + all_msgs_dim

        self.value_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        global_grid: torch.Tensor,
        all_states: torch.Tensor,
        all_messages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value estimate from global information.

        Args:
            global_grid:  (B, num_grid_channels, grid_size, grid_size)
            all_states:   (B, num_agents * STATE_DIM)
            all_messages: (B, num_agents * message_length * vocab_size)

        Returns:
            values: (B, 1) — value estimate.
        """
        grid_features = self.grid_cnn(global_grid)
        grid_flat = grid_features.reshape(grid_features.size(0), -1)  # (B, 256)

        combined = torch.cat([grid_flat, all_states, all_messages], dim=-1)
        return self.value_mlp(combined)


# ============================================================
# Agent Wrapper
# ============================================================

class SignalAgent(nn.Module):
    """
    Complete agent module combining policy and critic.

    This wraps PolicyNetwork + CentralizedCritic and provides convenience
    methods for action selection and value estimation.

    All agents SHARE weights (parameter sharing) but receive different
    observations based on their role and position.
    """

    def __init__(
        self,
        grid_size: int = 16,
        num_agents: int = 4,
        message_length: int = 3,
        vocab_size: int = 8,
        grid_feature_dim: int = 128,
        message_hidden_dim: int = 64,
        core_hidden_dim: int = 256,
    ):
        super().__init__()

        self.policy = PolicyNetwork(
            grid_feature_dim=grid_feature_dim,
            message_hidden_dim=message_hidden_dim,
            core_hidden_dim=core_hidden_dim,
            message_length=message_length,
            vocab_size=vocab_size,
            num_other_agents=num_agents - 1,
        )

        num_grid_channels = NUM_CHANNELS + num_agents  # 6 + 4 = 10
        self.critic = CentralizedCritic(
            grid_size=grid_size,
            num_grid_channels=num_grid_channels,
            num_agents=num_agents,
            message_length=message_length,
            vocab_size=vocab_size,
            hidden_dim=core_hidden_dim,
        )

        self.message_length = message_length
        self.vocab_size = vocab_size
        self.num_agents = num_agents

    def act(
        self,
        grid: torch.Tensor,
        state: torch.Tensor,
        messages: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select actions given local observations.

        Returns:
            move_idx, task_idx, msg_tokens, log_prob
        """
        from src.environment.spaces import sample_action_from_logits

        move_logits, task_logits, message_logits = self.policy(grid, state, messages)
        return sample_action_from_logits(
            move_logits, task_logits, message_logits,
            deterministic=deterministic,
        )

    def evaluate_value(
        self,
        global_grid: torch.Tensor,
        all_states: torch.Tensor,
        all_messages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value estimate from global state (training only).

        Returns:
            (B, 1) value estimate.
        """
        return self.critic(global_grid, all_states, all_messages)

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts for each component."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "policy_total": _count(self.policy),
            "policy_grid_encoder": _count(self.policy.grid_encoder),
            "policy_message_encoder": _count(self.policy.message_encoder),
            "policy_core": _count(self.policy.core),
            "policy_heads": (
                _count(self.policy.move_head)
                + _count(self.policy.task_head)
                + _count(self.policy.message_head)
            ),
            "critic_total": _count(self.critic),
            "total": _count(self),
        }
