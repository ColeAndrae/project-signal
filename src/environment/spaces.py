"""
Observation encoding and action decoding for Project SIGNAL.

This module handles the conversion between raw CrisisGrid environment
data (dicts of numpy arrays) and the flat/structured tensors expected
by the policy and value networks.

Key responsibilities:
    - Pad variable-size grid observations (Scout 7x7 vs others 5x5) to uniform size
    - Normalize all observation channels to [0, 1] range
    - Decode composite actions from network outputs (3 heads → env action dict)
    - Provide shape constants for network construction
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.environment.crisisgrid import (
    NUM_CHANNELS,
    VISION_RADIUS,
    Role,
)

# ============================================================
# Constants
# ============================================================

# All observations are padded to the Scout's vision size (largest)
MAX_VISION_RADIUS = VISION_RADIUS[Role.SCOUT]  # 3
MAX_VISION_SIZE = 2 * MAX_VISION_RADIUS + 1      # 7

# Fixed dimensions
STATE_DIM = 8
GRID_SHAPE = (MAX_VISION_SIZE, MAX_VISION_SIZE, NUM_CHANNELS)  # (7, 7, 6)


# ============================================================
# Observation Encoding
# ============================================================

def encode_grid_observation(
    grid: np.ndarray,
    role: Role,
) -> torch.Tensor:
    """
    Pad a local grid observation to MAX_VISION_SIZE and convert to tensor.

    Agents with smaller vision (5x5) get zero-padded borders.
    Output shape: (NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE) — channels-first for Conv2d.

    Args:
        grid: Raw grid observation, shape (V, V, C) where V varies by role.
        role: Agent role (determines original vision size).

    Returns:
        Tensor of shape (C, MAX_VISION_SIZE, MAX_VISION_SIZE), float32.
    """
    v = grid.shape[0]
    pad_total = MAX_VISION_SIZE - v

    if pad_total > 0:
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        # Pad spatial dims with -1.0 (out-of-bounds marker, distinct from 0.0 = empty)
        grid = np.pad(
            grid,
            ((pad_before, pad_after), (pad_before, pad_after), (0, 0)),
            mode="constant",
            constant_values=-1.0,
        )

    # Convert to channels-first: (H, W, C) → (C, H, W)
    tensor = torch.from_numpy(grid).float().permute(2, 0, 1)
    return tensor


def encode_state(state: np.ndarray) -> torch.Tensor:
    """
    Convert agent state vector to tensor.

    Args:
        state: Raw state array, shape (STATE_DIM,). Already normalized in env.

    Returns:
        Tensor of shape (STATE_DIM,), float32.
    """
    return torch.from_numpy(state).float()


def encode_messages(
    messages: np.ndarray,
    vocab_size: int,
) -> torch.Tensor:
    """
    One-hot encode incoming messages and flatten.

    Args:
        messages: Integer message array, shape (num_other_agents, message_length).
        vocab_size: Size of the discrete vocabulary.

    Returns:
        Tensor of shape (num_other_agents, message_length * vocab_size), float32.
    """
    num_agents, msg_len = messages.shape
    msg_tensor = torch.from_numpy(messages).long()

    # One-hot each token: (N, L) → (N, L, V)
    one_hot = F.one_hot(msg_tensor, num_classes=vocab_size).float()

    # Flatten per-agent: (N, L, V) → (N, L * V)
    return one_hot.reshape(num_agents, msg_len * vocab_size)


def encode_observation(
    obs: dict[str, np.ndarray],
    role: Role,
    vocab_size: int = 8,
) -> dict[str, torch.Tensor]:
    """
    Encode a complete per-agent observation into network-ready tensors.

    Args:
        obs: Raw observation dict from CrisisGrid with keys 'grid', 'state', 'messages'.
        role: Agent role enum.
        vocab_size: Communication vocabulary size.

    Returns:
        Dict with keys:
            'grid':     (C, MAX_VISION_SIZE, MAX_VISION_SIZE)
            'state':    (STATE_DIM,)
            'messages': (num_other_agents, message_length * vocab_size)
    """
    return {
        "grid": encode_grid_observation(obs["grid"], role),
        "state": encode_state(obs["state"]),
        "messages": encode_messages(obs["messages"], vocab_size),
    }


def batch_encode_observations(
    obs_list: list[dict[str, np.ndarray]],
    roles: list[Role],
    vocab_size: int = 8,
) -> dict[str, torch.Tensor]:
    """
    Encode and batch a list of observations (e.g., from all agents or across timesteps).

    Args:
        obs_list: List of raw observation dicts.
        roles: Corresponding role for each observation.
        vocab_size: Communication vocabulary size.

    Returns:
        Batched dict with keys:
            'grid':     (B, C, H, W)
            'state':    (B, STATE_DIM)
            'messages': (B, num_other_agents, message_length * vocab_size)
    """
    encoded = [encode_observation(o, r, vocab_size) for o, r in zip(obs_list, roles)]
    return {
        "grid": torch.stack([e["grid"] for e in encoded]),
        "state": torch.stack([e["state"] for e in encoded]),
        "messages": torch.stack([e["messages"] for e in encoded]),
    }


# ============================================================
# Action Decoding
# ============================================================

def decode_actions(
    move_indices: torch.Tensor,
    task_indices: torch.Tensor,
    message_tokens: torch.Tensor,
) -> list[dict[str, any]]:
    """
    Convert network output indices into environment action dicts.

    Args:
        move_indices:    (B,) int tensor — movement action per agent.
        task_indices:    (B,) int tensor — task action per agent.
        message_tokens:  (B, message_length) int tensor — message tokens per agent.

    Returns:
        List of action dicts, one per agent in batch:
        [{'move': int, 'task': int, 'message': list[int]}, ...]
    """
    batch_size = move_indices.shape[0]
    actions = []
    for i in range(batch_size):
        actions.append({
            "move": move_indices[i].item(),
            "task": task_indices[i].item(),
            "message": message_tokens[i].tolist(),
        })
    return actions


def sample_action_from_logits(
    move_logits: torch.Tensor,
    task_logits: torch.Tensor,
    message_logits: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample composite actions from the three policy heads.

    Args:
        move_logits:    (B, 5) — raw logits for movement.
        task_logits:    (B, 6) — raw logits for task.
        message_logits: (B, message_length, vocab_size) — raw logits per token.
        deterministic:  If True, take argmax instead of sampling.

    Returns:
        move_idx:   (B,) — sampled movement indices.
        task_idx:   (B,) — sampled task indices.
        msg_tokens: (B, message_length) — sampled message tokens.
        log_prob:   (B,) — total log probability of the composite action.
    """
    # Movement
    move_dist = torch.distributions.Categorical(logits=move_logits)
    move_idx = move_logits.argmax(dim=-1) if deterministic else move_dist.sample()
    move_logp = move_dist.log_prob(move_idx)

    # Task
    task_dist = torch.distributions.Categorical(logits=task_logits)
    task_idx = task_logits.argmax(dim=-1) if deterministic else task_dist.sample()
    task_logp = task_dist.log_prob(task_idx)

    # Message (independent per token position)
    B, L, V = message_logits.shape
    msg_flat = message_logits.reshape(B * L, V)
    msg_dist = torch.distributions.Categorical(logits=msg_flat)
    msg_flat_idx = msg_flat.argmax(dim=-1) if deterministic else msg_dist.sample()
    msg_flat_logp = msg_dist.log_prob(msg_flat_idx)

    msg_tokens = msg_flat_idx.reshape(B, L)
    msg_logp = msg_flat_logp.reshape(B, L).sum(dim=-1)  # sum across token positions

    # Total log-prob is the sum of independent factored heads
    total_log_prob = move_logp + task_logp + msg_logp

    return move_idx, task_idx, msg_tokens, total_log_prob


def compute_action_log_prob(
    move_logits: torch.Tensor,
    task_logits: torch.Tensor,
    message_logits: torch.Tensor,
    move_idx: torch.Tensor,
    task_idx: torch.Tensor,
    msg_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log probability and entropy for given actions (used in PPO update).

    Args:
        move_logits:    (B, 5)
        task_logits:    (B, 6)
        message_logits: (B, L, V)
        move_idx:       (B,)
        task_idx:       (B,)
        msg_tokens:     (B, L)

    Returns:
        log_prob: (B,) — total log probability of the composite action.
        entropy:  (B,) — total entropy across all action heads.
    """
    move_dist = torch.distributions.Categorical(logits=move_logits)
    task_dist = torch.distributions.Categorical(logits=task_logits)

    B, L, V = message_logits.shape
    msg_dist = torch.distributions.Categorical(logits=message_logits.reshape(B * L, V))

    move_logp = move_dist.log_prob(move_idx)
    task_logp = task_dist.log_prob(task_idx)
    msg_logp = msg_dist.log_prob(msg_tokens.reshape(B * L)).reshape(B, L).sum(dim=-1)

    log_prob = move_logp + task_logp + msg_logp

    move_ent = move_dist.entropy()
    task_ent = task_dist.entropy()
    msg_ent = msg_dist.entropy().reshape(B, L).sum(dim=-1)

    entropy = move_ent + task_ent + msg_ent

    return log_prob, entropy
