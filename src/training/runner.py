"""
Episode runner for Project SIGNAL.

Orchestrates the interaction between environment, agents, and buffer:
    1. Reset environment
    2. For each timestep: encode obs → agent.act() → env.step() → buffer.insert()
    3. Compute GAE returns
    4. Hand buffer to trainer

Also handles intrinsic reward injection from the ICM module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.environment.crisisgrid import CrisisGrid, NUM_CHANNELS
from src.environment.spaces import (
    encode_observation,
    MAX_VISION_SIZE,
    STATE_DIM,
)
from src.agents.networks import SignalAgent
from src.agents.curiosity import IntrinsicCuriosityModule
from src.training.buffer import RolloutBuffer


def _encode_all_agents(
    obs: dict[int, dict[str, np.ndarray]],
    env: CrisisGrid,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode observations for all agents into batched tensors.

    Returns:
        grids:    (A, C, H, W)
        states:   (A, state_dim)
        messages: (A, num_other, msg_flat)
    """
    all_grids = []
    all_states = []
    all_msgs = []

    for i in range(env.num_agents):
        enc = encode_observation(obs[i], env.agents[i].role, vocab_size)
        all_grids.append(enc["grid"])
        all_states.append(enc["state"])
        all_msgs.append(enc["messages"])

    return (
        torch.stack(all_grids),
        torch.stack(all_states),
        torch.stack(all_msgs),
    )


def _build_global_critic_inputs(
    env: CrisisGrid,
    agent_states: torch.Tensor,
    raw_messages: np.ndarray,
    vocab_size: int,
    message_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the centralized critic's inputs from the environment's global state.

    Returns:
        global_grid:    (C_global, H, W)
        all_states:     (A * state_dim,)
        all_messages:   (A * msg_len * vocab,)
    """
    # Global grid: (H, W, C) → (C, H, W)
    global_state = env.get_global_state()
    global_grid = torch.from_numpy(global_state).float().permute(2, 0, 1)

    # All states flattened
    all_states = agent_states.reshape(-1)

    # All messages one-hot and flattened
    msg_tensor = torch.from_numpy(raw_messages).long()  # (A, msg_len)
    msg_onehot = F.one_hot(msg_tensor, num_classes=vocab_size).float()  # (A, L, V)
    all_messages = msg_onehot.reshape(-1)  # (A * L * V,)

    return global_grid, all_states, all_messages


def run_episode(
    env: CrisisGrid,
    agent: SignalAgent,
    buffer: RolloutBuffer,
    icm: IntrinsicCuriosityModule | None = None,
    seed: int | None = None,
    deterministic: bool = False,
) -> dict[str, Any]:
    """
    Run a single episode, collecting data into the buffer.

    Args:
        env: The CrisisGrid environment.
        agent: The shared SignalAgent (policy + critic).
        buffer: RolloutBuffer to fill.
        icm: Optional ICM for intrinsic rewards.
        seed: Random seed for the episode.
        deterministic: If True, take greedy actions (for evaluation).

    Returns:
        Episode info dict with metrics.
    """
    buffer.reset()
    obs = env.reset(seed=seed)
    vocab_size = env.vocab_size
    message_length = env.message_length

    episode_reward = 0.0
    episode_intrinsic_reward = 0.0
    episode_steps = 0

    # Track messages sent (for analysis)
    all_episode_messages = []

    # Raw message buffer (what agents sent last step)
    raw_messages = np.zeros((env.num_agents, message_length), dtype=np.int64)

    # Encode initial observations
    grids, states, messages_in = _encode_all_agents(obs, env, vocab_size)

    done = False

    while not done:
        with torch.no_grad():
            # Get actions from policy
            move_idx, task_idx, msg_tokens, log_probs = agent.act(
                grids, states, messages_in, deterministic=deterministic,
            )

            # Get value estimate from centralized critic
            global_grid, all_states_flat, all_msgs_flat = _build_global_critic_inputs(
                env, states, raw_messages, vocab_size, message_length,
            )
            value = agent.evaluate_value(
                global_grid.unsqueeze(0),
                all_states_flat.unsqueeze(0),
                all_msgs_flat.unsqueeze(0),
            ).item()

        # Build env actions
        actions = {}
        new_raw_messages = np.zeros_like(raw_messages)
        for i in range(env.num_agents):
            msg = msg_tokens[i].numpy()
            new_raw_messages[i] = msg
            actions[i] = {
                "move": move_idx[i].item(),
                "task": task_idx[i].item(),
                "message": msg.tolist(),
            }

        all_episode_messages.append(new_raw_messages.copy())

        # Step environment
        next_obs, reward, done, info = env.step(actions)

        # Encode next observations
        next_grids, next_states, next_messages_in = _encode_all_agents(
            next_obs, env, vocab_size,
        )

        # Compute intrinsic reward
        intrinsic_reward = 0.0
        if icm is not None:
            with torch.no_grad():
                # Average intrinsic reward across agents
                msg_for_icm = messages_in.mean(dim=1)[:, :24]
                ir = icm.compute_intrinsic_reward(
                    grids, states, next_grids, next_states,
                    msg_for_icm, normalize=True,
                )
                intrinsic_reward = ir.mean().item()
                episode_intrinsic_reward += intrinsic_reward

        total_reward = reward + intrinsic_reward

        # Store transition
        buffer.insert(
            grids=grids,
            states=states,
            messages_in=messages_in,
            moves=move_idx,
            tasks=task_idx,
            msg_tokens=msg_tokens,
            log_probs=log_probs,
            global_grid=global_grid,
            all_states=all_states_flat,
            all_messages=all_msgs_flat,
            reward=total_reward,
            done=done,
            value=value,
        )
        buffer.insert_next_obs(next_grids, next_states)

        # Advance
        obs = next_obs
        grids, states, messages_in = next_grids, next_states, next_messages_in
        raw_messages = new_raw_messages
        episode_reward += reward
        episode_steps += 1

    # Compute GAE returns (terminal state value = 0)
    buffer.compute_returns(last_value=0.0, gamma=0.99, gae_lambda=0.95)

    return {
        "episode_reward": episode_reward,
        "episode_intrinsic_reward": episode_intrinsic_reward,
        "episode_steps": episode_steps,
        "victims_rescued": env.rescued_victim_count(),
        "victims_dead": env.dead_victim_count(),
        "victims_alive": env.alive_victim_count(),
        "messages": all_episode_messages,
    }


def evaluate(
    env: CrisisGrid,
    agent: SignalAgent,
    num_episodes: int = 10,
    seed_start: int = 10000,
) -> dict[str, float]:
    """
    Evaluate agent performance over multiple episodes (no training).

    Returns:
        Averaged metrics dict.
    """
    buffer = RolloutBuffer(max_steps=env.max_steps, num_agents=env.num_agents, global_grid_size=env.grid_size)
    metrics = {
        "reward": [], "rescued": [], "dead": [], "steps": [],
    }

    for i in range(num_episodes):
        info = run_episode(
            env, agent, buffer, icm=None,
            seed=seed_start + i, deterministic=True,
        )
        metrics["reward"].append(info["episode_reward"])
        metrics["rescued"].append(info["victims_rescued"])
        metrics["dead"].append(info["victims_dead"])
        metrics["steps"].append(info["episode_steps"])

    return {
        "eval_reward_mean": np.mean(metrics["reward"]),
        "eval_reward_std": np.std(metrics["reward"]),
        "eval_rescued_mean": np.mean(metrics["rescued"]),
        "eval_dead_mean": np.mean(metrics["dead"]),
        "eval_steps_mean": np.mean(metrics["steps"]),
    }
