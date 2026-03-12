"""
Rollout buffer for multi-agent PPO training.

Stores transitions from parallel episode collection and computes GAE advantages.
Each entry stores data for ALL agents at a single timestep, enabling the
centralized critic to compute values from global state while each agent's
policy data remains separate.

Storage layout (per timestep):
    - Per-agent: grid, state, messages, move, task, msg_tokens, log_prob
    - Global: global_grid, all_states, all_messages, reward, done, value
"""

from __future__ import annotations

import torch
import numpy as np


class RolloutBuffer:
    """
    Fixed-size buffer that stores one episode's transitions for all agents.

    Usage:
        buf = RolloutBuffer(max_steps=200, num_agents=4, ...)
        for each step:
            buf.insert(...)
        buf.compute_returns(last_value, gamma, gae_lambda)
        batches = buf.generate_batches(mini_batch_size)
    """

    def __init__(
        self,
        max_steps: int = 200,
        num_agents: int = 4,
        grid_channels: int = 6,
        grid_size: int = 7,        # MAX_VISION_SIZE (padded)
        state_dim: int = 8,
        num_other_agents: int = 3,
        message_flat_dim: int = 24, # message_length * vocab_size
        message_length: int = 3,
        global_grid_channels: int = 10,
        global_grid_size: int = 16,
        vocab_size: int = 8,
    ):
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.message_length = message_length
        self.vocab_size = vocab_size
        self.ptr = 0

        N = max_steps
        A = num_agents

        # Per-agent observations (stored flat: timestep × agent)
        self.grids = torch.zeros(N, A, grid_channels, grid_size, grid_size)
        self.states = torch.zeros(N, A, state_dim)
        self.messages_in = torch.zeros(N, A, num_other_agents, message_flat_dim)

        # Per-agent actions
        self.moves = torch.zeros(N, A, dtype=torch.long)
        self.tasks = torch.zeros(N, A, dtype=torch.long)
        self.msg_tokens = torch.zeros(N, A, message_length, dtype=torch.long)
        self.log_probs = torch.zeros(N, A)

        # Global (shared across agents at each timestep)
        self.global_grids = torch.zeros(N, global_grid_channels, global_grid_size, global_grid_size)
        self.all_agent_states = torch.zeros(N, A * state_dim)
        self.all_agent_messages = torch.zeros(N, A * message_length * vocab_size)
        self.rewards = torch.zeros(N)
        self.dones = torch.zeros(N)
        self.values = torch.zeros(N)

        # Computed after episode
        self.returns = torch.zeros(N)
        self.advantages = torch.zeros(N)

        # For ICM: store next-step observations
        self.next_grids = torch.zeros(N, A, grid_channels, grid_size, grid_size)
        self.next_states = torch.zeros(N, A, state_dim)

    def insert(
        self,
        grids: torch.Tensor,           # (A, C, H, W)
        states: torch.Tensor,          # (A, state_dim)
        messages_in: torch.Tensor,     # (A, num_other, msg_flat)
        moves: torch.Tensor,           # (A,)
        tasks: torch.Tensor,           # (A,)
        msg_tokens: torch.Tensor,      # (A, msg_len)
        log_probs: torch.Tensor,       # (A,)
        global_grid: torch.Tensor,     # (C_global, H_global, W_global)
        all_states: torch.Tensor,      # (A * state_dim,)
        all_messages: torch.Tensor,    # (A * msg_len * vocab,)
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Insert one timestep of data for all agents."""
        t = self.ptr
        self.grids[t] = grids
        self.states[t] = states
        self.messages_in[t] = messages_in
        self.moves[t] = moves
        self.tasks[t] = tasks
        self.msg_tokens[t] = msg_tokens
        self.log_probs[t] = log_probs
        self.global_grids[t] = global_grid
        self.all_agent_states[t] = all_states
        self.all_agent_messages[t] = all_messages
        self.rewards[t] = reward
        self.dones[t] = float(done)
        self.values[t] = value
        self.ptr += 1

    def insert_next_obs(
        self,
        next_grids: torch.Tensor,     # (A, C, H, W)
        next_states: torch.Tensor,    # (A, state_dim)
    ) -> None:
        """Insert next-step observations for ICM (called after insert)."""
        t = self.ptr - 1
        if t >= 0:
            self.next_grids[t] = next_grids
            self.next_states[t] = next_states

    @property
    def size(self) -> int:
        return self.ptr

    def compute_returns(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_value: V(s_T) — value estimate of the state after the last stored step.
            gamma: Discount factor.
            gae_lambda: GAE lambda for bias-variance tradeoff.
        """
        T = self.ptr
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t].item()
            else:
                next_value = self.values[t + 1].item()
                next_non_terminal = 1.0 - self.dones[t].item()

            delta = (
                self.rewards[t].item()
                + gamma * next_value * next_non_terminal
                - self.values[t].item()
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t].item()

    def generate_batches(
        self, mini_batch_size: int = 64,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Generate randomized mini-batches for PPO updates.

        Flattens (timestep, agent) into a single batch dimension,
        then splits into mini-batches.

        Returns:
            List of dicts, each containing tensors for one mini-batch.
        """
        T = self.ptr
        A = self.num_agents
        total = T * A

        # Expand shared reward/advantage/return to per-agent
        rewards_expanded = self.rewards[:T].unsqueeze(1).expand(T, A).reshape(total)
        returns_expanded = self.returns[:T].unsqueeze(1).expand(T, A).reshape(total)
        advantages_expanded = self.advantages[:T].unsqueeze(1).expand(T, A).reshape(total)
        values_expanded = self.values[:T].unsqueeze(1).expand(T, A).reshape(total)

        # Flatten time × agents
        flat = {
            "grids": self.grids[:T].reshape(total, *self.grids.shape[2:]),
            "states": self.states[:T].reshape(total, *self.states.shape[2:]),
            "messages_in": self.messages_in[:T].reshape(total, *self.messages_in.shape[2:]),
            "moves": self.moves[:T].reshape(total),
            "tasks": self.tasks[:T].reshape(total),
            "msg_tokens": self.msg_tokens[:T].reshape(total, self.message_length),
            "log_probs": self.log_probs[:T].reshape(total),
            "returns": returns_expanded,
            "advantages": advantages_expanded,
            "values": values_expanded,
            # Global state (repeated per agent for critic input)
            "global_grids": self.global_grids[:T].unsqueeze(1).expand(
                T, A, *self.global_grids.shape[1:]
            ).reshape(total, *self.global_grids.shape[1:]),
            "all_agent_states": self.all_agent_states[:T].unsqueeze(1).expand(
                T, A, self.all_agent_states.shape[1]
            ).reshape(total, self.all_agent_states.shape[1]),
            "all_agent_messages": self.all_agent_messages[:T].unsqueeze(1).expand(
                T, A, self.all_agent_messages.shape[1]
            ).reshape(total, self.all_agent_messages.shape[1]),
            # ICM data
            "next_grids": self.next_grids[:T].reshape(total, *self.next_grids.shape[2:]),
            "next_states": self.next_states[:T].reshape(total, *self.next_states.shape[2:]),
        }

        # Normalize advantages
        adv = flat["advantages"]
        adv_std = adv.std()
        if adv_std > 1e-8:
            flat["advantages"] = (adv - adv.mean()) / (adv_std + 1e-8)

        # Shuffle and split
        indices = torch.randperm(total)
        batches = []
        for start in range(0, total, mini_batch_size):
            end = min(start + mini_batch_size, total)
            idx = indices[start:end]
            batch = {k: v[idx] for k, v in flat.items()}
            batches.append(batch)

        return batches

    def reset(self) -> None:
        """Reset the buffer for a new episode."""
        self.ptr = 0
