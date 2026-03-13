"""
Ablation experiments for Project SIGNAL.

The critical experiment: does emergent communication actually help?
We compare trained agents with their communication channel intact vs.
agents whose messages are replaced with zeros (silenced).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.environment.crisisgrid import CrisisGrid
from src.agents.networks import SignalAgent
from src.training.buffer import RolloutBuffer
from src.training.runner import run_episode


def run_ablation(
    env: CrisisGrid,
    agent: SignalAgent,
    num_episodes: int = 20,
    seed_start: int = 50000,
) -> dict[str, Any]:
    """
    Compare agent performance with and without communication.

    'With communication': normal evaluation (agents send and receive messages).
    'Without communication': messages are zeroed out before being fed to the policy.

    Args:
        env: CrisisGrid environment.
        agent: Trained SignalAgent.
        num_episodes: Number of episodes per condition.
        seed_start: Starting seed for reproducibility.

    Returns:
        Dict with per-condition metrics and statistical comparison.
    """
    buffer = RolloutBuffer(max_steps=env.max_steps, num_agents=env.num_agents, global_grid_size=env.grid_size)

    results = {"with_comm": [], "without_comm": []}

    # With communication (normal)
    for i in range(num_episodes):
        info = run_episode(
            env, agent, buffer, icm=None,
            seed=seed_start + i, deterministic=True,
        )
        results["with_comm"].append(info)

    # Without communication (silenced)
    original_forward = agent.policy.forward

    def silenced_forward(grid, state, messages):
        """Replace incoming messages with zeros."""
        silent_messages = torch.zeros_like(messages)
        return original_forward(grid, state, silent_messages)

    agent.policy.forward = silenced_forward

    for i in range(num_episodes):
        info = run_episode(
            env, agent, buffer, icm=None,
            seed=seed_start + i, deterministic=True,
        )
        results["without_comm"].append(info)

    # Restore original forward
    agent.policy.forward = original_forward

    # Compute statistics
    summary = {}
    for condition in ["with_comm", "without_comm"]:
        rewards = [e["episode_reward"] for e in results[condition]]
        rescued = [e["victims_rescued"] for e in results[condition]]
        dead = [e["victims_dead"] for e in results[condition]]
        summary[condition] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "rescued_mean": float(np.mean(rescued)),
            "rescued_std": float(np.std(rescued)),
            "dead_mean": float(np.mean(dead)),
        }

    # Improvement from communication
    r_with = summary["with_comm"]["reward_mean"]
    r_without = summary["without_comm"]["reward_mean"]
    summary["comm_reward_delta"] = r_with - r_without
    summary["comm_rescued_delta"] = (
        summary["with_comm"]["rescued_mean"] - summary["without_comm"]["rescued_mean"]
    )
    summary["comm_helps"] = r_with > r_without

    return summary


def format_ablation_report(summary: dict[str, Any]) -> str:
    """Format ablation results as a readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  ABLATION STUDY: Communication vs. No Communication")
    lines.append("=" * 70)
    lines.append("")

    for condition, label in [("with_comm", "WITH Communication"), ("without_comm", "WITHOUT Communication")]:
        s = summary[condition]
        lines.append(f"  {label}:")
        lines.append(f"    Reward:  {s['reward_mean']:7.1f} +/- {s['reward_std']:5.1f}")
        lines.append(f"    Rescued: {s['rescued_mean']:5.1f} +/- {s['rescued_std']:4.1f}")
        lines.append(f"    Dead:    {s['dead_mean']:5.1f}")
        lines.append("")

    lines.append(f"  DELTA (communication effect):")
    lines.append(f"    Reward improvement:  {summary['comm_reward_delta']:+.1f}")
    lines.append(f"    Rescued improvement: {summary['comm_rescued_delta']:+.1f}")
    lines.append("")

    if summary["comm_helps"]:
        lines.append("  CONCLUSION: Communication HELPS — emergent language is functional.")
    else:
        lines.append("  CONCLUSION: Communication does NOT help yet — language may need more training.")

    lines.append("=" * 70)
    return "\n".join(lines)
