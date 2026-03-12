#!/usr/bin/env python3
"""
Project SIGNAL — Post-Training Analysis Script.

Loads a trained checkpoint, runs evaluation episodes, and analyzes
the emergent communication protocol.

Usage:
    python3 scripts/analyze_language.py
    python3 scripts/analyze_language.py --checkpoint models/checkpoints/best.pt
    python3 scripts/analyze_language.py --episodes 50 --ablation
"""

import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from src.environment.crisisgrid import CrisisGrid, Severity, VISION_RADIUS
from src.environment.spaces import encode_observation
from src.agents.networks import SignalAgent
from src.training.buffer import RolloutBuffer
from src.training.runner import run_episode
from src.analysis.language import (
    compute_message_entropy,
    compute_token_frequencies,
    compute_role_communication_patterns,
    compute_mutual_information,
    generate_analysis_report,
)
from src.analysis.ablation import run_ablation, format_ablation_report


def collect_detailed_episodes(
    env: CrisisGrid,
    agent: SignalAgent,
    num_episodes: int = 20,
    seed_start: int = 20000,
) -> tuple[list[np.ndarray], list[dict], list[dict]]:
    """
    Run evaluation episodes and collect messages + context for analysis.

    Returns:
        all_messages: Flat list of per-step message arrays.
        all_contexts: Per-step context dicts for grounding analysis.
        episode_infos: Per-episode summary dicts.
    """
    buffer = RolloutBuffer(max_steps=env.max_steps, num_agents=env.num_agents)
    all_messages = []
    all_contexts = []
    episode_infos = []

    for i in range(num_episodes):
        info = run_episode(
            env, agent, buffer, icm=None,
            seed=seed_start + i, deterministic=True,
        )
        episode_infos.append(info)

        # Collect messages from this episode
        for msg_step in info["messages"]:
            all_messages.append(msg_step)

    return all_messages, all_contexts, episode_infos


def main():
    parser = argparse.ArgumentParser(description="Project SIGNAL — Language Analysis")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--ablation", action="store_true", help="Run communication ablation")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    # Setup
    env = CrisisGrid(grid_size=args.grid_size, max_steps=args.max_steps)
    agent = SignalAgent(grid_size=args.grid_size)

    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  Trained for {checkpoint.get('episode', '?')} episodes")
        if "eval_reward" in checkpoint:
            print(f"  Best eval reward: {checkpoint['eval_reward']:.1f}")
    except FileNotFoundError:
        print(f"No checkpoint found at {args.checkpoint}")
        print("Running analysis with untrained agent (baseline)...")

    print()

    # Collect episodes
    print(f"Running {args.episodes} evaluation episodes...")
    messages, contexts, infos = collect_detailed_episodes(
        env, agent, num_episodes=args.episodes,
    )
    print(f"  Collected {len(messages)} timesteps of messages")
    print()

    # Generate report
    report = generate_analysis_report(messages, infos, vocab_size=env.vocab_size)
    print(report)

    # Ablation study
    if args.ablation:
        print()
        print("Running ablation study (this may take a moment)...")
        ablation = run_ablation(env, agent, num_episodes=args.episodes)
        print(format_ablation_report(ablation))

    # Save raw data
    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
