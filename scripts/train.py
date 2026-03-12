#!/usr/bin/env python3
"""
Project SIGNAL — Demo: Random agents in CrisisGrid.

Run this to verify the environment works and see ASCII rendering.
Usage:
    python scripts/train.py --demo
    python scripts/train.py --demo --seed 42 --episodes 3
"""

import argparse
import sys
sys.path.insert(0, ".")

import numpy as np
from src.environment.crisisgrid import CrisisGrid


def run_demo(seed: int = 42, episodes: int = 1, render_every: int = 50) -> None:
    """Run episodes with random agents and print metrics."""
    env = CrisisGrid(max_steps=200)
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        step = 0

        print(f"\n{'='*60}")
        print(f"  EPISODE {ep + 1}/{episodes}")
        print(f"{'='*60}")
        env.render()

        while not done:
            # Random actions
            actions = {}
            for i in range(env.num_agents):
                actions[i] = {
                    "move": int(rng.integers(0, 5)),
                    "task": int(rng.integers(0, 6)),
                    "message": rng.integers(0, env.vocab_size, size=env.message_length).tolist(),
                }
            obs, reward, done, info = env.step(actions)
            total_reward += reward
            step += 1

            if step % render_every == 0 or done:
                print(f"\n--- Step {step} ---")
                env.render()

        print(f"\n  EPISODE SUMMARY")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {total_reward:.1f}")
        print(f"  Rescued: {env.rescued_victim_count()}")
        print(f"  Dead: {env.dead_victim_count()}")
        print(f"  Still Alive: {env.alive_victim_count()}")


def main():
    parser = argparse.ArgumentParser(description="Project SIGNAL Training / Demo")
    parser.add_argument("--demo", action="store_true", help="Run random-agent demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    args = parser.parse_args()

    if args.demo:
        run_demo(seed=args.seed, episodes=args.episodes)
    else:
        print("Training mode not yet implemented. Use --demo for environment demo.")
        print("Next phase: implement networks.py and mappo.py")


if __name__ == "__main__":
    main()
