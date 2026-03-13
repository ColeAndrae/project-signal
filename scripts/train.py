#!/usr/bin/env python3
"""
Project SIGNAL — Training Script.

Usage:
    python3 scripts/train.py --demo                # random agent demo
    python3 scripts/train.py                       # train with defaults
    python3 scripts/train.py --episodes 2000       # custom episode count
    python3 scripts/train.py --no-curiosity        # disable ICM
    python3 scripts/train.py --eval-only --checkpoint models/checkpoints/best.pt
"""

import argparse
import sys
import os
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from src.environment.crisisgrid import CrisisGrid
from src.agents.networks import SignalAgent
from src.agents.curiosity import IntrinsicCuriosityModule
from src.training.buffer import RolloutBuffer
from src.training.mappo import MAPPOTrainer
from src.training.runner import run_episode, evaluate


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


def train(args: argparse.Namespace) -> None:
    """Main training loop."""
    print("=" * 60)
    print("  Project SIGNAL — Training")
    print("=" * 60)

    # Environment
    env = CrisisGrid(
        grid_size=args.grid_size,
        num_victims=args.num_victims,
        max_steps=args.max_steps,
    )

    # Agent (shared weights across all 4 agents)
    agent = SignalAgent(
        grid_size=args.grid_size,
        num_agents=env.num_agents,
        global_grid_size=args.grid_size,
        message_length=env.message_length,
        vocab_size=env.vocab_size,
    )

    param_counts = agent.count_parameters()
    print(f"  Agent parameters: {param_counts['total']:,}")

    # ICM (optional)
    icm = None
    if args.use_curiosity:
        icm = IntrinsicCuriosityModule(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            anneal_episodes=args.anneal_episodes,
        )
        print(f"  ICM parameters: {icm.count_parameters()['total']:,}")
        print(f"  Curiosity beta: {args.beta_start} -> {args.beta_end} over {args.anneal_episodes} eps")
    else:
        print("  Curiosity: DISABLED")

    # Trainer
    trainer = MAPPOTrainer(
        agent=agent,
        icm=icm,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        ppo_clip=args.ppo_clip,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
    )

    # Buffer
    buffer = RolloutBuffer(
        max_steps=args.max_steps,
        num_agents=env.num_agents,
        global_grid_size=args.grid_size,
    )

    # Tracking
    best_eval_reward = float("-inf")
    reward_history = []
    rescued_history = []

    print(f"  Episodes: {args.episodes}")
    print(f"  Eval every: {args.eval_interval} episodes")
    print()

    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        # Collect episode
        ep_info = run_episode(
            env, agent, buffer, icm=icm,
            seed=ep,
        )

        # Train
        train_metrics = trainer.update(buffer)

        # Update curiosity beta
        if icm is not None:
            icm.update_beta(ep)

        # Track
        reward_history.append(ep_info["episode_reward"])
        rescued_history.append(ep_info["victims_rescued"])

        # Log
        if ep % args.log_interval == 0:
            recent_reward = np.mean(reward_history[-args.log_interval:])
            recent_rescued = np.mean(rescued_history[-args.log_interval:])
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed

            beta_str = f"  beta={icm.beta:.3f}" if icm else ""
            print(
                f"  Ep {ep:5d} | "
                f"R={recent_reward:7.1f} | "
                f"Rescued={recent_rescued:4.1f} | "
                f"Dead={ep_info['victims_dead']:2d} | "
                f"PL={train_metrics['policy_loss']:6.3f} | "
                f"VL={train_metrics['value_loss']:6.3f} | "
                f"Ent={train_metrics['entropy']:5.2f} | "
                f"Clip={train_metrics['clip_fraction']:4.2f}"
                f"{beta_str} | "
                f"{eps_per_sec:.1f} ep/s"
            )

        # Evaluate
        if ep % args.eval_interval == 0:
            eval_metrics = evaluate(env, agent, num_episodes=args.eval_episodes)
            print(
                f"  >>> EVAL | "
                f"Reward={eval_metrics['eval_reward_mean']:7.1f} +/- {eval_metrics['eval_reward_std']:5.1f} | "
                f"Rescued={eval_metrics['eval_rescued_mean']:4.1f} | "
                f"Dead={eval_metrics['eval_dead_mean']:4.1f}"
            )

            # Save best model
            if eval_metrics["eval_reward_mean"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval_reward_mean"]
                os.makedirs("models/checkpoints", exist_ok=True)
                torch.save({
                    "episode": ep,
                    "agent_state_dict": agent.state_dict(),
                    "icm_state_dict": icm.state_dict() if icm else None,
                    "eval_reward": best_eval_reward,
                }, "models/checkpoints/best.pt")
                print(f"  >>> Saved new best model (reward={best_eval_reward:.1f})")

    # Final save
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save({
        "episode": args.episodes,
        "agent_state_dict": agent.state_dict(),
        "icm_state_dict": icm.state_dict() if icm else None,
        "reward_history": reward_history,
        "rescued_history": rescued_history,
    }, "models/checkpoints/final.pt")

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  Training complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Best eval reward: {best_eval_reward:.1f}")
    print(f"  Final rescued rate: {np.mean(rescued_history[-50:]):.1f}")
    print(f"  Checkpoints saved to models/checkpoints/")
    print("=" * 60)


def eval_checkpoint(args: argparse.Namespace) -> None:
    """Load a checkpoint and evaluate."""
    env = CrisisGrid(grid_size=args.grid_size, max_steps=args.max_steps)
    agent = SignalAgent(grid_size=args.grid_size)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    print(f"Loaded checkpoint from episode {checkpoint.get('episode', '?')}")

    metrics = evaluate(env, agent, num_episodes=args.eval_episodes)
    print(f"Reward:  {metrics['eval_reward_mean']:.1f} +/- {metrics['eval_reward_std']:.1f}")
    print(f"Rescued: {metrics['eval_rescued_mean']:.1f}")
    print(f"Dead:    {metrics['eval_dead_mean']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Project SIGNAL Training")

    # Mode
    parser.add_argument("--demo", action="store_true", help="Run random-agent demo")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best.pt")

    # Environment
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--num-victims", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=200)

    # Training
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=64)

    # Curiosity
    parser.add_argument("--no-curiosity", dest="use_curiosity", action="store_false")
    parser.add_argument("--beta-start", type=float, default=0.5)
    parser.add_argument("--beta-end", type=float, default=0.01)
    parser.add_argument("--anneal-episodes", type=int, default=1500)

    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)

    parser.set_defaults(use_curiosity=True)
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.eval_only:
        eval_checkpoint(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
