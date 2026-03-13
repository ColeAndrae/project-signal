#!/usr/bin/env python3
"""
Project SIGNAL — Visualization & Report Generator.

Generates publication-quality plots and a comprehensive results report.

Usage:
    python3 scripts/visualize_results.py
    python3 scripts/visualize_results.py --checkpoint models/checkpoints/final.pt --grid-size 8
    python3 scripts/visualize_results.py --checkpoint models/checkpoints/final.pt --grid-size 8 --ablation
"""

import argparse
import sys
import os

sys.path.insert(0, ".")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from src.environment.crisisgrid import CrisisGrid
from src.agents.networks import SignalAgent
from src.agents.curiosity import IntrinsicCuriosityModule
from src.training.buffer import RolloutBuffer
from src.training.runner import run_episode
from src.analysis.language import (
    compute_message_entropy,
    compute_token_frequencies,
    compute_role_communication_patterns,
    generate_analysis_report,
)
from src.analysis.ablation import run_ablation

# ============================================================
# Style configuration
# ============================================================

COLORS = {
    "bg": "#0D1117",
    "surface": "#161B22",
    "border": "#30363D",
    "text": "#E6EDF3",
    "text_dim": "#8B949E",
    "accent": "#58A6FF",
    "green": "#3FB950",
    "red": "#F85149",
    "orange": "#D29922",
    "purple": "#BC8CFF",
    "cyan": "#39D2C0",
    "pink": "#F778BA",
}

ROLE_COLORS = {
    "Medic": "#F85149",
    "Engineer": "#D29922",
    "Scout": "#58A6FF",
    "Carrier": "#3FB950",
}

TOKEN_CMAP = LinearSegmentedColormap.from_list(
    "signal", ["#0D1117", "#58A6FF", "#39D2C0", "#3FB950"]
)


def apply_dark_style():
    """Apply dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["border"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text_dim"],
        "ytick.color": COLORS["text_dim"],
        "grid.color": COLORS["border"],
        "grid.alpha": 0.3,
        "font.family": "monospace",
        "font.size": 11,
    })


# ============================================================
# Plot generators
# ============================================================

def plot_training_curves(reward_history, rescued_history, output_dir):
    """Plot reward and rescue rate over training."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("PROJECT SIGNAL — Training Curves", fontsize=16, fontweight="bold", y=0.95)

    episodes = range(1, len(reward_history) + 1)

    # Smoothed reward
    window = min(50, len(reward_history) // 5)
    if window > 1:
        reward_smooth = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        ep_smooth = range(window, len(reward_history) + 1)
    else:
        reward_smooth = reward_history
        ep_smooth = episodes

    ax1.plot(episodes, reward_history, alpha=0.15, color=COLORS["accent"], linewidth=0.5)
    ax1.plot(ep_smooth, reward_smooth, color=COLORS["accent"], linewidth=2, label="Smoothed")
    ax1.set_ylabel("Episode Reward")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Smoothed rescued
    if window > 1:
        rescued_smooth = np.convolve(rescued_history, np.ones(window) / window, mode="valid")
    else:
        rescued_smooth = rescued_history

    ax2.plot(episodes, rescued_history, alpha=0.15, color=COLORS["green"], linewidth=0.5)
    ax2.plot(ep_smooth, rescued_smooth, color=COLORS["green"], linewidth=2, label="Smoothed")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Victims Rescued")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def plot_token_heatmap(messages, vocab_size, output_dir):
    """Plot token frequency heatmap per position."""
    freq = compute_token_frequencies(messages, vocab_size)
    freq_matrix = freq["frequencies"]  # (L, V)

    # Normalize per position
    row_sums = freq_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = freq_matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Token Frequency by Position", fontsize=14, fontweight="bold")

    im = ax.imshow(norm_matrix, cmap=TOKEN_CMAP, aspect="auto", vmin=0)
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Position in Message")
    ax.set_xticks(range(vocab_size))
    ax.set_yticks(range(freq_matrix.shape[0]))
    ax.set_yticklabels([f"Pos {i}" for i in range(freq_matrix.shape[0])])

    # Annotate cells
    for i in range(freq_matrix.shape[0]):
        for j in range(vocab_size):
            val = norm_matrix[i, j]
            color = COLORS["bg"] if val > 0.15 else COLORS["text_dim"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Frequency (normalized)")
    plt.tight_layout()
    path = os.path.join(output_dir, "token_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def plot_role_distributions(messages, output_dir):
    """Plot per-role token usage distributions."""
    roles = compute_role_communication_patterns(messages)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Communication Patterns by Role", fontsize=14, fontweight="bold", y=0.98)

    for ax, (role_name, data) in zip(axes.flat, roles.items()):
        dist = data["token_distribution"]
        total = sum(dist)
        pcts = [d / total * 100 if total > 0 else 0 for d in dist]
        bars = ax.bar(range(len(dist)), pcts, color=ROLE_COLORS[role_name], alpha=0.85, edgecolor=COLORS["border"])
        ax.set_title(f"{role_name} (H={data['entropy']:.2f} bits)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Token ID")
        ax.set_ylabel("Usage %")
        ax.set_xticks(range(len(dist)))
        ax.set_ylim(0, max(pcts) * 1.2 if pcts else 1)
        ax.grid(True, axis="y")

        for bar, pct in zip(bars, pcts):
            if pct > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=8, color=COLORS["text_dim"])

    plt.tight_layout()
    path = os.path.join(output_dir, "role_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def plot_top_messages(messages, vocab_size, output_dir):
    """Plot the most frequent messages."""
    freq = compute_token_frequencies(messages, vocab_size)
    top = freq["most_common_messages"][:10]
    total = freq["total_messages"]

    labels = [str(msg) for msg, _ in top]
    counts = [count for _, count in top]
    pcts = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Top 10 Most Frequent Messages", fontsize=14, fontweight="bold")

    bars = ax.barh(range(len(labels)), pcts, color=COLORS["cyan"], alpha=0.85, edgecolor=COLORS["border"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, fontfamily="monospace")
    ax.set_xlabel("Frequency (%)")
    ax.invert_yaxis()
    ax.grid(True, axis="x")

    for bar, pct, count in zip(bars, pcts, counts):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% ({count}x)", va="center", fontsize=9, color=COLORS["text_dim"])

    plt.tight_layout()
    path = os.path.join(output_dir, "top_messages.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def plot_ablation(ablation_summary, output_dir):
    """Plot ablation comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Communication Ablation Study", fontsize=14, fontweight="bold")

    conditions = ["With\nComm", "Without\nComm"]

    # Reward comparison
    rewards = [ablation_summary["with_comm"]["reward_mean"], ablation_summary["without_comm"]["reward_mean"]]
    reward_errs = [ablation_summary["with_comm"]["reward_std"], ablation_summary["without_comm"]["reward_std"]]
    bars1 = ax1.bar(conditions, rewards, yerr=reward_errs, capsize=5,
                    color=[COLORS["green"], COLORS["red"]], alpha=0.85, edgecolor=COLORS["border"])
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Reward", fontweight="bold")
    ax1.grid(True, axis="y")
    delta_r = ablation_summary["comm_reward_delta"]
    ax1.text(0.5, max(rewards) * 0.9, f"Delta: {delta_r:+.0f}", ha="center",
             fontsize=11, fontweight="bold", color=COLORS["accent"])

    # Rescued comparison
    rescued = [ablation_summary["with_comm"]["rescued_mean"], ablation_summary["without_comm"]["rescued_mean"]]
    rescued_errs = [ablation_summary["with_comm"]["rescued_std"], ablation_summary["without_comm"]["rescued_std"]]
    bars2 = ax2.bar(conditions, rescued, yerr=rescued_errs, capsize=5,
                    color=[COLORS["green"], COLORS["red"]], alpha=0.85, edgecolor=COLORS["border"])
    ax2.set_ylabel("Mean Rescued")
    ax2.set_title("Victims Rescued", fontweight="bold")
    ax2.grid(True, axis="y")
    delta_v = ablation_summary["comm_rescued_delta"]
    ax2.text(0.5, max(rescued) * 0.9 if max(rescued) > 0 else 0.5, f"Delta: {delta_v:+.1f}", ha="center",
             fontsize=11, fontweight="bold", color=COLORS["accent"])

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_study.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def plot_entropy_summary(messages, vocab_size, output_dir):
    """Plot entropy gauge."""
    entropy = compute_message_entropy(messages, vocab_size)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Message Entropy Analysis", fontsize=14, fontweight="bold")

    positions = range(len(entropy["per_position"]))
    bars = ax.bar(positions, entropy["per_position"], color=COLORS["purple"], alpha=0.85, edgecolor=COLORS["border"])
    ax.axhline(y=entropy["max_entropy"], color=COLORS["red"], linestyle="--", alpha=0.5, label=f"Max ({entropy['max_entropy']:.1f} bits)")
    ax.axhline(y=entropy["mean_entropy"], color=COLORS["accent"], linestyle="-", alpha=0.7, label=f"Mean ({entropy['mean_entropy']:.2f} bits)")

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Entropy (bits)")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Pos {i}" for i in positions])
    ax.legend()
    ax.grid(True, axis="y")

    ax.text(0.98, 0.95, f"Uniformity: {entropy['uniformity']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            fontweight="bold", color=COLORS["cyan"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["surface"], edgecolor=COLORS["border"]))

    plt.tight_layout()
    path = os.path.join(output_dir, "entropy_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ============================================================
# Report generator
# ============================================================

def generate_markdown_report(
    messages, infos, ablation_summary, entropy, freq, checkpoint_info, output_dir
):
    """Generate a comprehensive markdown report with embedded image references."""
    lines = []
    lines.append("# Project SIGNAL — Results Report")
    lines.append("")
    lines.append("## Training Summary")
    lines.append("")
    lines.append(f"- **Episodes trained:** {checkpoint_info.get('episode', 'N/A')}")
    lines.append(f"- **Best eval reward:** {checkpoint_info.get('eval_reward', 0):.1f}")
    lines.append(f"- **Total parameters:** 626,022 (policy) + 158,976 (ICM)")
    lines.append("")
    lines.append("![Training Curves](training_curves.png)")
    lines.append("")

    lines.append("## Emergent Language Analysis")
    lines.append("")
    lines.append(f"- **Mean entropy:** {entropy['mean_entropy']:.3f} / {entropy['max_entropy']:.3f} bits")
    lines.append(f"- **Uniformity ratio:** {entropy['uniformity']:.3f}")
    lines.append(f"- **Unique messages:** {freq['unique_messages']} / {freq['total_messages']}")
    lines.append(f"- **Vocabulary utilization:** {freq['unique_messages'] / 512 * 100:.1f}% of 512 possible")
    lines.append("")
    lines.append("![Entropy Analysis](entropy_analysis.png)")
    lines.append("")
    lines.append("![Token Heatmap](token_heatmap.png)")
    lines.append("")
    lines.append("![Top Messages](top_messages.png)")
    lines.append("")

    lines.append("## Role Communication Patterns")
    lines.append("")
    lines.append("Each agent role develops distinct communication behavior:")
    lines.append("")
    lines.append("![Role Distributions](role_distributions.png)")
    lines.append("")

    if ablation_summary:
        lines.append("## Ablation Study: Does Communication Help?")
        lines.append("")
        helps = "YES" if ablation_summary["comm_helps"] else "NO"
        lines.append(f"- **Communication helps:** {helps}")
        lines.append(f"- **Reward delta:** {ablation_summary['comm_reward_delta']:+.1f}")
        lines.append(f"- **Rescued delta:** {ablation_summary['comm_rescued_delta']:+.1f}")
        lines.append("")
        lines.append("![Ablation Study](ablation_study.png)")
        lines.append("")

    if infos:
        rewards = [e.get("episode_reward", 0) for e in infos]
        rescued = [e.get("victims_rescued", 0) for e in infos]
        dead = [e.get("victims_dead", 0) for e in infos]
        lines.append("## Evaluation Performance")
        lines.append("")
        lines.append(f"- **Mean reward:** {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        lines.append(f"- **Mean rescued:** {np.mean(rescued):.1f}")
        lines.append(f"- **Mean dead:** {np.mean(dead):.1f}")
        total_v = np.mean([e.get("victims_rescued", 0) + e.get("victims_dead", 0) + e.get("victims_alive", 0) for e in infos])
        if total_v > 0:
            lines.append(f"- **Survival rate:** {np.mean(rescued) / total_v * 100:.1f}%")
        lines.append("")

    report_text = "\n".join(lines)
    path = os.path.join(output_dir, "RESULTS.md")
    with open(path, "w") as f:
        f.write(report_text)
    print(f"  [SAVED] {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Project SIGNAL — Visualization")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/final.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--output-dir", type=str, default="docs/figures")
    args = parser.parse_args()

    apply_dark_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load environment and agent
    env = CrisisGrid(grid_size=args.grid_size, max_steps=args.max_steps)
    agent = SignalAgent(grid_size=args.grid_size)

    checkpoint_info = {}
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        checkpoint_info = checkpoint
        print(f"Loaded: {args.checkpoint} (episode {checkpoint.get('episode', '?')})")
    except FileNotFoundError:
        print(f"No checkpoint at {args.checkpoint}, using untrained agent")

    # Plot training curves if available
    if "reward_history" in checkpoint_info:
        print("\nGenerating training curves...")
        plot_training_curves(
            checkpoint_info["reward_history"],
            checkpoint_info.get("rescued_history", [0] * len(checkpoint_info["reward_history"])),
            args.output_dir,
        )

    # Collect evaluation episodes
    print(f"\nRunning {args.episodes} evaluation episodes...")
    buffer = RolloutBuffer(max_steps=env.max_steps, num_agents=env.num_agents, global_grid_size=env.grid_size)
    all_messages = []
    infos = []
    for i in range(args.episodes):
        info = run_episode(env, agent, buffer, icm=None, seed=20000 + i, deterministic=True)
        infos.append(info)
        all_messages.extend(info["messages"])
    print(f"  Collected {len(all_messages)} timesteps")

    # Generate plots
    print("\nGenerating plots...")
    entropy = compute_message_entropy(all_messages, vocab_size=env.vocab_size)
    freq = compute_token_frequencies(all_messages, vocab_size=env.vocab_size)

    plot_entropy_summary(all_messages, env.vocab_size, args.output_dir)
    plot_token_heatmap(all_messages, env.vocab_size, args.output_dir)
    plot_role_distributions(all_messages, args.output_dir)
    plot_top_messages(all_messages, env.vocab_size, args.output_dir)

    # Ablation
    ablation_summary = None
    if args.ablation:
        print("\nRunning ablation study...")
        ablation_summary = run_ablation(env, agent, num_episodes=args.episodes)
        plot_ablation(ablation_summary, args.output_dir)

    # Markdown report
    print("\nGenerating report...")
    generate_markdown_report(all_messages, infos, ablation_summary, entropy, freq, checkpoint_info, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
