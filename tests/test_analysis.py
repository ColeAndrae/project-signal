"""
Tests for Phase 5: Analysis tools (language.py, ablation.py).

Run from project root:
    python3 tests/test_analysis.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

from src.environment.crisisgrid import CrisisGrid
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


def _make_messages(n_steps: int = 100, n_agents: int = 4, msg_len: int = 3, vocab: int = 8) -> list[np.ndarray]:
    """Helper to generate random message logs."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, vocab, size=(n_agents, msg_len)) for _ in range(n_steps)]


def test_message_entropy_random():
    """Random messages should have high entropy (near uniform)."""
    messages = _make_messages(500)
    result = compute_message_entropy(messages, vocab_size=8)
    assert result["uniformity"] > 0.7, f"Random messages should be near-uniform, got {result['uniformity']:.3f}"
    assert len(result["per_position"]) == 3
    print(f"[PASS] test_message_entropy_random — uniformity={result['uniformity']:.3f}")


def test_message_entropy_collapsed():
    """Constant messages should have zero entropy."""
    messages = [np.zeros((4, 3), dtype=int) for _ in range(100)]
    result = compute_message_entropy(messages, vocab_size=8)
    assert result["mean_entropy"] == 0.0, f"Constant messages should have 0 entropy, got {result['mean_entropy']}"
    assert result["uniformity"] == 0.0
    print("[PASS] test_message_entropy_collapsed")


def test_token_frequencies():
    """Should count tokens and find most common messages."""
    messages = _make_messages(200)
    result = compute_token_frequencies(messages, vocab_size=8)
    assert result["frequencies"].shape == (3, 8), f"Expected (3, 8), got {result['frequencies'].shape}"
    assert result["total_messages"] == 200 * 4
    assert result["unique_messages"] > 0
    assert len(result["most_common_messages"]) <= 10
    print(f"[PASS] test_token_frequencies — {result['unique_messages']} unique messages")


def test_role_patterns():
    """Each role should get its own analysis."""
    messages = _make_messages(100)
    result = compute_role_communication_patterns(messages)
    assert "Medic" in result
    assert "Scout" in result
    assert "entropy" in result["Medic"]
    assert len(result["Medic"]["token_distribution"]) == 8
    print("[PASS] test_role_patterns")


def test_mutual_information_zero():
    """MI should be ~0 when messages and context are independent."""
    rng = np.random.default_rng(42)
    messages = [rng.integers(0, 8, size=(4, 3)) for _ in range(200)]
    contexts = [rng.integers(0, 4, size=(4,)) for _ in range(200)]
    mi = compute_mutual_information(messages, contexts, vocab_size=8)
    assert mi < 0.3, f"Independent signals should have low MI, got {mi:.3f}"
    print(f"[PASS] test_mutual_information_zero — MI={mi:.3f}")


def test_mutual_information_high():
    """MI should be high when messages perfectly encode context."""
    rng = np.random.default_rng(42)
    messages = []
    contexts = []
    for _ in range(500):
        ctx = rng.integers(0, 4, size=(4,))
        msg = np.stack([np.array([c, c, c]) for c in ctx])  # message = context repeated
        messages.append(msg)
        contexts.append(ctx)

    mi = compute_mutual_information(messages, contexts, vocab_size=8)
    assert mi > 1.0, f"Perfectly correlated should have high MI, got {mi:.3f}"
    print(f"[PASS] test_mutual_information_high — MI={mi:.3f}")


def test_generate_report():
    """Report should be a non-empty string with expected sections."""
    messages = _make_messages(50)
    infos = [
        {"episode_reward": -50.0, "victims_rescued": 2, "victims_dead": 8, "victims_alive": 2}
        for _ in range(5)
    ]
    report = generate_analysis_report(messages, infos)
    assert len(report) > 100
    assert "ENTROPY" in report
    assert "TOKEN FREQUENCIES" in report
    assert "PERFORMANCE" in report
    print("[PASS] test_generate_report")


def test_ablation_runs():
    """Ablation study should complete and return comparison metrics."""
    env = CrisisGrid(max_steps=30, num_victims=4, num_supplies=2, num_rubble=2)
    agent = SignalAgent(grid_size=16)
    summary = run_ablation(env, agent, num_episodes=3, seed_start=99999)

    assert "with_comm" in summary
    assert "without_comm" in summary
    assert "comm_reward_delta" in summary
    assert "comm_helps" in summary
    assert isinstance(summary["with_comm"]["reward_mean"], float)

    report = format_ablation_report(summary)
    assert "Communication" in report
    print(f"[PASS] test_ablation_runs — delta={summary['comm_reward_delta']:+.1f}")


def test_analysis_with_real_episodes():
    """Run full pipeline: episode collection + analysis on untrained agent."""
    env = CrisisGrid(max_steps=30, num_victims=4, num_supplies=2, num_rubble=2)
    agent = SignalAgent(grid_size=16)
    buffer = RolloutBuffer(max_steps=30, num_agents=4)

    all_messages = []
    infos = []
    for i in range(3):
        info = run_episode(env, agent, buffer, seed=i + 7000, deterministic=True)
        infos.append(info)
        all_messages.extend(info["messages"])

    report = generate_analysis_report(all_messages, infos, vocab_size=8)
    assert "SIGNAL" in report
    entropy = compute_message_entropy(all_messages, vocab_size=8)
    freq = compute_token_frequencies(all_messages, vocab_size=8)

    print(f"[PASS] test_analysis_with_real_episodes — "
          f"entropy={entropy['mean_entropy']:.2f}, "
          f"unique_msgs={freq['unique_messages']}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 5 Tests: Analysis & Ablation")
    print("=" * 60)
    print()

    print("--- language.py ---")
    test_message_entropy_random()
    test_message_entropy_collapsed()
    test_token_frequencies()
    test_role_patterns()
    test_mutual_information_zero()
    test_mutual_information_high()
    test_generate_report()
    print()

    print("--- ablation.py ---")
    test_ablation_runs()
    print()

    print("--- integration ---")
    test_analysis_with_real_episodes()

    print()
    print("=" * 60)
    print("  ALL PHASE 5 TESTS PASSED")
    print("=" * 60)
