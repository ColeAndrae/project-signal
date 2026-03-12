"""
Tests for Phase 4: Training infrastructure (buffer, mappo, runner, train).

Run from project root:
    python3 tests/test_training.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

from src.environment.crisisgrid import CrisisGrid, NUM_CHANNELS
from src.environment.spaces import encode_observation, STATE_DIM
from src.agents.networks import SignalAgent
from src.agents.curiosity import IntrinsicCuriosityModule
from src.training.buffer import RolloutBuffer
from src.training.mappo import MAPPOTrainer
from src.training.runner import run_episode, evaluate


def test_buffer_insert_and_size():
    """Buffer should track insertions correctly."""
    buf = RolloutBuffer(max_steps=200, num_agents=4)
    assert buf.size == 0

    # Insert dummy data
    for _ in range(10):
        buf.insert(
            grids=torch.randn(4, 6, 7, 7),
            states=torch.randn(4, 7),
            messages_in=torch.randn(4, 3, 24),
            moves=torch.randint(0, 5, (4,)),
            tasks=torch.randint(0, 6, (4,)),
            msg_tokens=torch.randint(0, 8, (4, 3)),
            log_probs=torch.randn(4),
            global_grid=torch.randn(10, 16, 16),
            all_states=torch.randn(28),
            all_messages=torch.randn(96),
            reward=1.0,
            done=False,
            value=0.5,
        )
        buf.insert_next_obs(torch.randn(4, 6, 7, 7), torch.randn(4, 7))

    assert buf.size == 10
    print("[PASS] test_buffer_insert_and_size")


def test_buffer_compute_returns():
    """GAE computation should produce non-zero advantages."""
    buf = RolloutBuffer(max_steps=200, num_agents=4)
    for i in range(20):
        buf.insert(
            grids=torch.randn(4, 6, 7, 7),
            states=torch.randn(4, 7),
            messages_in=torch.randn(4, 3, 24),
            moves=torch.randint(0, 5, (4,)),
            tasks=torch.randint(0, 6, (4,)),
            msg_tokens=torch.randint(0, 8, (4, 3)),
            log_probs=torch.randn(4),
            global_grid=torch.randn(10, 16, 16),
            all_states=torch.randn(28),
            all_messages=torch.randn(96),
            reward=float(i) * 0.1,
            done=(i == 19),
            value=float(i) * 0.05,
        )

    buf.compute_returns(last_value=0.0)
    assert buf.advantages[:20].abs().sum() > 0, "Advantages should be non-zero"
    assert buf.returns[:20].abs().sum() > 0, "Returns should be non-zero"
    print("[PASS] test_buffer_compute_returns")


def test_buffer_generate_batches():
    """Batch generation should cover all data points."""
    buf = RolloutBuffer(max_steps=200, num_agents=4)
    for i in range(10):
        buf.insert(
            grids=torch.randn(4, 6, 7, 7),
            states=torch.randn(4, 7),
            messages_in=torch.randn(4, 3, 24),
            moves=torch.randint(0, 5, (4,)),
            tasks=torch.randint(0, 6, (4,)),
            msg_tokens=torch.randint(0, 8, (4, 3)),
            log_probs=torch.randn(4),
            global_grid=torch.randn(10, 16, 16),
            all_states=torch.randn(28),
            all_messages=torch.randn(96),
            reward=1.0,
            done=(i == 9),
            value=0.5,
        )
        buf.insert_next_obs(torch.randn(4, 6, 7, 7), torch.randn(4, 7))

    buf.compute_returns(last_value=0.0)
    batches = buf.generate_batches(mini_batch_size=16)

    total_samples = sum(b["grids"].shape[0] for b in batches)
    assert total_samples == 40, f"Expected 40 samples (10 steps * 4 agents), got {total_samples}"

    # Check all expected keys exist
    expected_keys = {
        "grids", "states", "messages_in", "moves", "tasks", "msg_tokens",
        "log_probs", "returns", "advantages", "values",
        "global_grids", "all_agent_states", "all_agent_messages",
        "next_grids", "next_states",
    }
    assert expected_keys == set(batches[0].keys()), f"Missing keys: {expected_keys - set(batches[0].keys())}"
    print("[PASS] test_buffer_generate_batches")


def test_buffer_reset():
    """Buffer reset should clear the pointer."""
    buf = RolloutBuffer(max_steps=200, num_agents=4)
    buf.insert(
        grids=torch.randn(4, 6, 7, 7), states=torch.randn(4, 7),
        messages_in=torch.randn(4, 3, 24), moves=torch.randint(0, 5, (4,)),
        tasks=torch.randint(0, 6, (4,)), msg_tokens=torch.randint(0, 8, (4, 3)),
        log_probs=torch.randn(4), global_grid=torch.randn(10, 16, 16),
        all_states=torch.randn(28), all_messages=torch.randn(96),
        reward=1.0, done=False, value=0.5,
    )
    assert buf.size == 1
    buf.reset()
    assert buf.size == 0
    print("[PASS] test_buffer_reset")


def test_run_episode():
    """Full episode collection should fill buffer and return valid metrics."""
    env = CrisisGrid(max_steps=50, num_victims=4, num_supplies=2, num_rubble=3)
    agent = SignalAgent(grid_size=16)
    buffer = RolloutBuffer(max_steps=50, num_agents=4)

    info = run_episode(env, agent, buffer, seed=42)

    assert buffer.size > 0, "Buffer should have data"
    assert buffer.size <= 50, f"Buffer size {buffer.size} exceeds max_steps"
    assert "episode_reward" in info
    assert "victims_rescued" in info
    assert "messages" in info
    assert len(info["messages"]) == buffer.size
    print(f"[PASS] test_run_episode — {buffer.size} steps, reward={info['episode_reward']:.1f}")


def test_run_episode_with_icm():
    """Episode with ICM should compute intrinsic rewards."""
    env = CrisisGrid(max_steps=50, num_victims=4, num_supplies=2, num_rubble=3)
    agent = SignalAgent(grid_size=16)
    icm = IntrinsicCuriosityModule()
    buffer = RolloutBuffer(max_steps=50, num_agents=4)

    info = run_episode(env, agent, buffer, icm=icm, seed=42)

    assert info["episode_intrinsic_reward"] != 0.0, "ICM should produce non-zero intrinsic reward"
    print(f"[PASS] test_run_episode_with_icm — intrinsic_r={info['episode_intrinsic_reward']:.3f}")


def test_mappo_update():
    """MAPPO trainer should run one update without errors and return metrics."""
    env = CrisisGrid(max_steps=50, num_victims=4, num_supplies=2, num_rubble=3)
    agent = SignalAgent(grid_size=16)
    icm = IntrinsicCuriosityModule()
    buffer = RolloutBuffer(max_steps=50, num_agents=4)
    trainer = MAPPOTrainer(agent=agent, icm=icm, ppo_epochs=2, mini_batch_size=16)

    # Collect an episode
    run_episode(env, agent, buffer, icm=icm, seed=42)

    # Run update
    metrics = trainer.update(buffer)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "icm_loss" in metrics
    assert "clip_fraction" in metrics
    assert metrics["entropy"] > 0, "Entropy should be positive"
    print(f"[PASS] test_mappo_update — PL={metrics['policy_loss']:.3f} VL={metrics['value_loss']:.3f}")


def test_mappo_improves():
    """Multiple training updates should change the policy (weights should differ)."""
    env = CrisisGrid(max_steps=50, num_victims=4, num_supplies=2, num_rubble=3)
    agent = SignalAgent(grid_size=16)
    buffer = RolloutBuffer(max_steps=50, num_agents=4)
    trainer = MAPPOTrainer(agent=agent, icm=None, ppo_epochs=2, mini_batch_size=16)

    # Snapshot initial weights
    initial_weights = agent.policy.move_head.weight.data.clone()

    for ep in range(5):
        run_episode(env, agent, buffer, seed=ep)
        trainer.update(buffer)

    final_weights = agent.policy.move_head.weight.data
    weight_diff = (final_weights - initial_weights).abs().sum().item()
    assert weight_diff > 0.01, f"Weights barely changed: diff={weight_diff}"
    print(f"[PASS] test_mappo_improves — weight_diff={weight_diff:.4f}")


def test_evaluate():
    """Evaluation should run without training and return stable metrics."""
    env = CrisisGrid(max_steps=50, num_victims=4, num_supplies=2, num_rubble=3)
    agent = SignalAgent(grid_size=16)

    metrics = evaluate(env, agent, num_episodes=3, seed_start=999)

    assert "eval_reward_mean" in metrics
    assert "eval_rescued_mean" in metrics
    assert isinstance(metrics["eval_reward_mean"], float)
    print(f"[PASS] test_evaluate — reward={metrics['eval_reward_mean']:.1f}")


def test_mini_training_loop():
    """Simulate a tiny training run: 10 episodes of collect → train."""
    env = CrisisGrid(max_steps=30, num_victims=4, num_supplies=2, num_rubble=2)
    agent = SignalAgent(grid_size=16)
    icm = IntrinsicCuriosityModule(beta_start=0.3, beta_end=0.01, anneal_episodes=10)
    buffer = RolloutBuffer(max_steps=30, num_agents=4)
    trainer = MAPPOTrainer(agent=agent, icm=icm, ppo_epochs=2, mini_batch_size=16)

    rewards = []
    for ep in range(1, 11):
        info = run_episode(env, agent, buffer, icm=icm, seed=ep)
        trainer.update(buffer)
        icm.update_beta(ep)
        rewards.append(info["episode_reward"])

    print(f"  Rewards over 10 episodes: {[f'{r:.0f}' for r in rewards]}")
    print(f"  Final beta: {icm.beta:.3f}")
    assert len(rewards) == 10
    assert icm.beta < 0.3, "Beta should have annealed"
    print(f"[PASS] test_mini_training_loop")


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 4 Tests: Training Infrastructure")
    print("=" * 60)
    print()

    print("--- buffer.py ---")
    test_buffer_insert_and_size()
    test_buffer_compute_returns()
    test_buffer_generate_batches()
    test_buffer_reset()
    print()

    print("--- runner.py ---")
    test_run_episode()
    test_run_episode_with_icm()
    print()

    print("--- mappo.py ---")
    test_mappo_update()
    test_mappo_improves()
    test_evaluate()
    print()

    print("--- integration ---")
    test_mini_training_loop()

    print()
    print("=" * 60)
    print("  ALL PHASE 4 TESTS PASSED")
    print("=" * 60)
