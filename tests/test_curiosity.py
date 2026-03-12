"""
Tests for Phase 3: Intrinsic Curiosity Module (curiosity.py).

Run from project root:
    python3 tests/test_curiosity.py
"""

import sys
sys.path.insert(0, ".")

import torch
import numpy as np

from src.environment.crisisgrid import CrisisGrid, NUM_CHANNELS
from src.environment.spaces import encode_observation, MAX_VISION_SIZE, STATE_DIM
from src.agents.curiosity import (
    FeatureEncoder,
    ForwardDynamicsModel,
    IntrinsicCuriosityModule,
)


def test_feature_encoder_shape():
    """FeatureEncoder should produce (B, feature_dim) from grid + state."""
    encoder = FeatureEncoder(grid_channels=NUM_CHANNELS, feature_dim=64)
    grid = torch.randn(4, NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
    state = torch.randn(4, STATE_DIM)
    out = encoder(grid, state)
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"
    print("[PASS] test_feature_encoder_shape")


def test_forward_model_shape():
    """ForwardDynamicsModel should predict features of same dimension."""
    model = ForwardDynamicsModel(feature_dim=64, message_dim=24)
    features = torch.randn(4, 64)
    message = torch.randn(4, 24)
    pred = model(features, message)
    assert pred.shape == (4, 64), f"Expected (4, 64), got {pred.shape}"
    print("[PASS] test_forward_model_shape")


def test_intrinsic_reward_shape():
    """Intrinsic reward should be (B,) shaped."""
    icm = IntrinsicCuriosityModule()
    B = 4
    grid_t = torch.randn(B, NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
    state_t = torch.randn(B, STATE_DIM)
    grid_tp1 = torch.randn(B, NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
    state_tp1 = torch.randn(B, STATE_DIM)
    message = torch.randn(B, 24)
    reward = icm.compute_intrinsic_reward(
        grid_t, state_t, grid_tp1, state_tp1, message, normalize=False,
    )
    assert reward.shape == (B,), f"Expected ({B},), got {reward.shape}"
    assert (reward >= 0).all(), "Unnormalized reward should be non-negative"
    print("[PASS] test_intrinsic_reward_shape")


def test_intrinsic_reward_normalized():
    """Normalized reward should have reasonable magnitude."""
    icm = IntrinsicCuriosityModule(beta_start=1.0)
    B = 16
    for _ in range(10):
        grid_t = torch.randn(B, NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
        state_t = torch.randn(B, STATE_DIM)
        grid_tp1 = torch.randn(B, NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
        state_tp1 = torch.randn(B, STATE_DIM)
        message = torch.randn(B, 24)
        reward = icm.compute_intrinsic_reward(
            grid_t, state_t, grid_tp1, state_tp1, message, normalize=True,
        )
        assert reward.shape == (B,)
    assert icm._reward_count == 10, "Running stats should update"
    print("[PASS] test_intrinsic_reward_normalized")


def test_forward_loss():
    """Forward model loss should be a scalar and backprop should work."""
    icm = IntrinsicCuriosityModule()
    B = 8
    grid_t = torch.randn(B, NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
    state_t = torch.randn(B, STATE_DIM)
    grid_tp1 = torch.randn(B, NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
    state_tp1 = torch.randn(B, STATE_DIM)
    message = torch.randn(B, 24)
    loss = icm.get_loss(grid_t, state_t, grid_tp1, state_tp1, message)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"
    loss.backward()
    grad_count = sum(
        1 for p in icm.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert grad_count > 0, "No gradients flowing"
    print("[PASS] test_forward_loss")


def test_beta_annealing():
    """Beta should anneal linearly from start to end."""
    icm = IntrinsicCuriosityModule(
        beta_start=1.0, beta_end=0.0, anneal_episodes=100,
    )
    assert icm.beta == 1.0, "Initial beta should be beta_start"
    icm.update_beta(0)
    assert icm.beta == 1.0, "Beta at episode 0 should be start"
    icm.update_beta(50)
    assert abs(icm.beta - 0.5) < 1e-6, f"Beta at midpoint should be 0.5, got {icm.beta}"
    icm.update_beta(100)
    assert icm.beta == 0.0, "Beta at end should be beta_end"
    icm.update_beta(200)
    assert icm.beta == 0.0, "Beta past anneal should stay at end"
    print("[PASS] test_beta_annealing")


def test_message_improves_prediction():
    """
    Informative messages should reduce prediction error after training.

    This is the core hypothesis: train the forward model on transitions where
    the message contains information about the next state. After training,
    predictions with the real message should be better than with random noise.
    """
    torch.manual_seed(42)
    icm = IntrinsicCuriosityModule(feature_dim=32, hidden_dim=64)
    optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)

    B = 32

    # Create a synthetic dataset where the message encodes the next state
    grid_t = torch.randn(B, NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
    state_t = torch.randn(B, STATE_DIM)
    grid_tp1 = torch.randn(B, NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
    state_tp1 = torch.randn(B, STATE_DIM)

    # The "informative" message: derived from next state (simplified)
    with torch.no_grad():
        next_features = icm.encode(grid_tp1, state_tp1)
    informative_msg = next_features[:, :24].detach()  # first 24 dims as message

    # Train the forward model
    for _ in range(200):
        optimizer.zero_grad()
        loss = icm.get_loss(grid_t, state_t, grid_tp1, state_tp1, informative_msg)
        loss.backward()
        optimizer.step()

    # Compare prediction error: informative message vs random noise
    with torch.no_grad():
        reward_informative = icm.compute_intrinsic_reward(
            grid_t, state_t, grid_tp1, state_tp1, informative_msg, normalize=False,
        )
        random_msg = torch.randn_like(informative_msg)
        reward_random = icm.compute_intrinsic_reward(
            grid_t, state_t, grid_tp1, state_tp1, random_msg, normalize=False,
        )

    mean_info = reward_informative.mean().item()
    mean_random = reward_random.mean().item()
    assert mean_info < mean_random, (
        f"Informative message should have lower prediction error: "
        f"info={mean_info:.4f} vs random={mean_random:.4f}"
    )
    print(f"[PASS] test_message_improves_prediction — "
          f"info_error={mean_info:.4f} < random_error={mean_random:.4f}")


def test_env_integration():
    """ICM should work with real environment observations."""
    icm = IntrinsicCuriosityModule()
    env = CrisisGrid()
    rng = np.random.default_rng(42)

    obs_t = env.reset(seed=42)
    actions = {
        i: {
            "move": int(rng.integers(0, 5)),
            "task": int(rng.integers(0, 6)),
            "message": rng.integers(0, 8, size=3).tolist(),
        }
        for i in range(4)
    }
    obs_tp1, _, _, _ = env.step(actions)

    # Compute curiosity reward for agent 0
    enc_t = encode_observation(obs_t[0], env.agents[0].role, vocab_size=8)
    enc_tp1 = encode_observation(obs_tp1[0], env.agents[0].role, vocab_size=8)

    grid_t = enc_t["grid"].unsqueeze(0)
    state_t = enc_t["state"].unsqueeze(0)
    grid_tp1 = enc_tp1["grid"].unsqueeze(0)
    state_tp1 = enc_tp1["state"].unsqueeze(0)

    # Flatten all received messages into a single vector for agent 0
    msg = enc_tp1["messages"].reshape(1, -1)[:, :24]  # take first 24 dims

    with torch.no_grad():
        reward = icm.compute_intrinsic_reward(
            grid_t, state_t, grid_tp1, state_tp1, msg, normalize=False,
        )
    assert reward.shape == (1,)
    assert reward.item() >= 0
    print("[PASS] test_env_integration")


def test_parameter_count():
    """ICM should be lightweight relative to the policy."""
    icm = IntrinsicCuriosityModule()
    counts = icm.count_parameters()
    print(f"  ICM parameter counts: {counts}")
    assert counts["total"] < 200_000, f"ICM too large: {counts['total']}"
    print(f"[PASS] test_parameter_count — {counts['total']:,} parameters")


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 3 Tests: curiosity.py (ICM)")
    print("=" * 60)
    print()
    test_feature_encoder_shape()
    test_forward_model_shape()
    test_intrinsic_reward_shape()
    test_intrinsic_reward_normalized()
    test_forward_loss()
    test_beta_annealing()
    test_message_improves_prediction()
    test_env_integration()
    test_parameter_count()
    print()
    print("=" * 60)
    print("  ALL PHASE 3 TESTS PASSED")
    print("=" * 60)
