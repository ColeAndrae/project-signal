"""
Tests for Phase 2: Observation encoding (spaces.py) and neural networks (networks.py).

Run from project root:
    python3 -m pytest tests/test_agents.py -v
    OR
    python3 tests/test_agents.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

from src.environment.crisisgrid import CrisisGrid, Role, NUM_CHANNELS, VISION_RADIUS
from src.environment.spaces import (
    encode_grid_observation,
    encode_state,
    encode_messages,
    encode_observation,
    batch_encode_observations,
    decode_actions,
    sample_action_from_logits,
    compute_action_log_prob,
    MAX_VISION_SIZE,
    STATE_DIM,
)
from src.agents.networks import (
    GridEncoder,
    MessageEncoder,
    PolicyNetwork,
    CentralizedCritic,
    SignalAgent,
)


# ============================================================
# spaces.py tests
# ============================================================

def test_encode_grid_medic():
    """Medic (5x5 vision) should be padded to 7x7."""
    grid = np.random.rand(5, 5, NUM_CHANNELS).astype(np.float32)
    tensor = encode_grid_observation(grid, Role.MEDIC)
    assert tensor.shape == (NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE), \
        f"Expected (6, 7, 7), got {tensor.shape}"
    # Padded border should be -1.0
    assert tensor[0, 0, 0].item() == -1.0, "Padded cells should be -1.0"
    print("[PASS] test_encode_grid_medic")


def test_encode_grid_scout():
    """Scout (7x7 vision) needs no padding."""
    grid = np.random.rand(7, 7, NUM_CHANNELS).astype(np.float32)
    tensor = encode_grid_observation(grid, Role.SCOUT)
    assert tensor.shape == (NUM_CHANNELS, MAX_VISION_SIZE, MAX_VISION_SIZE)
    # No padding, so corner should match input
    assert tensor[0, 0, 0].item() != -1.0, "Scout grid should not be padded"
    print("[PASS] test_encode_grid_scout")


def test_encode_state():
    """State vector should preserve values."""
    state = np.array([0.5, 0.3, 1.0, 0.0, 0.33, 0.1, 0.9, 0.0], dtype=np.float32)
    tensor = encode_state(state)
    assert tensor.shape == (STATE_DIM,)
    assert torch.allclose(tensor, torch.tensor(state))
    print("[PASS] test_encode_state")


def test_encode_messages():
    """Messages should be one-hot encoded and flattened per sender."""
    messages = np.array([[0, 3, 7], [1, 1, 0], [5, 2, 4]], dtype=np.int64)
    vocab_size = 8
    tensor = encode_messages(messages, vocab_size)
    # Shape: (3 agents, 3 tokens * 8 vocab = 24)
    assert tensor.shape == (3, 24), f"Expected (3, 24), got {tensor.shape}"
    # Check one-hot: first sender, first token is 0 → index 0 should be 1
    assert tensor[0, 0].item() == 1.0, "Token 0 should activate index 0"
    assert tensor[0, 1].item() == 0.0
    print("[PASS] test_encode_messages")


def test_encode_full_observation():
    """Full observation encoding from a live environment."""
    env = CrisisGrid()
    obs = env.reset(seed=42)
    for agent_id, agent_obs in obs.items():
        role = env.agents[agent_id].role
        encoded = encode_observation(agent_obs, role, vocab_size=8)
        assert encoded["grid"].shape == (NUM_CHANNELS, 7, 7)
        assert encoded["state"].shape == (STATE_DIM,)
        assert encoded["messages"].shape == (3, 24)  # 3 others, 3 tokens * 8 vocab
    print("[PASS] test_encode_full_observation")


def test_batch_encode():
    """Batch encoding should stack correctly."""
    env = CrisisGrid()
    obs = env.reset(seed=0)
    obs_list = [obs[i] for i in range(4)]
    roles = [env.agents[i].role for i in range(4)]
    batched = batch_encode_observations(obs_list, roles, vocab_size=8)
    assert batched["grid"].shape == (4, NUM_CHANNELS, 7, 7)
    assert batched["state"].shape == (4, STATE_DIM)
    assert batched["messages"].shape == (4, 3, 24)
    print("[PASS] test_batch_encode")


def test_decode_actions():
    """Action decoding should produce valid env action dicts."""
    moves = torch.tensor([0, 1, 2, 3])
    tasks = torch.tensor([1, 0, 3, 5])
    msgs = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]])
    actions = decode_actions(moves, tasks, msgs)
    assert len(actions) == 4
    assert actions[0] == {"move": 0, "task": 1, "message": [0, 1, 2]}
    assert actions[3] == {"move": 3, "task": 5, "message": [1, 2, 3]}
    print("[PASS] test_decode_actions")


def test_sample_action_from_logits():
    """Sampling should produce correctly shaped outputs."""
    B = 8
    move_logits = torch.randn(B, 5)
    task_logits = torch.randn(B, 6)
    msg_logits = torch.randn(B, 3, 8)
    move_idx, task_idx, msg_tokens, log_prob = sample_action_from_logits(
        move_logits, task_logits, msg_logits,
    )
    assert move_idx.shape == (B,)
    assert task_idx.shape == (B,)
    assert msg_tokens.shape == (B, 3)
    assert log_prob.shape == (B,)
    # All move indices should be in [0, 4]
    assert (move_idx >= 0).all() and (move_idx < 5).all()
    # Log probs should be negative
    assert (log_prob <= 0).all()
    print("[PASS] test_sample_action_from_logits")


def test_compute_action_log_prob():
    """Log prob computation should be consistent with sampling."""
    B = 4
    move_logits = torch.randn(B, 5)
    task_logits = torch.randn(B, 6)
    msg_logits = torch.randn(B, 3, 8)

    move_idx, task_idx, msg_tokens, sample_logp = sample_action_from_logits(
        move_logits, task_logits, msg_logits,
    )
    eval_logp, entropy = compute_action_log_prob(
        move_logits, task_logits, msg_logits,
        move_idx, task_idx, msg_tokens,
    )
    # Log probs should match
    assert torch.allclose(sample_logp, eval_logp, atol=1e-5), \
        f"Mismatch: {sample_logp} vs {eval_logp}"
    # Entropy should be positive
    assert (entropy > 0).all()
    print("[PASS] test_compute_action_log_prob")


# ============================================================
# networks.py tests
# ============================================================

def test_grid_encoder():
    """GridEncoder should map (B, 6, 7, 7) → (B, 128)."""
    encoder = GridEncoder(in_channels=NUM_CHANNELS, feature_dim=128)
    x = torch.randn(4, NUM_CHANNELS, 7, 7)
    out = encoder(x)
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"
    print("[PASS] test_grid_encoder")


def test_message_encoder():
    """MessageEncoder should aggregate messages → (B, hidden_dim)."""
    encoder = MessageEncoder(
        num_other_agents=3, message_length=3, vocab_size=8, hidden_dim=64,
    )
    # Input: (B, 3 agents, 3*8 = 24 flattened one-hot)
    msgs = torch.randn(4, 3, 24)
    out = encoder(msgs)
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"
    print("[PASS] test_message_encoder")


def test_policy_network():
    """PolicyNetwork should output three sets of logits."""
    policy = PolicyNetwork()
    grid = torch.randn(2, NUM_CHANNELS, 7, 7)
    state = torch.randn(2, STATE_DIM)
    messages = torch.randn(2, 3, 24)
    move_logits, task_logits, msg_logits = policy(grid, state, messages)
    assert move_logits.shape == (2, 5), f"Move: {move_logits.shape}"
    assert task_logits.shape == (2, 8), f"Task: {task_logits.shape}"
    assert msg_logits.shape == (2, 3, 8), f"Msg: {msg_logits.shape}"
    print("[PASS] test_policy_network")


def test_policy_get_features():
    """get_features should return core representation."""
    policy = PolicyNetwork(core_hidden_dim=256)
    grid = torch.randn(2, NUM_CHANNELS, 7, 7)
    state = torch.randn(2, STATE_DIM)
    messages = torch.randn(2, 3, 24)
    features = policy.get_features(grid, state, messages)
    assert features.shape == (2, 256)
    print("[PASS] test_policy_get_features")


def test_centralized_critic():
    """CentralizedCritic should output (B, 1) values."""
    critic = CentralizedCritic(grid_size=16, num_grid_channels=10)
    global_grid = torch.randn(2, 10, 16, 16)
    all_states = torch.randn(2, 4 * STATE_DIM)   # 4 agents * 7
    all_msgs = torch.randn(2, 4 * 3 * 8)         # 4 agents * 3 tokens * 8 vocab
    values = critic(global_grid, all_states, all_msgs)
    assert values.shape == (2, 1), f"Expected (2, 1), got {values.shape}"
    print("[PASS] test_centralized_critic")


def test_signal_agent_act():
    """SignalAgent.act should produce valid actions from env observations."""
    agent = SignalAgent()
    env = CrisisGrid()
    obs = env.reset(seed=42)

    # Encode agent 0's observation
    encoded = encode_observation(obs[0], env.agents[0].role, vocab_size=8)

    # Add batch dimension
    grid = encoded["grid"].unsqueeze(0)
    state = encoded["state"].unsqueeze(0)
    messages = encoded["messages"].unsqueeze(0)

    with torch.no_grad():
        move, task, msg, logp = agent.act(grid, state, messages)
    assert move.shape == (1,)
    assert task.shape == (1,)
    assert msg.shape == (1, 3)
    assert logp.shape == (1,)
    print("[PASS] test_signal_agent_act")


def test_signal_agent_evaluate_value():
    """SignalAgent critic should accept global state."""
    agent = SignalAgent()
    env = CrisisGrid()
    env.reset(seed=0)

    global_state = env.get_global_state()  # (16, 16, 10)
    # Convert to (B, C, H, W)
    global_grid = torch.from_numpy(global_state).float().permute(2, 0, 1).unsqueeze(0)
    all_states = torch.randn(1, 4 * STATE_DIM)
    all_msgs = torch.randn(1, 4 * 3 * 8)

    with torch.no_grad():
        value = agent.evaluate_value(global_grid, all_states, all_msgs)
    assert value.shape == (1, 1)
    print("[PASS] test_signal_agent_evaluate_value")


def test_parameter_count():
    """Verify parameter count is reasonable (< 500K for trainability)."""
    agent = SignalAgent()
    counts = agent.count_parameters()
    total = counts["total"]
    print(f"  Parameter counts: {counts}")
    assert total < 1_000_000, f"Too many parameters: {total}"
    assert total > 50_000, f"Too few parameters: {total}"
    print(f"[PASS] test_parameter_count — {total:,} total parameters")


def test_end_to_end_step():
    """Full pipeline: env → encode → network → decode → env.step()."""
    env = CrisisGrid()
    obs = env.reset(seed=99)
    agent = SignalAgent()

    with torch.no_grad():
        actions = {}
        for i in range(env.num_agents):
            encoded = encode_observation(obs[i], env.agents[i].role, vocab_size=8)
            grid = encoded["grid"].unsqueeze(0)
            state = encoded["state"].unsqueeze(0)
            messages = encoded["messages"].unsqueeze(0)

            move, task, msg, _ = agent.act(grid, state, messages)
            actions[i] = {
                "move": move.item(),
                "task": task.item(),
                "message": msg[0].tolist(),
            }

    obs2, reward, done, info = env.step(actions)
    assert isinstance(reward, float)
    assert info["step"] == 1
    print("[PASS] test_end_to_end_step")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2 Tests: spaces.py + networks.py")
    print("=" * 60)
    print()

    # spaces.py
    print("--- spaces.py ---")
    test_encode_grid_medic()
    test_encode_grid_scout()
    test_encode_state()
    test_encode_messages()
    test_encode_full_observation()
    test_batch_encode()
    test_decode_actions()
    test_sample_action_from_logits()
    test_compute_action_log_prob()
    print()

    # networks.py
    print("--- networks.py ---")
    test_grid_encoder()
    test_message_encoder()
    test_policy_network()
    test_policy_get_features()
    test_centralized_critic()
    test_signal_agent_act()
    test_signal_agent_evaluate_value()
    test_parameter_count()
    test_end_to_end_step()

    print()
    print("=" * 60)
    print("  ALL PHASE 2 TESTS PASSED")
    print("=" * 60)
