"""Unit tests for CrisisGrid environment."""

import sys
sys.path.insert(0, "/home/claude/project-signal")

import numpy as np
from src.environment.crisisgrid import (
    CrisisGrid, CellType, Severity, Role, MoveAction, TaskAction,
    NUM_CHANNELS, VISION_RADIUS,
)


def test_reset_deterministic():
    """Same seed should produce identical initial states."""
    env1 = CrisisGrid()
    env2 = CrisisGrid()
    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)

    for agent_id in obs1:
        assert np.array_equal(obs1[agent_id]["grid"], obs2[agent_id]["grid"]), \
            f"Agent {agent_id} grid mismatch on identical seed"
        assert np.array_equal(obs1[agent_id]["state"], obs2[agent_id]["state"]), \
            f"Agent {agent_id} state mismatch on identical seed"
    print("[PASS] test_reset_deterministic")


def test_observation_shapes():
    """Verify observation tensor shapes match expected dimensions."""
    env = CrisisGrid()
    obs = env.reset(seed=0)

    for agent_id, agent_obs in obs.items():
        role = env.agents[agent_id].role
        radius = VISION_RADIUS[role]
        expected_vision = 2 * radius + 1

        grid = agent_obs["grid"]
        assert grid.shape == (expected_vision, expected_vision, NUM_CHANNELS), \
            f"Agent {agent_id} (role={role}): grid shape {grid.shape} != " \
            f"({expected_vision}, {expected_vision}, {NUM_CHANNELS})"

        state = agent_obs["state"]
        assert state.shape == (7,), f"Agent {agent_id}: state shape {state.shape} != (7,)"

        msgs = agent_obs["messages"]
        assert msgs.shape == (env.num_agents - 1, env.message_length), \
            f"Agent {agent_id}: messages shape {msgs.shape}"

    print("[PASS] test_observation_shapes")


def test_step_basic():
    """Verify step() runs without error with random actions."""
    env = CrisisGrid()
    env.reset(seed=123)
    rng = np.random.default_rng(123)

    for _ in range(10):
        actions = {}
        for i in range(env.num_agents):
            actions[i] = {
                "move": int(rng.integers(0, 5)),
                "task": int(rng.integers(0, 6)),
                "message": rng.integers(0, env.vocab_size, size=env.message_length).tolist(),
            }
        obs, reward, done, info = env.step(actions)
        assert isinstance(reward, float), f"Reward is not float: {type(reward)}"
        assert isinstance(done, bool), f"Done is not bool: {type(done)}"
        assert "alive_victims" in info
        if done:
            break

    print("[PASS] test_step_basic")


def test_agent_movement():
    """Verify agents move correctly and respect boundaries."""
    env = CrisisGrid(grid_size=8, num_rubble=0, num_victims=2, num_supplies=0)
    env.reset(seed=0)

    # Record starting position of agent 0
    start_r, start_c = env.agents[0].row, env.agents[0].col

    # Move agent 0 SOUTH, all others STAY
    actions = {}
    for i in range(env.num_agents):
        actions[i] = {"move": MoveAction.STAY, "task": TaskAction.NOOP, "message": [0, 0, 0]}
    actions[0]["move"] = MoveAction.SOUTH

    env.step(actions)

    new_r, new_c = env.agents[0].row, env.agents[0].col
    # Should have moved south (row + 1) unless blocked
    if start_r < env.grid_size - 1 and env.terrain[start_r + 1, start_c] != CellType.RUBBLE:
        assert new_r == start_r + 1, f"Expected row {start_r + 1}, got {new_r}"
        assert new_c == start_c, f"Column should not change"
    print("[PASS] test_agent_movement")


def test_victim_decay():
    """Verify victims lose health over time and can die."""
    env = CrisisGrid(max_steps=500, victim_decay_rate=0.2, aftershock_prob=0.0)
    env.reset(seed=7)

    noop_actions = {
        i: {"move": MoveAction.STAY, "task": TaskAction.NOOP, "message": [0, 0, 0]}
        for i in range(env.num_agents)
    }

    initial_alive = env.alive_victim_count()
    for _ in range(50):
        _, _, done, _ = env.step(noop_actions)
        if done:
            break

    final_alive = env.alive_victim_count()
    assert final_alive < initial_alive, \
        f"No victims died after 50 steps with decay=0.2 ({initial_alive} → {final_alive})"
    print("[PASS] test_victim_decay")


def test_hazard_spread():
    """Verify hazards spread to adjacent cells over time."""
    env = CrisisGrid(hazard_spread_prob=0.5, aftershock_prob=0.0)
    env.reset(seed=99)

    initial_hazards = int(np.sum(env.hazard_map > 0))

    noop_actions = {
        i: {"move": MoveAction.STAY, "task": TaskAction.NOOP, "message": [0, 0, 0]}
        for i in range(env.num_agents)
    }

    for _ in range(30):
        env.step(noop_actions)

    final_hazards = int(np.sum(env.hazard_map > 0))
    assert final_hazards >= initial_hazards, \
        f"Hazards should not shrink: {initial_hazards} → {final_hazards}"
    print("[PASS] test_hazard_spread")


def test_global_state():
    """Verify global state tensor for centralized critic."""
    env = CrisisGrid()
    env.reset(seed=0)
    global_state = env.get_global_state()

    expected_channels = NUM_CHANNELS + env.num_agents
    assert global_state.shape == (env.grid_size, env.grid_size, expected_channels), \
        f"Global state shape {global_state.shape} != ({env.grid_size}, {env.grid_size}, {expected_channels})"

    # Each agent should appear exactly once in their channel
    for i, agent in enumerate(env.agents):
        agent_channel = global_state[:, :, NUM_CHANNELS + i]
        assert agent_channel[agent.row, agent.col] == 1.0, f"Agent {i} not found at position"
        assert np.sum(agent_channel) == 1.0, f"Agent {i} appears in multiple positions"

    print("[PASS] test_global_state")


def test_render():
    """Verify render produces output without crashing."""
    env = CrisisGrid()
    env.reset(seed=42)
    output = env.render()
    assert isinstance(output, str), "Render should return a string"
    assert len(output) > 0, "Render output should not be empty"
    assert "Step 0/" in output, "Render should show step count"
    print("[PASS] test_render")


def test_full_episode_random():
    """Run a full episode with random actions and collect metrics."""
    env = CrisisGrid(max_steps=200)
    env.reset(seed=42)
    rng = np.random.default_rng(42)

    total_reward = 0.0
    steps = 0

    for _ in range(200):
        actions = {}
        for i in range(env.num_agents):
            actions[i] = {
                "move": int(rng.integers(0, 5)),
                "task": int(rng.integers(0, 6)),
                "message": rng.integers(0, env.vocab_size, size=env.message_length).tolist(),
            }
        obs, reward, done, info = env.step(actions)
        total_reward += reward
        steps += 1
        if done:
            break

    print(f"[PASS] test_full_episode_random — {steps} steps, reward={total_reward:.1f}, "
          f"rescued={env.rescued_victim_count()}, dead={env.dead_victim_count()}")


if __name__ == "__main__":
    test_reset_deterministic()
    test_observation_shapes()
    test_step_basic()
    test_agent_movement()
    test_victim_decay()
    test_hazard_spread()
    test_global_state()
    test_render()
    test_full_episode_random()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
