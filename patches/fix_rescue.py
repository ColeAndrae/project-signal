#!/usr/bin/env python3
"""
Patch script: Add victim carrying mechanic and reward shaping.

Modifies 4 files:
  1. src/environment/crisisgrid.py — CARRY_VICTIM/DROP_VICTIM actions, carrying state,
     victim transport, proximity rewards, state_vec extended to 8 dims
  2. src/environment/spaces.py — STATE_DIM 7 → 8
  3. src/agents/networks.py — task_dim 6 → 8
  4. src/training/buffer.py — state_dim 7 → 8

Run from project root:
    python3 patches/fix_rescue.py
"""

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


def patch_file(path: str, replacements: list[tuple[str, str]]) -> None:
    """Apply a list of (old, new) replacements to a file."""
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements:
        if old not in content:
            print(f"  WARNING: Pattern not found in {path}:")
            print(f"    {old[:80]}...")
            continue
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [PATCHED] {path}")


# ================================================================
# 1. crisisgrid.py — Major changes
# ================================================================

print("Patching crisisgrid.py...")
path = "src/environment/crisisgrid.py"

patch_file(path, [
    # --- Add CARRY_VICTIM and DROP_VICTIM to TaskAction ---
    (
        """class TaskAction(IntEnum):
    \"\"\"Task actions an agent can perform.\"\"\"
    NOOP = 0
    HEAL = 1
    CLEAR_RUBBLE = 2
    PICKUP_SUPPLY = 3
    DROP_SUPPLY = 4
    USE_SUPPLY = 5""",
        """class TaskAction(IntEnum):
    \"\"\"Task actions an agent can perform.\"\"\"
    NOOP = 0
    HEAL = 1
    CLEAR_RUBBLE = 2
    PICKUP_SUPPLY = 3
    DROP_SUPPLY = 4
    USE_SUPPLY = 5
    CARRY_VICTIM = 6
    DROP_VICTIM = 7"""
    ),

    # --- Add carrying_victim to AgentState ---
    (
        """@dataclass
class AgentState:
    \"\"\"Mutable state for a single agent.\"\"\"
    row: int
    col: int
    role: Role
    health: float = 1.0
    supplies_held: int = 0
    rubble_progress: dict[tuple[int, int], int] = field(default_factory=dict)""",
        """@dataclass
class AgentState:
    \"\"\"Mutable state for a single agent.\"\"\"
    row: int
    col: int
    role: Role
    health: float = 1.0
    supplies_held: int = 0
    carrying_victim_id: int | None = None  # ID of victim being carried, or None
    rubble_progress: dict[tuple[int, int], int] = field(default_factory=dict)"""
    ),

    # --- Update movement to transport carried victims ---
    (
        """        # --- Phase 2: Move agents ---
        for agent_id, action in actions.items():
            move = MoveAction(action.get("move", 0))
            agent = self.agents[agent_id]
            dr, dc = MOVE_DELTAS[move]
            nr, nc = agent.row + dr, agent.col + dc

            # Boundary check
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                continue
            # Rubble check — can't walk into rubble
            if self.terrain[nr, nc] == CellType.RUBBLE:
                continue
            # Move
            agent.row, agent.col = nr, nc""",
        """        # --- Phase 2: Move agents ---
        for agent_id, action in actions.items():
            move = MoveAction(action.get("move", 0))
            agent = self.agents[agent_id]
            dr, dc = MOVE_DELTAS[move]
            nr, nc = agent.row + dr, agent.col + dc

            # Boundary check
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                continue
            # Rubble check — can't walk into rubble
            if self.terrain[nr, nc] == CellType.RUBBLE:
                continue
            # Move agent
            agent.row, agent.col = nr, nc

            # If carrying a victim, move them too
            if agent.carrying_victim_id is not None:
                for v in self.victims:
                    if v.id == agent.carrying_victim_id and v.alive:
                        v.row, v.col = nr, nc
                        break"""
    ),

    # --- Add CARRY_VICTIM and DROP_VICTIM task handling ---
    (
        """            elif task == TaskAction.USE_SUPPLY:
                # Use supply on a victim at current location
                if agent.supplies_held > 0:
                    for victim in self.victims:
                        if victim.alive and victim.row == r and victim.col == c:
                            victim.health = min(1.0, victim.health + 0.3)
                            agent.supplies_held -= 1
                            break""",
        """            elif task == TaskAction.USE_SUPPLY:
                # Use supply on a victim at current location
                if agent.supplies_held > 0:
                    for victim in self.victims:
                        if victim.alive and victim.row == r and victim.col == c:
                            victim.health = min(1.0, victim.health + 0.3)
                            agent.supplies_held -= 1
                            break

            elif task == TaskAction.CARRY_VICTIM:
                # Pick up a victim at current location (one at a time)
                if agent.carrying_victim_id is None:
                    for victim in self.victims:
                        if victim.alive and victim.row == r and victim.col == c:
                            agent.carrying_victim_id = victim.id
                            break

            elif task == TaskAction.DROP_VICTIM:
                # Drop carried victim at current location
                if agent.carrying_victim_id is not None:
                    agent.carrying_victim_id = None"""
    ),

    # --- Update rescue check to include carried victims at shelters ---
    (
        """        # --- Phase 4: Check rescues (victim at shelter) ---
        shelter_set = set(self._shelters)
        for victim in self.victims:
            if victim.alive and (victim.row, victim.col) in shelter_set:
                victim.rescued = True
                mult = SEVERITY_MULT[victim.severity]
                reward += self.r_rescued * mult
                info["victims_rescued"] += 1""",
        """        # --- Phase 4: Check rescues (victim at shelter) ---
        shelter_set = set(self._shelters)
        for victim in self.victims:
            if victim.alive and (victim.row, victim.col) in shelter_set:
                victim.rescued = True
                mult = SEVERITY_MULT[victim.severity]
                reward += self.r_rescued * mult
                info["victims_rescued"] += 1
                # Release from carrying
                for ag in self.agents:
                    if ag.carrying_victim_id == victim.id:
                        ag.carrying_victim_id = None

        # --- Phase 4b: Proximity reward shaping ---
        for ag in self.agents:
            # Small reward for being near alive victims (encourages approach)
            min_victim_dist = float("inf")
            for v in self.victims:
                if v.alive:
                    d = abs(ag.row - v.row) + abs(ag.col - v.col)
                    min_victim_dist = min(min_victim_dist, d)
            if min_victim_dist < float("inf"):
                reward += 0.02 * max(0, 5 - min_victim_dist)  # bonus within 5 steps

            # Larger reward for carrying a victim toward a shelter
            if ag.carrying_victim_id is not None:
                min_shelter_dist = min(
                    abs(ag.row - sr) + abs(ag.col - sc) for sr, sc in self._shelters
                )
                reward += 0.1 * max(0, 8 - min_shelter_dist)  # bonus within 8 steps"""
    ),

    # --- Update state vector from 7 to 8 dims (add carrying flag) ---
    (
        """            # Agent state vector
            state_vec = np.array([
                agent.row / self.grid_size,
                agent.col / self.grid_size,
                agent.health,
                agent.supplies_held / CARRY_CAPACITY[agent.role],
                float(agent.role) / 3.0,
                self.step_count / self.max_steps,
                float(self.max_steps - self.step_count) / self.max_steps,
            ], dtype=np.float32)""",
        """            # Agent state vector (8 dims)
            state_vec = np.array([
                agent.row / self.grid_size,
                agent.col / self.grid_size,
                agent.health,
                agent.supplies_held / CARRY_CAPACITY[agent.role],
                float(agent.role) / 3.0,
                self.step_count / self.max_steps,
                float(self.max_steps - self.step_count) / self.max_steps,
                1.0 if agent.carrying_victim_id is not None else 0.0,
            ], dtype=np.float32)"""
    ),

    # --- Update observation_shapes ---
    (
        """        return {
            "grid_medic": (5, 5, NUM_CHANNELS),
            "grid_scout": (7, 7, NUM_CHANNELS),
            "state": (7,),""",
        """        return {
            "grid_medic": (5, 5, NUM_CHANNELS),
            "grid_scout": (7, 7, NUM_CHANNELS),
            "state": (8,),"""
    ),

    # --- Fix victim decay for carried victims (slower decay when carried) ---
    (
        """        # Victim health decay
        for victim in self.victims:
            if victim.alive:
                decay = self.victim_decay_rate
                if victim.severity == Severity.CRITICAL:
                    decay *= 2.5  # Critical victims decay faster
                elif victim.severity == Severity.SERIOUS:
                    decay *= 1.5
                victim.health -= decay""",
        """        # Victim health decay
        carried_ids = {a.carrying_victim_id for a in self.agents if a.carrying_victim_id is not None}
        for victim in self.victims:
            if victim.alive:
                decay = self.victim_decay_rate
                if victim.severity == Severity.CRITICAL:
                    decay *= 2.5  # Critical victims decay faster
                elif victim.severity == Severity.SERIOUS:
                    decay *= 1.5
                # Carried victims decay slower (being attended to)
                if victim.id in carried_ids:
                    decay *= 0.5
                victim.health -= decay"""
    ),
])


# ================================================================
# 2. spaces.py — STATE_DIM 7 → 8
# ================================================================

print("Patching spaces.py...")
patch_file("src/environment/spaces.py", [
    ("STATE_DIM = 7", "STATE_DIM = 8"),
])


# ================================================================
# 3. networks.py — task_dim 6 → 8
# ================================================================

print("Patching networks.py...")
patch_file("src/agents/networks.py", [
    ("task_dim: int = 6,", "task_dim: int = 8,"),
])


# ================================================================
# 4. buffer.py — state_dim 7 → 8
# ================================================================

print("Patching buffer.py...")
patch_file("src/training/buffer.py", [
    ("state_dim: int = 7,", "state_dim: int = 8,"),
])


# ================================================================
# 5. Update test_environment.py state shape assertion
# ================================================================

print("Patching test_environment.py...")
patch_file("tests/test_environment.py", [
    ('assert state.shape == (7,), f"Agent {agent_id}: state shape {state.shape} != (7,)"',
     'assert state.shape == (8,), f"Agent {agent_id}: state shape {state.shape} != (8,)"'),
])


# ================================================================
# Validate syntax
# ================================================================

print()
print("Validating syntax...")
import ast

files_to_check = [
    "src/environment/crisisgrid.py",
    "src/environment/spaces.py",
    "src/agents/networks.py",
    "src/training/buffer.py",
    "tests/test_environment.py",
]

all_ok = True
for fpath in files_to_check:
    try:
        with open(fpath) as f:
            ast.parse(f.read())
        print(f"  [OK] {fpath}")
    except SyntaxError as e:
        print(f"  [FAIL] {fpath}: {e}")
        all_ok = False

print()
if all_ok:
    print("All patches applied successfully. Run tests:")
    print("  python3 tests/test_environment.py")
    print("  python3 tests/test_agents.py")
    print("  python3 tests/test_training.py")
else:
    print("ERRORS DETECTED — check output above.")
    sys.exit(1)
