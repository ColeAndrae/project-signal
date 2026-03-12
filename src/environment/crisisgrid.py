"""
CrisisGrid — Multi-Agent Disaster Response Environment for Project SIGNAL.

A 16x16 grid world where 4 specialized agents must coordinate via emergent
communication to triage victims in a dynamic disaster zone.

Design principles:
    - Gym-like API: reset() → observations, step(actions) → (obs, reward, done, info)
    - Partial observability: each agent sees only its local vision window
    - Communication is part of the action/observation interface, not the env
    - Deterministic given a seed; stochastic dynamics via internal RNG
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


# ============================================================
# Constants & Enumerations
# ============================================================

class CellType(IntEnum):
    """Grid cell terrain types."""
    EMPTY = 0
    SHELTER = 1
    HAZARD = 2
    RUBBLE = 3
    SUPPLY = 4


class Severity(IntEnum):
    """Victim severity tiers — higher value = more urgent."""
    STABLE = 1
    SERIOUS = 2
    CRITICAL = 3


class Role(IntEnum):
    """Agent specialization roles."""
    MEDIC = 0
    ENGINEER = 1
    SCOUT = 2
    CARRIER = 3


class MoveAction(IntEnum):
    """Movement directions."""
    STAY = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


class TaskAction(IntEnum):
    """Task actions an agent can perform."""
    NOOP = 0
    HEAL = 1
    CLEAR_RUBBLE = 2
    PICKUP_SUPPLY = 3
    DROP_SUPPLY = 4
    USE_SUPPLY = 5


# Direction deltas: (row_delta, col_delta)
MOVE_DELTAS = {
    MoveAction.STAY: (0, 0),
    MoveAction.NORTH: (-1, 0),
    MoveAction.SOUTH: (1, 0),
    MoveAction.EAST: (0, 1),
    MoveAction.WEST: (0, -1),
}

# Observation grid channels
CH_TERRAIN = 0      # CellType encoded as float
CH_HAZARD = 1       # Hazard intensity (0.0 - 1.0)
CH_VICTIM = 2       # 1.0 if victim present
CH_SEVERITY = 3     # Severity / 3.0 (normalized)
CH_SUPPLY = 4       # 1.0 if supply cache present
CH_AGENT = 5        # Agent role_id + 1 if present, 0 otherwise
NUM_CHANNELS = 6

# Role-specific parameters
VISION_RADIUS = {Role.MEDIC: 2, Role.ENGINEER: 2, Role.SCOUT: 3, Role.CARRIER: 2}
CARRY_CAPACITY = {Role.MEDIC: 1, Role.ENGINEER: 1, Role.SCOUT: 1, Role.CARRIER: 3}
RUBBLE_CLEAR_SPEED = {Role.MEDIC: 3, Role.ENGINEER: 1, Role.SCOUT: 3, Role.CARRIER: 3}
HEAL_POWER = {Role.MEDIC: 1.0, Role.ENGINEER: 0.2, Role.SCOUT: 0.2, Role.CARRIER: 0.2}

# Severity multipliers for reward
SEVERITY_MULT = {Severity.STABLE: 1.0, Severity.SERIOUS: 2.0, Severity.CRITICAL: 3.0}


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Victim:
    """A victim in the disaster zone."""
    row: int
    col: int
    severity: Severity
    health: float = 1.0          # 1.0 = full, 0.0 = dead
    stabilized: bool = False     # True if severity was reduced this episode
    rescued: bool = False        # True if delivered to shelter
    id: int = 0

    @property
    def alive(self) -> bool:
        return self.health > 0.0 and not self.rescued


@dataclass
class AgentState:
    """Mutable state for a single agent."""
    row: int
    col: int
    role: Role
    health: float = 1.0
    supplies_held: int = 0
    rubble_progress: dict[tuple[int, int], int] = field(default_factory=dict)


# ============================================================
# CrisisGrid Environment
# ============================================================

class CrisisGrid:
    """
    Multi-agent disaster response grid environment.

    Gym-like interface:
        obs = env.reset(seed=42)
        obs, reward, done, info = env.step(actions)
        env.render()

    Actions per agent: dict with keys 'move', 'task', 'message'
        - move: int in [0, 4] (MoveAction)
        - task: int in [0, 5] (TaskAction)
        - message: list[int] of length message_length, each in [0, vocab_size)

    Observations per agent: dict with keys:
        - 'grid': np.ndarray of shape (vision_size, vision_size, NUM_CHANNELS)
        - 'state': np.ndarray of shape (7,) — [row, col, health, supplies, role, step, max_step]
        - 'messages': np.ndarray of shape (num_agents-1, message_length)
    """

    def __init__(
        self,
        grid_size: int = 16,
        num_agents: int = 4,
        num_victims: int = 12,
        num_supplies: int = 6,
        num_rubble: int = 10,
        hazard_spread_prob: float = 0.05,
        victim_decay_rate: float = 0.05,
        aftershock_prob: float = 0.02,
        max_steps: int = 200,
        vocab_size: int = 8,
        message_length: int = 3,
        reward_config: dict[str, Any] | None = None,
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_victims = num_victims
        self.num_supplies = num_supplies
        self.num_rubble = num_rubble
        self.hazard_spread_prob = hazard_spread_prob
        self.victim_decay_rate = victim_decay_rate
        self.aftershock_prob = aftershock_prob
        self.max_steps = max_steps
        self.vocab_size = vocab_size
        self.message_length = message_length

        # Reward parameters
        rc = reward_config or {}
        self.r_rescued = rc.get("victim_rescued", 10.0)
        self.r_stabilized = rc.get("victim_stabilized", 3.0)
        self.r_died = rc.get("victim_died", -5.0)
        self.r_step = rc.get("step_cost", -0.1)
        self.r_hazard = rc.get("hazard_damage", -1.0)

        # Mutable state — initialized in reset()
        self.terrain: np.ndarray | None = None
        self.hazard_map: np.ndarray | None = None
        self.agents: list[AgentState] = []
        self.victims: list[Victim] = []
        self.supply_map: np.ndarray | None = None
        self.step_count: int = 0
        self._rng: np.random.Generator | None = None
        self._victim_counter: int = 0

        # Message buffer: messages[i] = message sent BY agent i last step
        self._messages = np.zeros((num_agents, message_length), dtype=np.int64)

        # Shelter positions (four corners)
        self._shelters = [
            (0, 0), (0, grid_size - 1),
            (grid_size - 1, 0), (grid_size - 1, grid_size - 1),
        ]

    # --------------------------------------------------------
    # Core API
    # --------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict[int, dict[str, np.ndarray]]:
        """Reset environment and return initial observations for all agents."""
        self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self._victim_counter = 0
        self._messages = np.zeros((self.num_agents, self.message_length), dtype=np.int64)

        # Initialize terrain grid
        self.terrain = np.full((self.grid_size, self.grid_size), CellType.EMPTY, dtype=np.int32)
        self.hazard_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.supply_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place shelters
        for r, c in self._shelters:
            self.terrain[r, c] = CellType.SHELTER

        # Place initial hazard cluster (2-3 connected cells near center)
        self._place_hazard_cluster()

        # Place rubble
        self._place_items(CellType.RUBBLE, self.num_rubble)

        # Place supply caches
        for _ in range(self.num_supplies):
            try:
                r, c = self._random_empty_cell()
            except RuntimeError:
                continue
            self.supply_map[r, c] += 1
            self.terrain[r, c] = CellType.SUPPLY

        # Place victims
        self.victims = []
        for _ in range(self.num_victims):
            try:
                r, c = self._random_empty_cell()
            except RuntimeError:
                continue
            sev = self._rng.choice(
                [Severity.STABLE, Severity.SERIOUS, Severity.CRITICAL],
                p=[0.4, 0.35, 0.25],
            )
            self.victims.append(Victim(
                row=r, col=c, severity=sev, id=self._victim_counter,
            ))
            self._victim_counter += 1

        # Place agents — one per quadrant
        self.agents = []
        quadrant_centers = [
            (self.grid_size // 4, self.grid_size // 4),
            (self.grid_size // 4, 3 * self.grid_size // 4),
            (3 * self.grid_size // 4, self.grid_size // 4),
            (3 * self.grid_size // 4, 3 * self.grid_size // 4),
        ]
        roles = [Role.MEDIC, Role.ENGINEER, Role.SCOUT, Role.CARRIER]
        for i, role in enumerate(roles):
            cr, cc = quadrant_centers[i]
            # Find nearest empty cell to quadrant center
            r, c = self._nearest_empty(cr, cc)
            self.agents.append(AgentState(row=r, col=c, role=role))

        return self._build_observations()

    def step(
        self, actions: dict[int, dict[str, Any]]
    ) -> tuple[dict[int, dict], float, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            actions: {agent_id: {'move': int, 'task': int, 'message': list[int]}}

        Returns:
            observations: per-agent observation dicts
            reward: shared team reward (float)
            done: whether episode has ended
            info: diagnostic information
        """
        self.step_count += 1
        reward = 0.0
        info = {"victims_rescued": 0, "victims_died": 0, "victims_stabilized": 0}

        # --- Phase 1: Collect messages ---
        for agent_id, action in actions.items():
            msg = np.array(action.get("message", [0] * self.message_length), dtype=np.int64)
            msg = np.clip(msg, 0, self.vocab_size - 1)
            self._messages[agent_id] = msg

        # --- Phase 2: Move agents ---
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
            agent.row, agent.col = nr, nc

        # --- Phase 3: Execute tasks ---
        for agent_id, action in actions.items():
            task = TaskAction(action.get("task", 0))
            agent = self.agents[agent_id]
            r, c = agent.row, agent.col

            if task == TaskAction.HEAL:
                # Heal victims at current location
                for victim in self.victims:
                    if victim.alive and victim.row == r and victim.col == c:
                        heal_amt = HEAL_POWER[agent.role]
                        victim.health = min(1.0, victim.health + heal_amt)
                        # Stabilization: Medic can downgrade severity
                        if (agent.role == Role.MEDIC
                                and victim.severity == Severity.CRITICAL
                                and not victim.stabilized):
                            victim.severity = Severity.SERIOUS
                            victim.stabilized = True
                            reward += self.r_stabilized * SEVERITY_MULT[Severity.CRITICAL]
                            info["victims_stabilized"] += 1

            elif task == TaskAction.CLEAR_RUBBLE:
                # Check adjacent cells for rubble
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    tr, tc = r + dr, c + dc
                    if (0 <= tr < self.grid_size and 0 <= tc < self.grid_size
                            and self.terrain[tr, tc] == CellType.RUBBLE):
                        pos = (tr, tc)
                        needed = RUBBLE_CLEAR_SPEED[agent.role]
                        agent.rubble_progress[pos] = agent.rubble_progress.get(pos, 0) + 1
                        if agent.rubble_progress[pos] >= needed:
                            self.terrain[tr, tc] = CellType.EMPTY
                            agent.rubble_progress.pop(pos, None)
                        break  # Clear one rubble per step

            elif task == TaskAction.PICKUP_SUPPLY:
                cap = CARRY_CAPACITY[agent.role]
                if self.supply_map[r, c] > 0 and agent.supplies_held < cap:
                    self.supply_map[r, c] -= 1
                    agent.supplies_held += 1
                    if self.supply_map[r, c] == 0:
                        self.terrain[r, c] = CellType.EMPTY

            elif task == TaskAction.DROP_SUPPLY:
                if agent.supplies_held > 0:
                    agent.supplies_held -= 1
                    self.supply_map[r, c] += 1
                    self.terrain[r, c] = CellType.SUPPLY

            elif task == TaskAction.USE_SUPPLY:
                # Use supply on a victim at current location
                if agent.supplies_held > 0:
                    for victim in self.victims:
                        if victim.alive and victim.row == r and victim.col == c:
                            victim.health = min(1.0, victim.health + 0.3)
                            agent.supplies_held -= 1
                            break

        # --- Phase 4: Check rescues (victim at shelter) ---
        shelter_set = set(self._shelters)
        for victim in self.victims:
            if victim.alive and (victim.row, victim.col) in shelter_set:
                victim.rescued = True
                mult = SEVERITY_MULT[victim.severity]
                reward += self.r_rescued * mult
                info["victims_rescued"] += 1

        # --- Phase 5: Environment dynamics ---

        # Hazard spreading
        new_hazard = self.hazard_map.copy()
        hazard_cells = np.argwhere(self.hazard_map > 0)
        for hr, hc in hazard_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = hr + dr, hc + dc
                if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                        and self.terrain[nr, nc] not in (CellType.SHELTER, CellType.RUBBLE)
                        and new_hazard[nr, nc] == 0):
                    if self._rng.random() < self.hazard_spread_prob:
                        new_hazard[nr, nc] = 0.5
                        self.terrain[nr, nc] = CellType.HAZARD
        self.hazard_map = new_hazard

        # Victim health decay
        for victim in self.victims:
            if victim.alive:
                decay = self.victim_decay_rate
                if victim.severity == Severity.CRITICAL:
                    decay *= 2.5  # Critical victims decay faster
                elif victim.severity == Severity.SERIOUS:
                    decay *= 1.5
                victim.health -= decay

                # Death check
                if victim.health <= 0.0:
                    victim.health = 0.0
                    mult = SEVERITY_MULT[victim.severity]
                    reward += self.r_died * mult
                    info["victims_died"] += 1

        # Aftershock — new victim
        if self._rng.random() < self.aftershock_prob:
            try:
                r, c = self._random_empty_cell()
            except RuntimeError:
                pass  # grid is full, skip this aftershock
            else:
                sev = self._rng.choice(
                    [Severity.STABLE, Severity.SERIOUS, Severity.CRITICAL],
                    p=[0.3, 0.4, 0.3],
                )
                self.victims.append(Victim(
                    row=r, col=c, severity=sev, id=self._victim_counter,
                ))
                self._victim_counter += 1

        # Agent hazard damage
        for agent in self.agents:
            if self.hazard_map[agent.row, agent.col] > 0:
                agent.health -= 0.1
                reward += self.r_hazard

        # Step cost
        reward += self.r_step * self.num_agents

        # --- Done condition ---
        all_victims_resolved = all(
            not v.alive for v in self.victims
        )
        done = self.step_count >= self.max_steps or all_victims_resolved

        # --- Final info ---
        alive_victims = sum(1 for v in self.victims if v.alive)
        info["alive_victims"] = alive_victims
        info["step"] = self.step_count
        info["total_victims_spawned"] = self._victim_counter

        return self._build_observations(), reward, done, info

    def render(self, mode: str = "ascii") -> str | None:
        """Render the grid as colored ASCII art."""
        if mode != "ascii":
            raise ValueError(f"Unsupported render mode: {mode}")

        # Symbol map
        grid_chars = []
        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                ch = "."
                if self.terrain[r, c] == CellType.SHELTER:
                    ch = "H"
                elif self.terrain[r, c] == CellType.HAZARD or self.hazard_map[r, c] > 0:
                    ch = "~"
                elif self.terrain[r, c] == CellType.RUBBLE:
                    ch = "#"
                elif self.supply_map[r, c] > 0:
                    ch = "+"

                # Victims override terrain display
                for v in self.victims:
                    if v.alive and v.row == r and v.col == c:
                        ch = str(int(v.severity))
                        break

                # Agents override everything
                for i, a in enumerate(self.agents):
                    if a.row == r and a.col == c:
                        ch = ["M", "E", "S", "C"][a.role]
                        break

                row_chars.append(ch)
            grid_chars.append(" ".join(row_chars))

        # Build legend
        header = (
            f"Step {self.step_count}/{self.max_steps} | "
            f"Alive: {sum(1 for v in self.victims if v.alive)} | "
            f"Rescued: {sum(1 for v in self.victims if v.rescued)} | "
            f"Dead: {sum(1 for v in self.victims if v.health <= 0 and not v.rescued)}"
        )
        legend = (
            "Legend: M=Medic E=Engineer S=Scout C=Carrier | "
            "1=Stable 2=Serious 3=Critical | H=Shelter #=Rubble ~=Hazard +=Supply"
        )
        output = "\n".join([header, "-" * len(header), *grid_chars, "-" * len(header), legend])
        print(output)
        return output

    # --------------------------------------------------------
    # Observation Building
    # --------------------------------------------------------

    def _build_observations(self) -> dict[int, dict[str, np.ndarray]]:
        """Build per-agent observations with partial observability."""
        obs = {}
        for i, agent in enumerate(self.agents):
            radius = VISION_RADIUS[agent.role]
            vision_size = 2 * radius + 1

            # Local grid observation (padded if near edges)
            grid_obs = np.zeros((vision_size, vision_size, NUM_CHANNELS), dtype=np.float32)

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    gr, gc = agent.row + dr, agent.col + dc
                    lr, lc = dr + radius, dc + radius  # local coords

                    if 0 <= gr < self.grid_size and 0 <= gc < self.grid_size:
                        grid_obs[lr, lc, CH_TERRAIN] = float(self.terrain[gr, gc]) / 4.0
                        grid_obs[lr, lc, CH_HAZARD] = self.hazard_map[gr, gc]
                        grid_obs[lr, lc, CH_SUPPLY] = float(self.supply_map[gr, gc] > 0)

                        # Victim info
                        for v in self.victims:
                            if v.alive and v.row == gr and v.col == gc:
                                grid_obs[lr, lc, CH_VICTIM] = 1.0
                                grid_obs[lr, lc, CH_SEVERITY] = float(v.severity) / 3.0
                                break

                        # Other agents
                        for j, other in enumerate(self.agents):
                            if j != i and other.row == gr and other.col == gc:
                                grid_obs[lr, lc, CH_AGENT] = float(other.role + 1) / 4.0
                                break
                    else:
                        # Out of bounds — encode as wall
                        grid_obs[lr, lc, CH_TERRAIN] = -1.0

            # Agent state vector
            state_vec = np.array([
                agent.row / self.grid_size,
                agent.col / self.grid_size,
                agent.health,
                agent.supplies_held / CARRY_CAPACITY[agent.role],
                float(agent.role) / 3.0,
                self.step_count / self.max_steps,
                float(self.max_steps - self.step_count) / self.max_steps,
            ], dtype=np.float32)

            # Messages from other agents
            other_msgs = np.delete(self._messages, i, axis=0)  # (num_agents-1, msg_len)

            obs[i] = {
                "grid": grid_obs,
                "state": state_vec,
                "messages": other_msgs.copy(),
            }

        return obs

    # --------------------------------------------------------
    # Procedural Generation Helpers
    # --------------------------------------------------------

    def _random_empty_cell(self) -> tuple[int, int]:
        """Find a random empty cell not occupied by agents or shelters."""
        shelter_set = set(self._shelters)
        agent_set = {(a.row, a.col) for a in self.agents}
        victim_set = {(v.row, v.col) for v in self.victims if v.alive}
        occupied = shelter_set | agent_set | victim_set

        for _ in range(1000):
            r = int(self._rng.integers(0, self.grid_size))
            c = int(self._rng.integers(0, self.grid_size))
            if (r, c) not in occupied and self.terrain[r, c] == CellType.EMPTY:
                return r, c

        # Fallback: exhaustive search
        empties = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in occupied and self.terrain[r, c] == CellType.EMPTY:
                    empties.append((r, c))
        if empties:
            idx = int(self._rng.integers(0, len(empties)))
            return empties[idx]
        raise RuntimeError("No empty cells available")

    def _nearest_empty(self, target_r: int, target_c: int) -> tuple[int, int]:
        """Find the nearest empty cell to a target position (BFS)."""
        from collections import deque
        visited = set()
        queue = deque([(target_r, target_c)])
        agent_set = {(a.row, a.col) for a in self.agents}

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            if (0 <= r < self.grid_size and 0 <= c < self.grid_size
                    and self.terrain[r, c] == CellType.EMPTY
                    and (r, c) not in agent_set):
                return r, c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    queue.append((nr, nc))

        return target_r, target_c  # fallback

    def _place_hazard_cluster(self) -> None:
        """Place a connected cluster of 2-4 hazard cells near the center."""
        center = self.grid_size // 2
        r = int(self._rng.integers(center - 3, center + 3))
        c = int(self._rng.integers(center - 3, center + 3))
        r = np.clip(r, 1, self.grid_size - 2)
        c = np.clip(c, 1, self.grid_size - 2)

        self.terrain[r, c] = CellType.HAZARD
        self.hazard_map[r, c] = 1.0

        cluster_size = int(self._rng.integers(2, 5))
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        self._rng.shuffle(candidates)

        for cr, cc in candidates[:cluster_size - 1]:
            if 0 <= cr < self.grid_size and 0 <= cc < self.grid_size:
                if self.terrain[cr, cc] == CellType.EMPTY:
                    self.terrain[cr, cc] = CellType.HAZARD
                    self.hazard_map[cr, cc] = 1.0

    def _place_items(self, cell_type: CellType, count: int) -> None:
        """Place N items of a given cell type on empty cells."""
        for _ in range(count):
            try:
                r, c = self._random_empty_cell()
            except RuntimeError:
                continue
            self.terrain[r, c] = cell_type

    # --------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------

    def get_global_state(self) -> np.ndarray:
        """
        Full global state tensor for the centralized critic.
        Shape: (grid_size, grid_size, NUM_CHANNELS + num_agents)
        """
        state = np.zeros(
            (self.grid_size, self.grid_size, NUM_CHANNELS + self.num_agents),
            dtype=np.float32,
        )

        # Base channels
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                state[r, c, CH_TERRAIN] = float(self.terrain[r, c]) / 4.0
                state[r, c, CH_HAZARD] = self.hazard_map[r, c]
                state[r, c, CH_SUPPLY] = float(self.supply_map[r, c] > 0)

        for v in self.victims:
            if v.alive:
                state[v.row, v.col, CH_VICTIM] = 1.0
                state[v.row, v.col, CH_SEVERITY] = float(v.severity) / 3.0

        # Per-agent channels (one-hot position per agent)
        for i, a in enumerate(self.agents):
            state[a.row, a.col, NUM_CHANNELS + i] = 1.0

        return state

    def alive_victim_count(self) -> int:
        return sum(1 for v in self.victims if v.alive)

    def rescued_victim_count(self) -> int:
        return sum(1 for v in self.victims if v.rescued)

    def dead_victim_count(self) -> int:
        return sum(1 for v in self.victims if v.health <= 0 and not v.rescued)

    @property
    def observation_shapes(self) -> dict[str, tuple]:
        """Return observation shapes for network construction."""
        return {
            "grid_medic": (5, 5, NUM_CHANNELS),
            "grid_scout": (7, 7, NUM_CHANNELS),
            "state": (7,),
            "messages": (self.num_agents - 1, self.message_length),
        }

    @property
    def action_dims(self) -> dict[str, int]:
        """Return action dimensions for network heads."""
        return {
            "move": len(MoveAction),
            "task": len(TaskAction),
            "message_per_token": self.vocab_size,
            "message_tokens": self.message_length,
        }
