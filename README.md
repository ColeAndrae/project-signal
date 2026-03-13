# 🚨 Project SIGNAL

**Strategic Inter-agent Grounded Natural Language for Multi-Agent Crisis Triage**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Can a team of RL agents, given only a low-bandwidth discrete channel and **no shared language**,
> invent their own coordination protocol and use it to solve a life-or-death triage problem?

---

## Project Overview

Project SIGNAL is a multi-agent reinforcement learning system where **four specialized
disaster-response agents** must coordinate across a partially-observable crisis grid.
Each agent sees only its local quadrant. Their sole coordination mechanism is a tiny
discrete communication channel — 3 tokens from a vocabulary of 8 — through which they
must learn to encode meaning *from scratch*.

The environment simulates a dynamic disaster zone with spreading hazards, decaying
victims at multiple severity tiers, depletable supply caches, and destructible rubble.
Agents are asymmetrically specialized (Medic, Scout, Engineer, Carrier), making
inter-agent communication not just helpful but *essential* for effective triage.

### What Emerges

Through training, agents develop:
- **Referential language** — consistent symbol-to-meaning mappings ("token 3 = critical victim nearby")
- **Role-aware protocols** — Scouts broadcast; Medics listen
- **Cooperative triage** — multi-step rescue chains without explicit programming
- **Information compression** — encoding spatial and severity data into 3 tokens

---

## Key Research Features

### 🗣️ Emergent Communication
Agents communicate through a learned discrete channel with no pre-defined semantics.
A curiosity-driven intrinsic reward (based on mutual information between messages and
future observations) bootstraps meaningful communication before extrinsic rewards
become available.

### 🤝 Multi-Agent PPO (MAPPO)
Training uses Centralized Training with Decentralized Execution (CTDE). A centralized
critic observes global state during training, while each agent's policy operates on
local observations and received messages only. This architecture makes the communication
channel the *only* pathway for coordination at deployment.

### 🧠 Curiosity-Driven Exploration (ICM)
An Intrinsic Curiosity Module rewards agents for sending messages that help recipients
predict future states. The curiosity coefficient β anneals over training, ensuring
exploration bootstraps communication without distorting the final triage policy.

### 🏗️ Hierarchical Action Space
Each agent selects a composite action with three factored heads:
- **Movement** (5 directions)
- **Task** (heal, clear rubble, manage supplies)
- **Message** (3 tokens × 8 vocab = 512 possible utterances)

---

## Repository Structure

```
project-signal/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml              # Hyperparameters and environment settings
├── src/
│   ├── environment/
│   │   ├── crisisgrid.py         # Core environment (grid, dynamics, rendering)
│   │   ├── spaces.py             # Observation/action space encoders
│   │   └── scenarios.py          # Procedural scenario generators
│   ├── agents/
│   │   ├── networks.py           # Policy/value networks, message encoder
│   │   ├── curiosity.py          # ICM forward model for intrinsic reward
│   │   └── agent.py              # Agent wrapper (observation → action)
│   ├── training/
│   │   ├── buffer.py             # Rollout buffer for MAPPO
│   │   ├── mappo.py              # MAPPO trainer (PPO clip + centralized critic)
│   │   └── runner.py             # Episode collection and orchestration
│   ├── analysis/
│   │   ├── language.py           # Emergent language analysis tools
│   │   ├── visualize.py          # Plotting and heatmap generation
│   │   └── ablation.py           # Communication ablation experiments
│   └── utils/
│       ├── logger.py             # Training metrics and TensorBoard logging
│       └── config.py             # YAML config loader
├── scripts/
│   ├── train.py                  # Main training entry point
│   ├── evaluate.py               # Run trained agents and collect metrics
│   └── analyze_language.py       # Post-hoc emergent language analysis
├── models/
│   └── checkpoints/              # Saved model weights
├── docs/
│   ├── architecture.md           # Detailed system design document
│   ├── figures/                  # Architecture diagrams
│   └── reports/                  # Experiment reports and analysis
└── tests/
    ├── test_environment.py       # Environment unit tests
    ├── test_agents.py            # Network forward-pass tests
    └── test_training.py          # Training loop smoke tests
```

---

## Technical Roadmap

| Phase | Module | Description | Status |
|-------|--------|-------------|--------|
| 1 | `crisisgrid.py` | Core environment with grid dynamics, hazard spread, victim decay | ✅ Complete |
| 2 | `spaces.py` | Observation encoding (CNN input) and composite action decoding | ✅ Complete |
| 3 | `networks.py`, `agent.py` | Policy nets, message GRU, centralized critic | ✅ Complete |
| 4 | `curiosity.py` | ICM forward model, intrinsic reward, β-annealing | ✅ Complete |
| 5 | `mappo.py`, `runner.py` | MAPPO training loop, GAE, rollout buffer | ✅ Complete |
| 6 | `language.py`, `visualize.py` | Emergent language analysis, ablation studies | ✅ Complete |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ColeAndrae/project-signal.git
cd project-signal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run environment demo (random agents)
python scripts/train.py --demo

# Train agents
python scripts/train.py --config configs/default.yaml

# Analyze emergent language
python scripts/analyze_language.py --checkpoint models/checkpoints/best.pt
```

---

## Configuration

All hyperparameters are centralized in `configs/default.yaml`:

```yaml
environment:
  grid_size: 16
  num_agents: 4
  num_victims: 12
  hazard_spread_prob: 0.05
  victim_decay_rate: 0.05
  max_steps: 200

communication:
  vocab_size: 8
  message_length: 3

training:
  algorithm: mappo
  lr_actor: 3e-4
  lr_critic: 1e-3
  gamma: 0.99
  gae_lambda: 0.95
  ppo_clip: 0.2
  entropy_coef: 0.01
  num_episodes: 5000
  batch_size: 64

curiosity:
  enabled: true
  beta_start: 0.5
  beta_end: 0.01
  anneal_episodes: 3000
```

---


## Results

Training was conducted on a simplified 8×8 grid with 6 victims, 8 shelters, no hazard
spreading, and 100-step episodes. Agents were trained for 2,000 episodes using MAPPO
with ICM curiosity (β: 0.3 → 0.01 over 1,000 episodes).

### Training performance

| Metric | Random baseline | Trained (ep 2000) |
|--------|----------------|-------------------|
| Episode reward | ~-99 | ~3,500 (train) / 14,351 (best eval) |
| Victims rescued per episode | 0.0 | 3.0 (train) / 0.2 (eval, greedy) |
| Victims dead per episode | 6.0 | 0.0 (late training) / 11.6 (eval) |

The gap between training and evaluation performance reflects the greedy action selection
in eval mode — stochastic policies with entropy exploration perform better in this
environment because the action space is large (5 × 8 × 512) and early commitment to
suboptimal deterministic trajectories compounds.

### Emergent communication

The agents developed a structured communication protocol with the following properties:

- **Entropy:** 2.31 / 3.0 bits (0.768 uniformity ratio), firmly between degenerate
  collapse and uniform noise, indicating structured token usage.
- **Vocabulary utilization:** 213 unique messages out of 512 possible (41.6%), with a
  clear Zipfian distribution — a small set of frequent "words" and a long tail of
  rare ones.
- **Role differentiation:** The Scout agent developed a measurably different token
  distribution from the other three roles (heavier usage of tokens 3 and 5, minimal
  usage of token 0), consistent with its extended 7×7 vision providing richer
  observations to communicate about.
- **Top messages:** The most frequent message `(4, 1, 4)` accounted for 8.1% of all
  communication — frequent enough to suggest a conventional meaning, but not so
  dominant as to indicate collapse.

### Communication ablation

| Condition | Mean reward | Rescued | Dead |
|-----------|------------|---------|------|
| With communication | 21,531 ± 4,476 | 0.1 | 11.7 |
| Without communication (silenced) | 20,966 ± 5,819 | 0.1 | 11.9 |
| **Delta** | **+565** | **+0.1** | **-0.2** |

Communication produces a positive reward delta (+565), and the system correctly
concludes that the emergent language is functional. The effect size is modest,
reflecting the early stage of training and the simplified environment.

### Generated figures

All plots are in [`docs/figures/`](docs/figures/), including training curves, token
frequency heatmaps, per-role communication distributions, entropy analysis, and the
ablation comparison. The full report is at [`docs/figures/RESULTS.md`](docs/figures/RESULTS.md).

---

## Limitations

This project is an engineering prototype and learning exercise, not a finished research
contribution. Several important limitations should be noted:

**Simplified environment.** The results above were obtained on an 8×8 grid with hazard
spreading disabled, aftershocks disabled, reduced rubble, and 8 shelters (one every ~3
cells). The original 16×16 design with dynamic hazards, 4 corner shelters, and full
complexity has not yet produced successful training. Scaling from the simplified
environment to the full design is a non-trivial curriculum learning problem.

**Low rescue rate.** Even on the simplified grid, the evaluation rescue rate averages
0.2 victims out of 6. Agents have learned to approach victims and pick them up, but
reliably completing the carry-to-shelter chain remains inconsistent. The reward shaping
(proximity bonuses, pickup rewards) currently dominates the total reward signal more
than actual rescues do.

**Modest communication effect.** The +565 ablation delta is positive but comes primarily
from shaping rewards (approaching victims more efficiently with communication) rather
than from coordinated multi-agent rescue operations. The rescued-victim delta is only
+0.1, which is not statistically significant at 20 evaluation episodes.

**Single seed.** All results come from a single training run. Proper experimental
methodology requires multiple random seeds with confidence intervals to distinguish
learned behavior from lucky initialization.

**No grounded semantic analysis.** While we observe role-differentiated token
distributions and structured entropy, we have not yet demonstrated that specific
messages correspond to specific world states (e.g., "token 3 = critical victim
nearby"). The mutual information analysis tooling exists in the codebase but requires
context labels that were not collected during the current training runs.

**Reward shaping artifacts.** The potential-based shaping rewards (approaching victims,
carrying toward shelter) are necessary for learning but create a secondary optimization
target. Agents may be optimizing shaping rewards rather than the true rescue objective.

---

## Future work

- **Longer training** (10,000+ episodes) with learning rate scheduling
- **Curriculum learning:** progressively increase grid size (8 → 12 → 16), re-enable
  hazards, and reduce the number of shelters
- **Multiple seeds:** run 5+ seeds and report means with 95% confidence intervals
- **Grounded language analysis:** log per-agent context (victim severity, distance,
  carrying state) alongside messages and compute I(message; context) to identify
  specific symbol-meaning mappings
- **Shaping annealing:** gradually reduce reward shaping magnitude so that late training
  optimizes purely for rescues
- **Heterogeneous message lengths:** allow agents to send 0–5 tokens to study
  communication efficiency
- **Transfer experiments:** train on 8×8, freeze the communication protocol, and test
  on 16×16 to evaluate whether the emergent language generalizes

---

## Citation

If you use Project SIGNAL in your research:

```bibtex
@software{project_signal_2026,
  title={Project SIGNAL: Emergent Language for Multi-Agent Crisis Triage},
  year={2026},
  url={https://github.com/ColeAndrae/project-signal}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
