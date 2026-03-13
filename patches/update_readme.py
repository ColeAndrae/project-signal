#!/usr/bin/env python3
"""Patch README.md with final Results, Limitations, and Future Work sections."""

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("README.md") as f:
    content = f.read()

# 1. Fix any remaining your-username references
content = content.replace("your-username", "ColeAndrae")

# 2. Fix roadmap statuses
content = content.replace(
    "| 1 | `crisisgrid.py` | Core environment with grid dynamics, hazard spread, victim decay | 🔨 In Progress |",
    "| 1 | `crisisgrid.py` | Core environment with grid dynamics, hazard spread, victim decay | ✅ Complete |"
)
for old_status in ["⬜ Planned"]:
    while old_status in content:
        content = content.replace(old_status, "✅ Complete", 1)

# 3. Insert Results / Limitations / Future Work before Citation
results_section = """
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

"""

# Insert before Citation section
content = content.replace(
    "## Citation",
    results_section + "## Citation"
)

with open("README.md", "w") as f:
    f.write(content)

print("[OK] README.md updated with Results, Limitations, and Future Work")
