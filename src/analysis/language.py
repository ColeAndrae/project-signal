"""
Emergent language analysis tools for Project SIGNAL.

After training, agents will have developed a discrete communication protocol.
This module provides tools to answer the key research questions:

    1. Did agents develop consistent symbol-to-meaning mappings?
    2. Do different roles communicate differently?
    3. Does communication actually help (ablation)?
    4. What "words" emerged and what do they refer to?

All functions accept message logs and environment state logs collected
during evaluation episodes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def compute_message_entropy(messages: list[np.ndarray], vocab_size: int = 8) -> dict[str, float]:
    """
    Compute entropy statistics for the emergent language.

    High entropy = agents use many different symbols (rich language).
    Low entropy = agents collapsed to a few symbols (degenerate).

    Args:
        messages: List of (num_agents, message_length) arrays, one per timestep.
        vocab_size: Size of the token vocabulary.

    Returns:
        Dict with per-position entropy, total entropy, and uniformity ratio.
    """
    if not messages:
        return {"per_position": [], "mean_entropy": 0.0, "uniformity": 0.0}

    all_msgs = np.stack(messages)  # (T, A, L)
    T, A, L = all_msgs.shape
    flat = all_msgs.reshape(-1, L)  # (T*A, L)

    max_entropy = np.log2(vocab_size)
    per_position = []

    for pos in range(L):
        counts = np.bincount(flat[:, pos], minlength=vocab_size).astype(float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        per_position.append(entropy)

    mean_ent = float(np.mean(per_position))
    uniformity = mean_ent / max_entropy if max_entropy > 0 else 0.0

    return {
        "per_position": per_position,
        "mean_entropy": mean_ent,
        "max_entropy": max_entropy,
        "uniformity": uniformity,  # 1.0 = uniform random, 0.0 = single token
    }


def compute_token_frequencies(
    messages: list[np.ndarray], vocab_size: int = 8,
) -> dict[str, Any]:
    """
    Count how often each token appears at each position.

    Args:
        messages: List of (num_agents, message_length) arrays.
        vocab_size: Token vocabulary size.

    Returns:
        Dict with frequency matrices and most common tokens.
    """
    if not messages:
        return {"frequencies": [], "most_common": []}

    all_msgs = np.stack(messages)
    T, A, L = all_msgs.shape
    flat = all_msgs.reshape(-1, L)

    freq_matrix = np.zeros((L, vocab_size), dtype=int)
    for pos in range(L):
        freq_matrix[pos] = np.bincount(flat[:, pos], minlength=vocab_size)

    # Most common full message
    msg_strings = [tuple(row) for row in flat]
    msg_counter = Counter(msg_strings)
    most_common = msg_counter.most_common(10)

    return {
        "frequencies": freq_matrix,  # (L, V) — count of each token at each position
        "most_common_messages": most_common,
        "unique_messages": len(msg_counter),
        "total_messages": len(msg_strings),
    }


def compute_role_communication_patterns(
    messages: list[np.ndarray],
    num_agents: int = 4,
    role_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze how each agent role uses the communication channel.

    Answers: Do Scouts broadcast more? Do Medics use specific tokens?

    Args:
        messages: List of (num_agents, message_length) arrays.
        num_agents: Number of agents.
        role_names: Names for each role index.

    Returns:
        Per-role token distribution and entropy.
    """
    if role_names is None:
        role_names = ["Medic", "Engineer", "Scout", "Carrier"]

    if not messages:
        return {name: {} for name in role_names}

    all_msgs = np.stack(messages)  # (T, A, L)
    T, A, L = all_msgs.shape

    results = {}
    for agent_id in range(min(num_agents, A)):
        agent_msgs = all_msgs[:, agent_id, :]  # (T, L)
        flat_tokens = agent_msgs.flatten()
        counts = np.bincount(flat_tokens, minlength=8).astype(float)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        entropy = -np.sum(probs_nz * np.log2(probs_nz))

        # Most common full messages for this agent
        msg_tuples = [tuple(row) for row in agent_msgs]
        top_msgs = Counter(msg_tuples).most_common(5)

        results[role_names[agent_id]] = {
            "token_distribution": counts.astype(int).tolist(),
            "entropy": float(entropy),
            "top_messages": top_msgs,
        }

    return results


def compute_message_context_correlation(
    messages: list[np.ndarray],
    env_states: list[dict[str, Any]],
    num_agents: int = 4,
) -> dict[str, Any]:
    """
    Correlate messages with world states to discover grounded meanings.

    For each unique message, collect the world states in which it was sent.
    If a message consistently appears when a CRITICAL victim is nearby,
    we've found a grounded referential symbol.

    Args:
        messages: List of (num_agents, message_length) arrays.
        env_states: List of dicts with keys like 'victim_nearby', 'hazard_nearby', etc.
            Each dict should have per-agent lists.

    Returns:
        Dict mapping message tuples to their associated context statistics.
    """
    if not messages or not env_states:
        return {"groundings": {}}

    groundings = defaultdict(lambda: {
        "count": 0,
        "critical_nearby": 0,
        "serious_nearby": 0,
        "stable_nearby": 0,
        "hazard_nearby": 0,
        "supply_nearby": 0,
        "rubble_nearby": 0,
    })

    for t, (msg_step, state) in enumerate(zip(messages, env_states)):
        for agent_id in range(min(num_agents, msg_step.shape[0])):
            msg_tuple = tuple(msg_step[agent_id])
            g = groundings[msg_tuple]
            g["count"] += 1

            # Accumulate context if available
            if "critical_nearby" in state:
                if state["critical_nearby"][agent_id]:
                    g["critical_nearby"] += 1
            if "serious_nearby" in state:
                if state["serious_nearby"][agent_id]:
                    g["serious_nearby"] += 1
            if "stable_nearby" in state:
                if state["stable_nearby"][agent_id]:
                    g["stable_nearby"] += 1
            if "hazard_nearby" in state:
                if state["hazard_nearby"][agent_id]:
                    g["hazard_nearby"] += 1
            if "supply_nearby" in state:
                if state["supply_nearby"][agent_id]:
                    g["supply_nearby"] += 1
            if "rubble_nearby" in state:
                if state["rubble_nearby"][agent_id]:
                    g["rubble_nearby"] += 1

    # Convert to percentages
    for msg_tuple, g in groundings.items():
        count = g["count"]
        if count > 0:
            for key in list(g.keys()):
                if key != "count":
                    g[f"{key}_pct"] = g[key] / count

    # Sort by frequency
    sorted_groundings = dict(
        sorted(groundings.items(), key=lambda x: x[1]["count"], reverse=True)
    )

    return {"groundings": sorted_groundings}


def compute_mutual_information(
    messages: list[np.ndarray],
    context_labels: list[np.ndarray],
    vocab_size: int = 8,
) -> float:
    """
    Compute mutual information I(message_token; context_label).

    Higher MI = messages carry more information about the world state.
    This is the key metric for evaluating emergent communication quality.

    Args:
        messages: List of (num_agents, message_length) arrays.
        context_labels: List of (num_agents,) integer arrays encoding the
            dominant context for each agent (e.g., 0=nothing, 1=critical_nearby,
            2=hazard_nearby, etc.)
        vocab_size: Token vocabulary size.

    Returns:
        Mutual information in bits.
    """
    if not messages or not context_labels:
        return 0.0

    all_msgs = np.concatenate([m.reshape(-1, m.shape[-1]) for m in messages])  # (N, L)
    all_ctx = np.concatenate([c.flatten() for c in context_labels])  # (N,)
    N = min(len(all_msgs), len(all_ctx))
    all_msgs = all_msgs[:N]
    all_ctx = all_ctx[:N]

    # Use first token position for simplicity
    tokens = all_msgs[:, 0]
    num_ctx = int(all_ctx.max()) + 1

    # Joint distribution P(token, context)
    joint = np.zeros((vocab_size, num_ctx))
    for tok, ctx in zip(tokens, all_ctx):
        joint[int(tok), int(ctx)] += 1
    joint /= joint.sum()

    # Marginals
    p_tok = joint.sum(axis=1)
    p_ctx = joint.sum(axis=0)

    # MI = sum P(t,c) * log2(P(t,c) / (P(t) * P(c)))
    mi = 0.0
    for t in range(vocab_size):
        for c in range(num_ctx):
            if joint[t, c] > 0 and p_tok[t] > 0 and p_ctx[c] > 0:
                mi += joint[t, c] * np.log2(joint[t, c] / (p_tok[t] * p_ctx[c]))

    return float(mi)


def generate_analysis_report(
    messages: list[np.ndarray],
    episode_infos: list[dict[str, Any]],
    vocab_size: int = 8,
) -> str:
    """
    Generate a human-readable analysis report of the emergent language.

    Args:
        messages: Collected messages from evaluation episodes.
        episode_infos: List of episode info dicts from runner.

    Returns:
        Formatted string report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  PROJECT SIGNAL — Emergent Language Analysis Report")
    lines.append("=" * 70)
    lines.append("")

    # Entropy
    entropy = compute_message_entropy(messages, vocab_size)
    lines.append("1. MESSAGE ENTROPY")
    lines.append(f"   Mean entropy:     {entropy['mean_entropy']:.3f} bits")
    lines.append(f"   Max entropy:      {entropy['max_entropy']:.3f} bits")
    lines.append(f"   Uniformity ratio: {entropy['uniformity']:.3f}")
    if entropy["uniformity"] < 0.3:
        lines.append("   WARNING: Low uniformity — language may have collapsed.")
    elif entropy["uniformity"] > 0.8:
        lines.append("   NOTE: High uniformity — messages may still be near-random.")
    else:
        lines.append("   GOOD: Moderate entropy suggests structured communication.")
    lines.append("")

    # Token frequencies
    freq = compute_token_frequencies(messages, vocab_size)
    lines.append("2. TOKEN FREQUENCIES")
    lines.append(f"   Unique messages: {freq['unique_messages']} / {freq['total_messages']}")
    lines.append(f"   Top 5 messages:")
    for msg, count in freq.get("most_common_messages", [])[:5]:
        pct = 100.0 * count / max(freq["total_messages"], 1)
        lines.append(f"     {msg} — {count}x ({pct:.1f}%)")
    lines.append("")

    # Role patterns
    roles = compute_role_communication_patterns(messages)
    lines.append("3. ROLE COMMUNICATION PATTERNS")
    for role_name, data in roles.items():
        if "entropy" in data:
            lines.append(f"   {role_name}:")
            lines.append(f"     Entropy: {data['entropy']:.3f} bits")
            lines.append(f"     Token dist: {data['token_distribution']}")
    lines.append("")

    # Performance summary
    if episode_infos:
        rewards = [e.get("episode_reward", 0) for e in episode_infos]
        rescued = [e.get("victims_rescued", 0) for e in episode_infos]
        dead = [e.get("victims_dead", 0) for e in episode_infos]
        lines.append("4. PERFORMANCE SUMMARY")
        lines.append(f"   Episodes evaluated: {len(episode_infos)}")
        lines.append(f"   Mean reward:  {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        lines.append(f"   Mean rescued: {np.mean(rescued):.1f}")
        lines.append(f"   Mean dead:    {np.mean(dead):.1f}")
        total_victims = np.mean([e.get("victims_rescued", 0) + e.get("victims_dead", 0) + e.get("victims_alive", 0) for e in episode_infos])
        if total_victims > 0:
            survival_rate = np.mean(rescued) / total_victims * 100
            lines.append(f"   Survival rate: {survival_rate:.1f}%")
    lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
