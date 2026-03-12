"""Analysis tools for Project SIGNAL emergent communication."""

from .language import (
    compute_message_entropy,
    compute_token_frequencies,
    compute_role_communication_patterns,
    compute_message_context_correlation,
    compute_mutual_information,
    generate_analysis_report,
)

from .ablation import run_ablation, format_ablation_report

__all__ = [
    "compute_message_entropy",
    "compute_token_frequencies",
    "compute_role_communication_patterns",
    "compute_message_context_correlation",
    "compute_mutual_information",
    "generate_analysis_report",
    "run_ablation",
    "format_ablation_report",
]
