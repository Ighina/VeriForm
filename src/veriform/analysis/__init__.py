"""
Analysis and visualization module for benchmark results.
"""

from .statistics import (
    compute_correlation,
    compute_error_rates,
    analyze_faithfulness,
    compute_summary_statistics,
    test_significance
)
from .visualization import (
    plot_error_rate_vs_perturbation,
    plot_correlation_heatmap,
    plot_per_run_statistics,
    create_summary_report
)

__all__ = [
    "compute_correlation",
    "compute_error_rates",
    "analyze_faithfulness",
    "compute_summary_statistics",
    "test_significance",
    "plot_error_rate_vs_perturbation",
    "plot_correlation_heatmap",
    "plot_per_run_statistics",
    "create_summary_report",
]
