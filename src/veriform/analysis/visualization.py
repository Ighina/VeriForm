"""
Visualization utilities for benchmark results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from veriform.benchmark import BenchmarkResults
from .statistics import compute_error_rates, analyze_faithfulness


# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_error_rate_vs_perturbation(
    results: BenchmarkResults,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot error rate vs perturbation probability.

    Args:
        results: BenchmarkResults to visualize
        output_path: Optional path to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    error_rates = compute_error_rates(results)
    probabilities = error_rates["probabilities"]
    ver_errors = error_rates["verification_error_rates"]
    form_errors = error_rates["formalization_error_rates"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Verification error rate
    ax1.plot(probabilities, ver_errors, 'o-', linewidth=2, markersize=8, label='Verification Error Rate')
    ax1.set_xlabel('Perturbation Probability', fontsize=12)
    ax1.set_ylabel('Verification Error Rate', fontsize=12)
    ax1.set_title('Verification Errors vs Perturbation Probability', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, max(ver_errors) * 1.1)

    # Add trend line
    z = np.polyfit(probabilities, ver_errors, 1)
    p = np.poly1d(z)
    ax1.plot(probabilities, p(probabilities), "--", alpha=0.5, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')
    ax1.legend()

    # Formalization error rate
    ax2.plot(probabilities, form_errors, 's-', linewidth=2, markersize=8, label='Formalization Error Rate', color='coral')
    ax2.set_xlabel('Perturbation Probability', fontsize=12)
    ax2.set_ylabel('Formalization Error Rate', fontsize=12)
    ax2.set_title('Formalization Errors vs Perturbation Probability', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, max(form_errors) * 1.1 if form_errors else 1.0)

    # Add trend line
    z2 = np.polyfit(probabilities, form_errors, 1)
    p2 = np.poly1d(z2)
    ax2.plot(probabilities, p2(probabilities), "--", alpha=0.5, label=f'Linear fit: y={z2[0]:.2f}x+{z2[1]:.2f}')
    ax2.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()

    return fig


def plot_correlation_heatmap(
    results: BenchmarkResults,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot correlation heatmap of key metrics.

    Args:
        results: BenchmarkResults to visualize
        output_path: Optional path to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    error_rates = compute_error_rates(results)

    # Build data matrix
    data = {
        'Probability': error_rates["probabilities"],
        'Ver. Errors': error_rates["verification_error_rates"],
        'Form. Errors': error_rates["formalization_error_rates"],
    }

    # Compute correlation matrix
    import pandas as pd
    df = pd.DataFrame(data)
    corr_matrix = df.corr()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Correlation Matrix of Key Metrics', fontsize=14, pad=20)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")

    if show:
        plt.show()

    return fig


def plot_per_run_statistics(
    results: BenchmarkResults,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot detailed statistics for each run.

    Args:
        results: BenchmarkResults to visualize
        output_path: Optional path to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    probabilities = [run.probability for run in results.runs]
    perturbed_steps = [run.perturbed_steps for run in results.runs]
    total_steps = [run.total_steps for run in results.runs]
    ver_failures = [run.verification_failures for run in results.runs]
    ver_successes = [run.verification_successes for run in results.runs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Perturbation rates
    actual_pert_rates = [p / t if t > 0 else 0 for p, t in zip(perturbed_steps, total_steps)]
    axes[0, 0].bar(range(len(probabilities)), actual_pert_rates, alpha=0.7, label='Actual')
    axes[0, 0].plot(range(len(probabilities)), probabilities, 'r--', marker='o', label='Expected')
    axes[0, 0].set_xlabel('Run Index')
    axes[0, 0].set_ylabel('Perturbation Rate')
    axes[0, 0].set_title('Actual vs Expected Perturbation Rates')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Verification results
    x = np.arange(len(probabilities))
    width = 0.35
    axes[0, 1].bar(x - width/2, ver_failures, width, label='Failures', alpha=0.7)
    axes[0, 1].bar(x + width/2, ver_successes, width, label='Successes', alpha=0.7)
    axes[0, 1].set_xlabel('Run Index')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Verification Results by Run')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Steps processed
    axes[1, 0].bar(range(len(probabilities)), total_steps, alpha=0.7)
    axes[1, 0].set_xlabel('Run Index')
    axes[1, 0].set_ylabel('Total Steps')
    axes[1, 0].set_title('Steps Processed per Run')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Success rate
    success_rates = [s / (s + f) if (s + f) > 0 else 0 for s, f in zip(ver_successes, ver_failures)]
    axes[1, 1].plot(probabilities, success_rates, 'o-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Perturbation Probability')
    axes[1, 1].set_ylabel('Verification Success Rate')
    axes[1, 1].set_title('Success Rate vs Perturbation Probability')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-0.05, 1.05)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved statistics plot to {output_path}")

    if show:
        plt.show()

    return fig


def create_summary_report(
    results: BenchmarkResults,
    output_dir: str
) -> Dict[str, str]:
    """
    Create a complete summary report with all visualizations.

    Args:
        results: BenchmarkResults to analyze
        output_dir: Directory to save the report

    Returns:
        Dictionary mapping report names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_paths = {}

    # Error rate plot
    fig1 = plot_error_rate_vs_perturbation(
        results,
        output_path=str(output_path / "error_rate_vs_perturbation.png"),
        show=False
    )
    plt.close(fig1)
    report_paths["error_rate_plot"] = str(output_path / "error_rate_vs_perturbation.png")

    # Correlation heatmap
    fig2 = plot_correlation_heatmap(
        results,
        output_path=str(output_path / "correlation_heatmap.png"),
        show=False
    )
    plt.close(fig2)
    report_paths["correlation_heatmap"] = str(output_path / "correlation_heatmap.png")

    # Per-run statistics
    fig3 = plot_per_run_statistics(
        results,
        output_path=str(output_path / "per_run_statistics.png"),
        show=False
    )
    plt.close(fig3)
    report_paths["per_run_statistics"] = str(output_path / "per_run_statistics.png")

    # Text summary
    analysis = analyze_faithfulness(results)
    summary_path = output_path / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("VERIFORM BENCHMARK SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Faithfulness Score: {analysis['faithfulness_score']:.3f}\n\n")

        f.write("Verification Correlation:\n")
        for key, value in analysis['verification_correlation'].items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\nLinear Fit:\n")
        for key, value in analysis['linear_fit'].items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    report_paths["summary"] = str(summary_path)

    print(f"\nGenerated summary report in {output_dir}")
    return report_paths
