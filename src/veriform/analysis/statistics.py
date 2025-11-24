"""
Statistical analysis of benchmark results.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple

from veriform.benchmark import BenchmarkResults, BenchmarkRun


def compute_correlation(
    probabilities: List[float],
    error_rates: List[float]
) -> Dict[str, float]:
    """
    Compute correlation between perturbation probability and error rate.

    Args:
        probabilities: List of perturbation probabilities
        error_rates: List of corresponding error rates

    Returns:
        Dictionary with correlation metrics
    """
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(probabilities, error_rates)

    # Spearman correlation (rank-based)
    spearman_r, spearman_p = stats.spearmanr(probabilities, error_rates)

    # Kendall's tau (rank-based)
    kendall_tau, kendall_p = stats.kendalltau(probabilities, error_rates)

    return {
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p_value": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p_value": kendall_p,
    }


def compute_error_rates(results: BenchmarkResults) -> Dict[str, Any]:
    """
    Compute various error rates from benchmark results.

    Args:
        results: BenchmarkResults to analyze

    Returns:
        Dictionary with error rate metrics
    """
    probabilities = []
    verification_error_rates = []
    formalization_error_rates = []

    for run in results.runs:
        probabilities.append(run.probability)

        # Verification error rate
        total_verifications = run.verification_failures + run.verification_successes
        if total_verifications > 0:
            ver_error_rate = run.verification_failures / total_verifications
        else:
            ver_error_rate = 0.0
        verification_error_rates.append(ver_error_rate)

        # Formalization error rate
        total_formalizations = run.successful_formalizations + run.failed_formalizations
        if total_formalizations > 0:
            form_error_rate = run.failed_formalizations / total_formalizations
        else:
            form_error_rate = 0.0
        formalization_error_rates.append(form_error_rate)

    return {
        "probabilities": probabilities,
        "verification_error_rates": verification_error_rates,
        "formalization_error_rates": formalization_error_rates,
    }


def analyze_faithfulness(results: BenchmarkResults) -> Dict[str, Any]:
    """
    Analyze the faithfulness of autoformalization.

    This is the key metric: how well does the error rate correlate with
    perturbation probability?

    Args:
        results: BenchmarkResults to analyze

    Returns:
        Dictionary with faithfulness analysis
    """
    error_rates = compute_error_rates(results)

    # Correlation for verification errors
    verification_correlation = compute_correlation(
        error_rates["probabilities"],
        error_rates["verification_error_rates"]
    )

    # Correlation for formalization errors
    formalization_correlation = compute_correlation(
        error_rates["probabilities"],
        error_rates["formalization_error_rates"]
    )

    # Linear regression for verification errors
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        error_rates["probabilities"],
        error_rates["verification_error_rates"]
    )

    linear_fit = {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err
    }

    # Faithfulness score (0-1, where 1 is perfect faithfulness)
    # Based on correlation strength
    faithfulness_score = abs(verification_correlation["pearson_r"])

    return {
        "faithfulness_score": faithfulness_score,
        "verification_correlation": verification_correlation,
        "formalization_correlation": formalization_correlation,
        "linear_fit": linear_fit,
        "error_rates": error_rates,
    }


def compute_summary_statistics(results: BenchmarkResults) -> Dict[str, Any]:
    """
    Compute summary statistics from benchmark results.

    Args:
        results: BenchmarkResults to analyze

    Returns:
        Dictionary with summary statistics
    """
    verification_error_rates = []
    formalization_error_rates = []
    perturbation_rates = []

    for run in results.runs:
        # Verification error rate
        total_ver = run.verification_failures + run.verification_successes
        if total_ver > 0:
            verification_error_rates.append(run.verification_failures / total_ver)

        # Formalization error rate
        total_form = run.successful_formalizations + run.failed_formalizations
        if total_form > 0:
            formalization_error_rates.append(run.failed_formalizations / total_form)

        # Actual perturbation rate
        if run.total_steps > 0:
            perturbation_rates.append(run.perturbed_steps / run.total_steps)

    return {
        "verification_error_rate": {
            "mean": np.mean(verification_error_rates),
            "std": np.std(verification_error_rates),
            "min": np.min(verification_error_rates),
            "max": np.max(verification_error_rates),
        },
        "formalization_error_rate": {
            "mean": np.mean(formalization_error_rates),
            "std": np.std(formalization_error_rates),
            "min": np.min(formalization_error_rates),
            "max": np.max(formalization_error_rates),
        },
        "perturbation_rate": {
            "mean": np.mean(perturbation_rates),
            "std": np.std(perturbation_rates),
            "min": np.min(perturbation_rates),
            "max": np.max(perturbation_rates),
        }
    }


def test_significance(
    probabilities: List[float],
    error_rates: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test statistical significance of the relationship.

    Args:
        probabilities: List of perturbation probabilities
        error_rates: List of corresponding error rates
        alpha: Significance level

    Returns:
        Dictionary with significance test results
    """
    # Linear regression test
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        probabilities,
        error_rates
    )

    is_significant = p_value < alpha

    # Effect size (Cohen's f-squared)
    r_squared = r_value ** 2
    if r_squared < 1.0:
        cohens_f2 = r_squared / (1 - r_squared)
    else:
        cohens_f2 = float('inf')

    # Interpretation
    if cohens_f2 < 0.02:
        effect_size = "negligible"
    elif cohens_f2 < 0.15:
        effect_size = "small"
    elif cohens_f2 < 0.35:
        effect_size = "medium"
    else:
        effect_size = "large"

    return {
        "is_significant": is_significant,
        "p_value": p_value,
        "alpha": alpha,
        "r_squared": r_squared,
        "cohens_f2": cohens_f2,
        "effect_size": effect_size,
        "interpretation": (
            f"The relationship is {'significant' if is_significant else 'not significant'} "
            f"at α={alpha} level with {effect_size} effect size."
        )
    }
