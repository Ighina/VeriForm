"""
Quickstart example for Veriform.

This demonstrates the basic workflow with a small sample.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from veriform import BenchmarkRunner, BenchmarkConfig
from veriform.analysis import analyze_faithfulness, plot_error_rate_vs_perturbation


def main():
    print("="*60)
    print("VERIFORM QUICKSTART")
    print("="*60)

    # Create a simple configuration
    config = BenchmarkConfig(
        # Test with fewer probabilities for speed
        perturbation_probabilities=[0.0, 0.25, 0.5, 0.75, 1.0],

        # Use fewer strategies
        perturbation_strategies=["operator_swap", "value_change"],

        # Small sample size for quick testing
        dataset_name="gsm8k",
        sample_size=10,

        # Use mock formalizer (no API keys needed)
        autoformalization_model="mock",

        # Output settings
        output_dir="./outputs/quickstart",
        save_intermediate=True,

        # Random seed
        random_seed=42
    )

    print("\nConfiguration:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Sample size: {config.sample_size}")
    print(f"  Perturbation probabilities: {config.perturbation_probabilities}")
    print(f"  Strategies: {config.perturbation_strategies}")

    # Run benchmark
    print("\nRunning benchmark...")
    runner = BenchmarkRunner(config)
    results = runner.run()

    # Analyze
    print("\nAnalyzing results...")
    analysis = analyze_faithfulness(results)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nFaithfulness Score: {analysis['faithfulness_score']:.3f}")
    print(f"  (Correlation between perturbation rate and error rate)")

    print(f"\nVerification Correlation:")
    print(f"  Pearson r: {analysis['verification_correlation']['pearson_r']:.3f}")
    print(f"  P-value: {analysis['verification_correlation']['pearson_p_value']:.4f}")

    print(f"\nLinear Fit:")
    slope = analysis['linear_fit']['slope']
    intercept = analysis['linear_fit']['intercept']
    r_squared = analysis['linear_fit']['r_squared']
    print(f"  y = {slope:.3f}x + {intercept:.3f}")
    print(f"  R² = {r_squared:.3f}")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if analysis['faithfulness_score'] > 0.8:
        interpretation = "EXCELLENT"
        meaning = "The autoformalization system is highly faithful."
    elif analysis['faithfulness_score'] > 0.6:
        interpretation = "GOOD"
        meaning = "The autoformalization system shows good faithfulness."
    elif analysis['faithfulness_score'] > 0.4:
        interpretation = "MODERATE"
        meaning = "The autoformalization system shows moderate faithfulness."
    else:
        interpretation = "POOR"
        meaning = "The autoformalization system shows poor faithfulness."

    print(f"\nFaithfulness: {interpretation}")
    print(f"{meaning}")
    print("\nA faithful system should produce more verification errors")
    print("as the perturbation rate increases.")

    # Create visualizations
    print("\nGenerating plot...")
    plot_error_rate_vs_perturbation(
        results,
        output_path=f"{config.output_dir}/plot.png",
        show=False
    )

    print(f"\nResults saved to: {config.output_dir}")
    print("\nQuickstart complete!")


if __name__ == "__main__":
    main()
