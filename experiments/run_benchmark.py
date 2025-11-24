#!/usr/bin/env python3
"""
Example script to run a Veriform benchmark.

Usage:
    python run_benchmark.py [--config CONFIG_PATH]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from veriform import BenchmarkRunner, BenchmarkConfig
from veriform.utils import load_config
from veriform.analysis import analyze_faithfulness, create_summary_report


def main():
    parser = argparse.ArgumentParser(description="Run Veriform benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override output dir if specified
    if args.output_dir:
        config.output_dir = args.output_dir

    # Check for API keys
    if not config.openai_api_key and not os.getenv("OPENAI_API_KEY"):
        if "gpt" in config.autoformalization_model.lower():
            print("Warning: OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            print("Falling back to mock formalizer for testing.")
            config.autoformalization_model = "mock"

    if not config.anthropic_api_key and not os.getenv("ANTHROPIC_API_KEY"):
        if "claude" in config.autoformalization_model.lower():
            print("Warning: Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            print("Falling back to mock formalizer for testing.")
            config.autoformalization_model = "mock"

    # Create runner
    print("\nInitializing benchmark runner...")
    runner = BenchmarkRunner(config)

    # Run benchmark
    print("Starting benchmark execution...\n")
    results = runner.run()

    # Analyze results
    print("\n" + "="*60)
    print("ANALYZING RESULTS")
    print("="*60)

    analysis = analyze_faithfulness(results)

    print(f"\nFaithfulness Score: {analysis['faithfulness_score']:.3f}")
    print(f"Pearson r: {analysis['verification_correlation']['pearson_r']:.3f}")
    print(f"P-value: {analysis['verification_correlation']['pearson_p_value']:.4f}")

    print(f"\nLinear fit: y = {analysis['linear_fit']['slope']:.3f}x + {analysis['linear_fit']['intercept']:.3f}")
    print(f"R² = {analysis['linear_fit']['r_squared']:.3f}")

    # Create visualizations
    print("\nGenerating visualizations...")
    report_paths = create_summary_report(results, config.output_dir)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {config.output_dir}")
    print("\nGenerated files:")
    for name, path in report_paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
