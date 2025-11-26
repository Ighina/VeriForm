"""
Benchmark runner for measuring autoformalization faithfulness.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from veriform.config import BenchmarkConfig
from veriform.data_collection import ReasoningChain, get_loader
from veriform.perturbation import PerturbationEngine
from veriform.autoformalization import (
    get_formalizer,
    FormalizationResult,
    LeanVerifier,
    MockLeanVerifier
)


@dataclass
class BenchmarkRun:
    """Results from a single benchmark run at a specific perturbation probability."""

    probability: float
    num_chains: int
    total_steps: int
    perturbed_steps: int
    successful_formalizations: int
    failed_formalizations: int
    verification_failures: int
    verification_successes: int
    chains_data: List[Dict[str, Any]]


@dataclass
class BenchmarkResults:
    """Complete results from a benchmark run."""

    config: Dict[str, Any]
    runs: List[BenchmarkRun]
    overall_statistics: Dict[str, Any]


class BenchmarkRunner:
    """Runner for executing autoformalization faithfulness benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components (lazy loading)
        self._loader = None
        self._formalizer = None
        self._verifier = None

    def _get_loader(self):
        """Get or create dataset loader."""
        if self._loader is None:
            self._loader = get_loader(
                self.config.dataset_name,
                num_samples=self.config.sample_size,
                seed=self.config.random_seed
            )
        return self._loader

    def _get_formalizer(self):
        """Get or create autoformalization instance."""
        if self._formalizer is None:
            # Determine provider from model name
            if "gpt" in self.config.autoformalization_model.lower():
                provider = "openai"
                api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            elif "claude" in self.config.autoformalization_model.lower():
                provider = "anthropic"
                api_key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            elif "mock" in self.config.autoformalization_model.lower():
                provider = "mock"
                api_key = None
            else:
                provider = "vllm"
                api_key = None

            self._formalizer = get_formalizer(
                provider=provider,
                model=self.config.autoformalization_model,
                api_key=api_key,
                temperature=self.config.autoformalization_temperature,
                max_retries=self.config.max_retries
            )
        return self._formalizer

    def _get_verifier(self):
        """Get or create Lean verifier."""
        if self._verifier is None:
            try:
                self._verifier = LeanVerifier(timeout=self.config.lean_timeout)
            except:
                # Fall back to mock verifier if Lean not available
                self._verifier = MockLeanVerifier(timeout=self.config.lean_timeout)
        return self._verifier

    def run_single_probability(
        self,
        probability: float,
        chains: List[ReasoningChain]
    ) -> BenchmarkRun:
        """
        Run benchmark for a single perturbation probability.

        Args:
            probability: Perturbation probability
            chains: List of reasoning chains to process

        Returns:
            BenchmarkRun with results
        """
        # Create perturbation engine
        engine = PerturbationEngine(
            strategies=self.config.perturbation_strategies,
            probability=probability,
            seed=self.config.random_seed
        )

        # Perturb chains
        perturbed_chains = engine.perturb_chains(chains)

        # Get perturbation statistics
        stats = engine.get_perturbation_statistics(perturbed_chains)

        # Formalize and verify each chain
        formalizer = self._get_formalizer()
        verifier = self._get_verifier()

        chains_data = []
        total_successful_formalizations = 0
        total_failed_formalizations = 0
        total_verification_failures = 0
        total_verification_successes = 0

        for chain in tqdm(perturbed_chains, desc=f"Processing (p={probability})"):
            # Formalize all steps
            formalization_results = formalizer.formalize_chain(chain)

            # Verify each formalization
            verification_results = []
            for result in formalization_results:
                if result.success and result.lean_code:
                    verification = verifier.verify(result.lean_code)
                    verification_results.append(verification)

                    if not verification.success:
                        total_verification_failures += 1
                    else:
                        total_verification_successes += 1

                    if result.success:
                        total_successful_formalizations += 1
                    else:
                        total_failed_formalizations += 1
                else:
                    total_failed_formalizations += 1

            # Store chain data
            chain_data = {
                "chain_id": chain.chain_id,
                "num_steps": len(chain.steps),
                "num_perturbed": sum(1 for s in chain.steps if s.is_perturbed),
                "formalization_results": [
                    {
                        "step_id": r.step_id,
                        "success": r.success,
                        "has_lean_code": bool(r.lean_code),
                        "error": r.error_message
                    }
                    for r in formalization_results
                ],
                "verification_results": [
                    {
                        "success": v.success,
                        "is_provable": v.is_provable,
                        "has_sorry": v.has_sorry,
                        "error": v.error_message
                    }
                    for v in verification_results
                ]
            }
            chains_data.append(chain_data)

        # Create benchmark run
        return BenchmarkRun(
            probability=probability,
            num_chains=len(perturbed_chains),
            total_steps=stats["total_steps"],
            perturbed_steps=stats["perturbed_steps"],
            successful_formalizations=total_successful_formalizations,
            failed_formalizations=total_failed_formalizations,
            verification_failures=total_verification_failures,
            verification_successes=total_verification_successes,
            chains_data=chains_data
        )

    def run(self) -> BenchmarkResults:
        """
        Run the complete benchmark.

        Returns:
            BenchmarkResults with all results
        """
        print(f"Loading dataset: {self.config.dataset_name}")
        loader = self._get_loader()
        chains = loader.load()
        print(f"Loaded {len(chains)} reasoning chains")

        # Run for each probability
        runs = []
        for probability in self.config.perturbation_probabilities:
            print(f"\nRunning benchmark for probability: {probability}")
            run = self.run_single_probability(probability, chains)
            runs.append(run)

            # Save intermediate results if configured
            if self.config.save_intermediate:
                self._save_run(run, probability)

        # Compute overall statistics
        overall_stats = self._compute_overall_statistics(runs)

        # Create results
        results = BenchmarkResults(
            config=asdict(self.config),
            runs=runs,
            overall_statistics=overall_stats
        )

        # Save final results
        self._save_results(results)

        return results

    def _compute_overall_statistics(self, runs: List[BenchmarkRun]) -> Dict[str, Any]:
        """Compute overall statistics across all runs."""
        return {
            "num_probabilities": len(runs),
            "total_chains_processed": sum(r.num_chains for r in runs),
            "total_steps_processed": sum(r.total_steps for r in runs),
            "total_perturbed_steps": sum(r.perturbed_steps for r in runs),
            "total_successful_formalizations": sum(r.successful_formalizations for r in runs),
            "total_failed_formalizations": sum(r.failed_formalizations for r in runs),
            "total_verification_failures": sum(r.verification_failures for r in runs),
            "total_verification_successes": sum(r.verification_successes for r in runs),
        }

    def _save_run(self, run: BenchmarkRun, probability: float):
        """Save results for a single run."""
        output_path = Path(self.config.output_dir) / f"run_p{probability:.2f}.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(run), f, indent=2)
        print(f"Saved run results to {output_path}")

    def _save_results(self, results: BenchmarkResults):
        """Save complete benchmark results."""
        output_path = Path(self.config.output_dir) / "results.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        print(f"\nSaved complete results to {output_path}")
