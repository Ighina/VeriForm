"""
Perturbation engine for applying strategies to reasoning steps.
"""

import random
from typing import List, Optional, Dict, Any

from veriform.data_collection import ReasoningStep, ReasoningChain
from .strategies import PerturbationStrategy, get_strategy


class PerturbationEngine:
    """Engine for applying perturbation strategies to reasoning steps."""

    def __init__(
        self,
        strategies: List[str],
        probability: float = 0.5,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the perturbation engine.

        Args:
            strategies: List of strategy names to use
            probability: Probability of perturbing each step (0.0 to 1.0)
            strategy_params: Optional parameters for each strategy
            seed: Random seed for reproducibility
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")

        self.strategies = strategies
        self.probability = probability
        self.strategy_params = strategy_params or {}
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        # Initialize strategy instances
        self._strategy_instances: List[PerturbationStrategy] = []
        for strategy_name in strategies:
            params = self.strategy_params.get(strategy_name, {})
            params['seed'] = seed
            strategy = get_strategy(strategy_name, **params)
            self._strategy_instances.append(strategy)

    def perturb_step(self, step: ReasoningStep) -> ReasoningStep:
        """
        Perturb a single reasoning step with the configured probability.

        Args:
            step: The reasoning step to perturb

        Returns:
            Either the original step or a perturbed version
        """
        # Decide whether to perturb this step
        if random.random() > self.probability:
            return step

        # Try applicable strategies in random order
        applicable_strategies = [
            s for s in self._strategy_instances if s.can_apply(step)
        ]

        if not applicable_strategies:
            # No applicable strategies, return original
            return step

        # Randomly select one applicable strategy
        strategy = random.choice(applicable_strategies)

        # Apply the strategy
        perturbed = strategy.apply(step)

        # Return perturbed step if successful, otherwise original
        return perturbed if perturbed is not None else step

    def perturb_chain(self, chain: ReasoningChain) -> ReasoningChain:
        """
        Perturb all steps in a reasoning chain.

        Args:
            chain: The reasoning chain to perturb

        Returns:
            A new chain with perturbed steps
        """
        perturbed_steps = [self.perturb_step(step) for step in chain.steps]

        # Create new chain with perturbed steps
        perturbed_chain = ReasoningChain(
            chain_id=chain.chain_id,
            problem_statement=chain.problem_statement,
            steps=perturbed_steps,
            final_answer=chain.final_answer,
            source_dataset=chain.source_dataset,
            metadata=chain.metadata.copy()
        )

        # Add perturbation info to metadata
        num_perturbed = sum(1 for step in perturbed_steps if step.is_perturbed)
        perturbed_chain.metadata.update({
            "num_perturbed_steps": num_perturbed,
            "total_steps": len(perturbed_steps),
            "perturbation_rate": num_perturbed / len(perturbed_steps) if perturbed_steps else 0,
            "perturbation_probability": self.probability,
            "strategies_used": self.strategies
        })

        return perturbed_chain

    def perturb_chains(self, chains: List[ReasoningChain]) -> List[ReasoningChain]:
        """
        Perturb multiple reasoning chains.

        Args:
            chains: List of reasoning chains to perturb

        Returns:
            List of perturbed chains
        """
        return [self.perturb_chain(chain) for chain in chains]

    def get_perturbation_statistics(self, chains: List[ReasoningChain]) -> Dict[str, Any]:
        """
        Get statistics about perturbations applied to a list of chains.

        Args:
            chains: List of perturbed chains

        Returns:
            Dictionary containing perturbation statistics
        """
        total_steps = 0
        perturbed_steps = 0
        perturbation_types = {}

        for chain in chains:
            for step in chain.steps:
                total_steps += 1
                if step.is_perturbed:
                    perturbed_steps += 1
                    strategy = step.perturbation_applied or "unknown"
                    perturbation_types[strategy] = perturbation_types.get(strategy, 0) + 1

        return {
            "total_steps": total_steps,
            "perturbed_steps": perturbed_steps,
            "actual_perturbation_rate": perturbed_steps / total_steps if total_steps > 0 else 0,
            "expected_perturbation_rate": self.probability,
            "perturbation_by_type": perturbation_types,
            "num_chains": len(chains)
        }
