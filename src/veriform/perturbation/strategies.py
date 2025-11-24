"""
Perturbation strategies for reasoning steps.
"""

import re
import random
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from veriform.data_collection import ReasoningStep


class PerturbationStrategy(ABC):
    """Abstract base class for perturbation strategies."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    @abstractmethod
    def apply(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        """
        Apply perturbation to a reasoning step.

        Returns:
            A new ReasoningStep with the perturbation applied, or None if
            perturbation cannot be applied.
        """
        pass

    @abstractmethod
    def can_apply(self, step: ReasoningStep) -> bool:
        """Check if this perturbation can be applied to the given step."""
        pass

    def _create_perturbed_step(
        self,
        original_step: ReasoningStep,
        new_content: str,
        strategy_name: str
    ) -> ReasoningStep:
        """Helper to create a perturbed step."""
        return ReasoningStep(
            step_id=original_step.step_id,
            content=new_content,
            step_type=original_step.step_type,
            previous_steps=original_step.previous_steps.copy(),
            metadata=original_step.metadata.copy(),
            is_perturbed=True,
            perturbation_applied=strategy_name,
            original_content=original_step.content
        )


class OperatorSwapStrategy(PerturbationStrategy):
    """Swap arithmetic operators (+, -, *, /)."""

    OPERATOR_SWAPS = {
        '+': ['-', '*'],
        '-': ['+', '/'],
        '*': ['/', '+'],
        '/': ['*', '-']
    }

    def can_apply(self, step: ReasoningStep) -> bool:
        """Check if step contains arithmetic operators."""
        return any(op in step.content for op in self.OPERATOR_SWAPS.keys())

    def apply(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        """Swap one arithmetic operator."""
        if not self.can_apply(step):
            return None

        content = step.content

        # Find all operator positions
        operator_positions = []
        for i, char in enumerate(content):
            if char in self.OPERATOR_SWAPS:
                operator_positions.append((i, char))

        if not operator_positions:
            return None

        # Randomly select one operator to swap
        pos, original_op = random.choice(operator_positions)
        new_op = random.choice(self.OPERATOR_SWAPS[original_op])

        # Replace the operator
        new_content = content[:pos] + new_op + content[pos + 1:]

        return self._create_perturbed_step(step, new_content, "operator_swap")


class ValueChangeStrategy(PerturbationStrategy):
    """Change numerical values in the step."""

    def __init__(self, max_delta: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta

    def can_apply(self, step: ReasoningStep) -> bool:
        """Check if step contains numbers."""
        return bool(re.search(r'\d+\.?\d*', step.content))

    def apply(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        """Change one numerical value."""
        if not self.can_apply(step):
            return None

        content = step.content

        # Find all numbers
        numbers = list(re.finditer(r'\d+\.?\d*', content))
        if not numbers:
            return None

        # Randomly select one number to change
        match = random.choice(numbers)
        original_value = float(match.group())

        # Apply a random change
        delta = random.uniform(-self.max_delta, self.max_delta)
        new_value = original_value + delta

        # Keep as integer if original was integer
        if '.' not in match.group():
            new_value = int(round(new_value))
            new_value_str = str(new_value)
        else:
            new_value_str = f"{new_value:.2f}"

        # Replace the value
        new_content = content[:match.start()] + new_value_str + content[match.end():]

        return self._create_perturbed_step(step, new_content, "value_change")


class LogicalNegationStrategy(PerturbationStrategy):
    """Negate logical statements."""

    NEGATION_PAIRS = [
        ("is", "is not"),
        ("are", "are not"),
        ("equals", "does not equal"),
        ("greater than", "less than"),
        ("less than", "greater than"),
        (">", "<"),
        ("<", ">"),
        (">=", "<="),
        ("<=", ">="),
    ]

    def can_apply(self, step: ReasoningStep) -> bool:
        """Check if step contains logical statements."""
        content_lower = step.content.lower()
        return any(pair[0] in content_lower for pair in self.NEGATION_PAIRS)

    def apply(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        """Negate a logical statement."""
        if not self.can_apply(step):
            return None

        content = step.content
        content_lower = content.lower()

        # Find applicable negations
        applicable = []
        for original, negated in self.NEGATION_PAIRS:
            if original in content_lower:
                applicable.append((original, negated))

        if not applicable:
            return None

        # Apply one random negation
        original, negated = random.choice(applicable)

        # Case-insensitive replace
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        new_content = pattern.sub(negated, content, count=1)

        return self._create_perturbed_step(step, new_content, "logical_negation")


class SignFlipStrategy(PerturbationStrategy):
    """Flip the sign of numbers."""

    def can_apply(self, step: ReasoningStep) -> bool:
        """Check if step contains numbers."""
        return bool(re.search(r'-?\d+\.?\d*', step.content))

    def apply(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        """Flip the sign of one number."""
        if not self.can_apply(step):
            return None

        content = step.content

        # Find all numbers (including negative)
        numbers = list(re.finditer(r'-?\d+\.?\d*', content))
        if not numbers:
            return None

        # Select a number to flip
        match = random.choice(numbers)
        original = match.group()

        if original.startswith('-'):
            new_value = original[1:]
        else:
            new_value = '-' + original

        new_content = content[:match.start()] + new_value + content[match.end():]

        return self._create_perturbed_step(step, new_content, "sign_flip")


class WordSwapStrategy(PerturbationStrategy):
    """Swap similar words to change meaning."""

    SWAP_PAIRS = [
        ("add", "subtract"),
        ("multiply", "divide"),
        ("increase", "decrease"),
        ("more", "less"),
        ("total", "difference"),
        ("sum", "product"),
        ("each", "all"),
        ("every", "some"),
    ]

    def can_apply(self, step: ReasoningStep) -> bool:
        """Check if step contains swappable words."""
        content_lower = step.content.lower()
        return any(word in content_lower for pair in self.SWAP_PAIRS for word in pair)

    def apply(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        """Swap one word."""
        if not self.can_apply(step):
            return None

        content = step.content
        content_lower = content.lower()

        # Find applicable swaps
        applicable = []
        for word1, word2 in self.SWAP_PAIRS:
            if word1 in content_lower:
                applicable.append((word1, word2))
            elif word2 in content_lower:
                applicable.append((word2, word1))

        if not applicable:
            return None

        # Apply one random swap
        original, replacement = random.choice(applicable)
        pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)

        def replace_match(match):
            text = match.group()
            if text[0].isupper():
                return replacement.capitalize()
            return replacement

        new_content = pattern.sub(replace_match, content, count=1)

        return self._create_perturbed_step(step, new_content, "word_swap")


# Registry of all strategies
STRATEGY_REGISTRY: Dict[str, type] = {
    "operator_swap": OperatorSwapStrategy,
    "value_change": ValueChangeStrategy,
    "logical_negation": LogicalNegationStrategy,
    "sign_flip": SignFlipStrategy,
    "word_swap": WordSwapStrategy,
}


def get_strategy(strategy_name: str, **kwargs) -> PerturbationStrategy:
    """Factory function to get a perturbation strategy."""
    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
    if strategy_class is None:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return strategy_class(**kwargs)
