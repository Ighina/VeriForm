"""
Statement classifier for routing to appropriate verification strategies.
"""

import re
from enum import Enum
from typing import Optional


class StatementType(Enum):
    """Types of statements for verification routing."""
    COMPUTATIONAL = "computational"  # Arithmetic, numerical calculations
    LOGICAL_MATHEMATICAL = "logical_mathematical"  # Abstract math, theorems
    GENERAL = "general"  # Other statements


class StatementClassifier:
    """
    Classifies statements to determine appropriate verification strategy.

    - COMPUTATIONAL: Statements involving arithmetic, numerical calculations,
                     concrete computations that can be executed
    - LOGICAL_MATHEMATICAL: Abstract mathematical reasoning, theorems,
                           formal logic that requires autoformalization
    - GENERAL: Everything else that needs LLM judgment
    """

    # Patterns for computational statements
    COMPUTATIONAL_PATTERNS = [
        r'\d+\s*[\+\-\*\/\%]\s*\d+',  # Basic arithmetic: 5 + 3, 10 * 2
        r'=\s*\d+',  # Equals number: = 15
        r'\d+\s*[<>=]+\s*\d+',  # Comparisons: 5 < 10, 3 >= 2
        r'\b(sum|product|quotient|remainder|mod|modulo)\b',  # Arithmetic operations
        r'\b(calculate|compute|evaluate)\b',  # Computational verbs
        r'\b\d+(\.\d+)?\s*(times|divided by|plus|minus)\s*\d+(\.\d+)?\b',  # Word arithmetic
        r'\$\d+',  # Money calculations
        r'\d+\s*%',  # Percentages
        r'\d+\s*(squared|cubed)',  # Powers
        r'sqrt\(\d+\)',  # Square root
    ]

    # Patterns for logical/mathematical statements
    LOGICAL_MATHEMATICAL_PATTERNS = [
        r'\b(theorem|lemma|proposition|corollary|axiom|conjecture)\b',
        r'\b(proof|prove|let|assume|suppose|given that)\b',
        r'\b(for all|there exists|if and only if|iff|implies)\b',
        r'\b(set|subset|union|intersection|element of)\b',
        r'\b(function|mapping|bijection|injection|surjection)\b',
        r'\b(continuous|differentiable|integrable|convergent)\b',
        r'\b(vector space|linear|matrix|eigenvalue|determinant)\b',
        r'\b(prime|composite|divisible|gcd|lcm)\b',
        r'\b(induction|recursive|recurrence)\b',
        r'\\forall|\\exists|\\in|\\subset|\\cap|\\cup',  # LaTeX symbols
        r'\b(logic|logical|entails|satisfies)\b',
    ]

    # Keywords suggesting general reasoning
    GENERAL_KEYWORDS = [
        'because', 'therefore', 'thus', 'hence', 'consequently',
        'follows that', 'we can conclude', 'this means',
        'according to', 'based on', 'given'
    ]

    @classmethod
    def classify(cls, statement: str) -> StatementType:
        """
        Classify a statement to determine verification strategy.

        Args:
            statement: The statement to classify

        Returns:
            StatementType indicating the classification
        """
        statement_lower = statement.lower()

        # Check for computational patterns
        computational_score = cls._count_pattern_matches(
            statement_lower, cls.COMPUTATIONAL_PATTERNS
        )

        # Check for logical/mathematical patterns
        logical_math_score = cls._count_pattern_matches(
            statement_lower, cls.LOGICAL_MATHEMATICAL_PATTERNS
        )

        # Decision logic
        if computational_score > 0 and computational_score >= logical_math_score:
            # Has computational elements and they dominate
            return StatementType.COMPUTATIONAL

        elif logical_math_score > 0:
            # Has logical/mathematical elements
            return StatementType.LOGICAL_MATHEMATICAL

        else:
            # Default to general reasoning
            return StatementType.GENERAL

    @staticmethod
    def _count_pattern_matches(text: str, patterns: list) -> int:
        """Count how many patterns match the text."""
        count = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count

    @classmethod
    def get_classification_confidence(cls, statement: str) -> float:
        """
        Get confidence score for the classification.

        Args:
            statement: The statement to analyze

        Returns:
            Confidence score between 0.0 and 1.0
        """
        statement_lower = statement.lower()

        computational_score = cls._count_pattern_matches(
            statement_lower, cls.COMPUTATIONAL_PATTERNS
        )
        logical_math_score = cls._count_pattern_matches(
            statement_lower, cls.LOGICAL_MATHEMATICAL_PATTERNS
        )

        total_patterns = len(cls.COMPUTATIONAL_PATTERNS) + len(cls.LOGICAL_MATHEMATICAL_PATTERNS)
        max_possible_score = max(computational_score + logical_math_score, 1)

        # Normalize to 0-1 range
        confidence = min(max_possible_score / 5.0, 1.0)  # 5+ matches = high confidence

        return confidence
