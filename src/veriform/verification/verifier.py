"""
Base verifier interfaces and data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List


class VerificationStrategy(Enum):
    """Types of verification strategies."""
    PYTHON_SYNTHESIS = "python_synthesis"
    AUTOFORMALIZATION = "autoformalization"
    LLM_JUDGE = "llm_judge"


@dataclass
class VerificationResult:
    """Result of verifying a statement."""

    statement: str
    is_correct: bool
    strategy_used: VerificationStrategy
    confidence: float  # 0.0 to 1.0
    explanation: str
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Verifier(ABC):
    """Abstract base class for verification strategies."""

    @abstractmethod
    def verify(
        self,
        statement: str,
        context: Optional[List[str]] = None,
        problem_statement: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify a statement.

        Args:
            statement: The statement to verify
            context: Previous statements/steps for context
            problem_statement: The original problem statement

        Returns:
            VerificationResult with verification outcome
        """
        pass

    @abstractmethod
    def can_verify(self, statement: str) -> bool:
        """
        Check if this verifier can handle the given statement.

        Args:
            statement: The statement to check

        Returns:
            True if this verifier can handle the statement
        """
        pass
