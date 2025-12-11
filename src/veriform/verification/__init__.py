"""
Agentic verification module for VeriForm.

This module provides intelligent verification strategies that route
to different verification backends based on statement type.
"""

from .verifier import (
    VerificationResult,
    VerificationStrategy,
    Verifier
)
from .classifier import StatementClassifier, StatementType
from .python_verifier import PythonVerifier
from .llm_judge import LLMJudgeVerifier
from .agentic_verifier import AgenticVerifier

__all__ = [
    "VerificationResult",
    "VerificationStrategy",
    "Verifier",
    "StatementClassifier",
    "StatementType",
    "PythonVerifier",
    "LLMJudgeVerifier",
    "AgenticVerifier",
]
