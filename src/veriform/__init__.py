"""
Veriform: Verification-Oriented Formalization Benchmarking

A framework for measuring the faithfulness of autoformalization systems.
"""

__version__ = "0.1.0"

from veriform.benchmark import BenchmarkRunner
from veriform.config import BenchmarkConfig, AgenticVerifierConfig
from veriform.verification import AgenticVerifier

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "AgenticVerifierConfig",
    "AgenticVerifier",
    "__version__"
]
