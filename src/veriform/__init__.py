"""
Veriform: Verification-Oriented Formalization Benchmarking

A framework for measuring the faithfulness of autoformalization systems.
"""

__version__ = "0.1.0"

from veriform.benchmark import BenchmarkRunner
from veriform.config import BenchmarkConfig

__all__ = ["BenchmarkRunner", "BenchmarkConfig", "__version__"]
