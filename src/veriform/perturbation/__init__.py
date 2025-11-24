"""
Perturbation module for introducing errors into reasoning steps.
"""

from .strategies import (
    PerturbationStrategy,
    OperatorSwapStrategy,
    ValueChangeStrategy,
    LogicalNegationStrategy,
    SignFlipStrategy,
    WordSwapStrategy,
    get_strategy,
    STRATEGY_REGISTRY
)
from .engine import PerturbationEngine

__all__ = [
    "PerturbationStrategy",
    "OperatorSwapStrategy",
    "ValueChangeStrategy",
    "LogicalNegationStrategy",
    "SignFlipStrategy",
    "WordSwapStrategy",
    "get_strategy",
    "STRATEGY_REGISTRY",
    "PerturbationEngine",
]
