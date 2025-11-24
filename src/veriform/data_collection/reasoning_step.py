"""
Core data structures for reasoning steps.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class StepType(Enum):
    """Type of reasoning step."""
    CALCULATION = "calculation"
    LOGICAL_DEDUCTION = "logical_deduction"
    ALGEBRAIC_MANIPULATION = "algebraic_manipulation"
    SUBSTITUTION = "substitution"
    SIMPLIFICATION = "simplification"
    OTHER = "other"


@dataclass
class ReasoningStep:
    """A single step in a chain of reasoning."""

    step_id: str
    content: str
    step_type: StepType = StepType.OTHER
    previous_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For perturbed steps
    is_perturbed: bool = False
    perturbation_applied: Optional[str] = None
    original_content: Optional[str] = None

    def __str__(self) -> str:
        return f"Step {self.step_id}: {self.content}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "content": self.content,
            "step_type": self.step_type.value,
            "previous_steps": self.previous_steps,
            "metadata": self.metadata,
            "is_perturbed": self.is_perturbed,
            "perturbation_applied": self.perturbation_applied,
            "original_content": self.original_content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        """Create from dictionary."""
        data["step_type"] = StepType(data.get("step_type", "other"))
        return cls(**data)


@dataclass
class ReasoningChain:
    """A complete chain of reasoning steps."""

    chain_id: str
    problem_statement: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    source_dataset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the chain."""
        self.steps.append(step)

    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        """Get a specific step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_context_for_step(self, step_id: str) -> List[ReasoningStep]:
        """Get all previous steps for a given step."""
        context = []
        for step in self.steps:
            if step.step_id == step_id:
                break
            context.append(step)
        return context

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chain_id": self.chain_id,
            "problem_statement": self.problem_statement,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "source_dataset": self.source_dataset,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningChain":
        """Create from dictionary."""
        data["steps"] = [ReasoningStep.from_dict(s) for s in data.get("steps", [])]
        return cls(**data)
