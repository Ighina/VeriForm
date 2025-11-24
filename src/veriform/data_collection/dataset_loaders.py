"""
Dataset loaders for various reasoning datasets.
"""

import re
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import random

from datasets import load_dataset

from .reasoning_step import ReasoningStep, ReasoningChain, StepType


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, split: str = "train", num_samples: Optional[int] = None, seed: int = 42):
        self.split = split
        self.num_samples = num_samples
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def load(self) -> List[ReasoningChain]:
        """Load and parse the dataset into reasoning chains."""
        pass

    @abstractmethod
    def parse_reasoning_steps(self, example: Dict[str, Any]) -> List[ReasoningStep]:
        """Parse an example into reasoning steps."""
        pass


class GSM8KLoader(DatasetLoader):
    """Loader for GSM8K dataset."""

    def load(self) -> List[ReasoningChain]:
        """Load GSM8K dataset."""
        dataset = load_dataset("gsm8k", "main", split=self.split)

        if self.num_samples:
            indices = random.sample(range(len(dataset)), min(self.num_samples, len(dataset)))
            dataset = dataset.select(indices)

        chains = []
        for idx, example in enumerate(dataset):
            chain = ReasoningChain(
                chain_id=f"gsm8k_{self.split}_{idx}",
                problem_statement=example["question"],
                steps=self.parse_reasoning_steps(example),
                final_answer=self._extract_final_answer(example["answer"]),
                source_dataset="gsm8k",
                metadata={"original_index": idx}
            )
            chains.append(chain)

        return chains

    def parse_reasoning_steps(self, example: Dict[str, Any]) -> List[ReasoningStep]:
        """Parse GSM8K solution into reasoning steps."""
        solution = example["answer"]

        # Split by newlines and filter empty lines
        lines = [line.strip() for line in solution.split("\n") if line.strip()]

        steps = []
        for step_idx, line in enumerate(lines):
            # Skip the final answer line
            if line.startswith("####"):
                continue

            # Classify step type based on content
            step_type = self._classify_step(line)

            step = ReasoningStep(
                step_id=f"step_{step_idx}",
                content=line,
                step_type=step_type,
                previous_steps=[f"step_{i}" for i in range(step_idx)],
                metadata={"line_number": step_idx}
            )
            steps.append(step)

        return steps

    def _classify_step(self, content: str) -> StepType:
        """Classify the type of reasoning step."""
        if any(op in content for op in ["+", "-", "*", "/", "="]):
            return StepType.CALCULATION
        elif any(word in content.lower() for word in ["therefore", "thus", "so", "hence"]):
            return StepType.LOGICAL_DEDUCTION
        else:
            return StepType.OTHER

    def _extract_final_answer(self, answer: str) -> str:
        """Extract the final numerical answer."""
        match = re.search(r"####\s*(.+)", answer)
        if match:
            return match.group(1).strip()
        return ""


class MATHLoader(DatasetLoader):
    """Loader for MATH dataset."""

    def load(self) -> List[ReasoningChain]:
        """Load MATH dataset."""
        dataset = load_dataset("hendrycks/math", split=self.split)

        if self.num_samples:
            indices = random.sample(range(len(dataset)), min(self.num_samples, len(dataset)))
            dataset = dataset.select(indices)

        chains = []
        for idx, example in enumerate(dataset):
            chain = ReasoningChain(
                chain_id=f"math_{self.split}_{idx}",
                problem_statement=example["problem"],
                steps=self.parse_reasoning_steps(example),
                final_answer=example["solution"],
                source_dataset="math",
                metadata={
                    "original_index": idx,
                    "level": example.get("level", "unknown"),
                    "type": example.get("type", "unknown")
                }
            )
            chains.append(chain)

        return chains

    def parse_reasoning_steps(self, example: Dict[str, Any]) -> List[ReasoningStep]:
        """Parse MATH solution into reasoning steps."""
        solution = example["solution"]

        # Split by sentences or double newlines
        lines = re.split(r'\n\n+|\. (?=[A-Z])', solution)
        lines = [line.strip() for line in lines if line.strip()]

        steps = []
        for step_idx, line in enumerate(lines):
            step_type = self._classify_step(line)

            step = ReasoningStep(
                step_id=f"step_{step_idx}",
                content=line,
                step_type=step_type,
                previous_steps=[f"step_{i}" for i in range(step_idx)],
                metadata={"line_number": step_idx}
            )
            steps.append(step)

        return steps

    def _classify_step(self, content: str) -> StepType:
        """Classify the type of reasoning step."""
        if "$" in content and any(op in content for op in ["=", "+", "-", "*", "/"]):
            return StepType.ALGEBRAIC_MANIPULATION
        elif any(word in content.lower() for word in ["substitute", "plug in", "replace"]):
            return StepType.SUBSTITUTION
        elif any(word in content.lower() for word in ["simplify", "reduce", "combine"]):
            return StepType.SIMPLIFICATION
        else:
            return StepType.OTHER


class CustomLoader(DatasetLoader):
    """Loader for custom datasets with flexible format."""

    def __init__(self, data: List[Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def load(self) -> List[ReasoningChain]:
        """Load custom data."""
        if self.num_samples:
            data = random.sample(self.data, min(self.num_samples, len(self.data)))
        else:
            data = self.data

        chains = []
        for idx, example in enumerate(data):
            chain = ReasoningChain(
                chain_id=f"custom_{idx}",
                problem_statement=example.get("problem", ""),
                steps=self.parse_reasoning_steps(example),
                final_answer=example.get("answer", ""),
                source_dataset="custom",
                metadata=example.get("metadata", {})
            )
            chains.append(chain)

        return chains

    def parse_reasoning_steps(self, example: Dict[str, Any]) -> List[ReasoningStep]:
        """Parse custom format into reasoning steps."""
        steps_data = example.get("steps", [])

        steps = []
        for step_idx, step_content in enumerate(steps_data):
            if isinstance(step_content, str):
                content = step_content
                step_type = StepType.OTHER
            elif isinstance(step_content, dict):
                content = step_content.get("content", "")
                step_type = StepType(step_content.get("type", "other"))
            else:
                continue

            step = ReasoningStep(
                step_id=f"step_{step_idx}",
                content=content,
                step_type=step_type,
                previous_steps=[f"step_{i}" for i in range(step_idx)],
            )
            steps.append(step)

        return steps


def get_loader(dataset_name: str, **kwargs) -> DatasetLoader:
    """Factory function to get the appropriate dataset loader."""
    loaders = {
        "gsm8k": GSM8KLoader,
        "math": MATHLoader,
        "custom": CustomLoader,
    }

    loader_class = loaders.get(dataset_name.lower())
    if loader_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    return loader_class(**kwargs)
