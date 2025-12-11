"""
Configuration classes for Veriform benchmarks.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    # Perturbation settings
    perturbation_probabilities: List[float] = Field(
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        description="List of perturbation probabilities to test"
    )
    perturbation_strategies: List[str] = Field(
        default=["operator_swap", "value_change", "logical_negation"],
        description="List of perturbation strategies to apply"
    )

    # Dataset settings
    dataset_name: str = Field(
        default="gsm8k",
        description="Name of the reasoning dataset to use"
    )
    sample_size: int = Field(
        default=1000,
        description="Number of reasoning steps to sample per probability"
    )

    # Autoformalization settings
    autoformalization_model: str = Field(
        default="gpt-4",
        description="Model to use for autoformalization"
    )
    autoformalization_temperature: float = Field(
        default=0.0,
        description="Temperature for autoformalization model"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries for API calls"
    )

    # Lean verification settings
    lean_timeout: int = Field(
        default=30,
        description="Timeout in seconds for Lean verification"
    )
    use_sorry_context: bool = Field(
        default=True,
        description="Whether to include previous steps as sorry lemmas"
    )

    # Output settings
    output_dir: str = Field(
        default="./experiments/outputs",
        description="Directory to save outputs"
    )
    save_intermediate: bool = Field(
        default=True,
        description="Whether to save intermediate results"
    )

    # Random seed
    random_seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    # API keys (loaded from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    class Config:
        extra = "allow"


class PerturbationConfig(BaseModel):
    """Configuration for perturbation strategies."""

    strategy_name: str
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    probability: float = Field(ge=0.0, le=1.0)


class DatasetConfig(BaseModel):
    """Configuration for dataset loading."""

    name: str
    split: str = "train"
    num_samples: Optional[int] = None
    filter_criteria: Optional[Dict[str, Any]] = None
    preprocessing_steps: List[str] = Field(default_factory=list)


class AgenticVerifierConfig(BaseModel):
    """Configuration for the agentic verifier."""

    # Model settings
    model: str = Field(
        default="gpt-4",
        description="Model to use for verification"
    )
    provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic)"
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for LLM calls"
    )

    # Strategy toggles
    enable_python_verifier: bool = Field(
        default=True,
        description="Enable Python synthesis verification for computational statements"
    )
    enable_autoformalization: bool = Field(
        default=True,
        description="Enable autoformalization + Lean verification for logical/mathematical statements"
    )
    enable_llm_judge: bool = Field(
        default=True,
        description="Enable LLM judge verification for general statements"
    )
    fallback_to_llm_judge: bool = Field(
        default=True,
        description="Whether to fallback to LLM judge if other methods fail"
    )

    # Python verifier settings
    python_execution_timeout: int = Field(
        default=5,
        description="Timeout in seconds for Python code execution"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries for verification attempts"
    )

    # Output settings
    verbose: bool = Field(
        default=False,
        description="Whether to print routing decisions and debug info"
    )
