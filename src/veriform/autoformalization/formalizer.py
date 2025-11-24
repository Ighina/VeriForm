"""
Autoformalization interface and implementations.
"""

import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from veriform.data_collection import ReasoningStep, ReasoningChain
from .prompts import (
    AUTOFORMALIZATION_SYSTEM_PROMPT,
    create_autoformalization_prompt,
    LEAN_CODE_EXTRACTION_PATTERNS
)


@dataclass
class FormalizationResult:
    """Result of autoformalizing a reasoning step."""

    step_id: str
    lean_code: str
    raw_response: str
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Autoformalization(ABC):
    """Abstract base class for autoformalization."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.kwargs = kwargs

    @abstractmethod
    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Call the LLM with the given prompt."""
        pass

    def formalize_step(
        self,
        step: ReasoningStep,
        context_steps: List[ReasoningStep],
        problem_statement: str
    ) -> FormalizationResult:
        """
        Formalize a single reasoning step.

        Args:
            step: The reasoning step to formalize
            context_steps: Previous steps in the reasoning chain
            problem_statement: The original problem statement

        Returns:
            FormalizationResult containing the Lean code
        """
        prompt = create_autoformalization_prompt(step, context_steps, problem_statement)

        # Try with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt, AUTOFORMALIZATION_SYSTEM_PROMPT)
                lean_code = self._extract_lean_code(response)

                if lean_code:
                    return FormalizationResult(
                        step_id=step.step_id,
                        lean_code=lean_code,
                        raw_response=response,
                        success=True,
                        metadata={"attempts": attempt + 1}
                    )
                else:
                    last_error = "Could not extract Lean code from response"

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        return FormalizationResult(
            step_id=step.step_id,
            lean_code="",
            raw_response="",
            success=False,
            error_message=last_error,
            metadata={"attempts": self.max_retries}
        )

    def formalize_chain(self, chain: ReasoningChain) -> List[FormalizationResult]:
        """
        Formalize all steps in a reasoning chain.

        Args:
            chain: The reasoning chain to formalize

        Returns:
            List of FormalizationResults
        """
        results = []

        for i, step in enumerate(chain.steps):
            context_steps = chain.steps[:i]
            result = self.formalize_step(step, context_steps, chain.problem_statement)
            results.append(result)

        return results

    def _extract_lean_code(self, response: str) -> Optional[str]:
        """Extract Lean code from LLM response."""
        # Try different extraction patterns
        for pattern in LEAN_CODE_EXTRACTION_PATTERNS:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()

        # If no code blocks found, try to extract from the whole response
        # (in case the model returned raw code without markdown)
        if "theorem" in response or "lemma" in response or "def" in response:
            return response.strip()

        return None


class OpenAIFormalizer(Autoformalization):
    """Autoformalization using OpenAI models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.client = OpenAI(api_key=api_key)

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            **self.kwargs
        )
        return response.choices[0].message.content


class AnthropicFormalizer(Autoformalization):
    """Autoformalization using Anthropic Claude models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        self.client = Anthropic(api_key=api_key)

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Call Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=4096,
            **self.kwargs
        )
        return response.content[0].text


class MockFormalizer(Autoformalization):
    """Mock formalizer for testing purposes."""

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Return a mock Lean code response."""
        return """```lean
theorem mock_theorem : 1 + 1 = 2 := by
  rfl
```"""


def get_formalizer(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    **kwargs
) -> Autoformalization:
    """
    Factory function to get an autoformalization instance.

    Args:
        provider: Provider name ("openai", "anthropic", "mock")
        model: Model name
        api_key: Optional API key
        **kwargs: Additional arguments passed to the formalizer

    Returns:
        Autoformalization instance
    """
    formalizers = {
        "openai": OpenAIFormalizer,
        "anthropic": AnthropicFormalizer,
        "mock": MockFormalizer,
    }

    formalizer_class = formalizers.get(provider.lower())
    if formalizer_class is None:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {list(formalizers.keys())}"
        )

    return formalizer_class(model=model, api_key=api_key, **kwargs)
