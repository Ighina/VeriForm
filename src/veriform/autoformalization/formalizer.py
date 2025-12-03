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
    step_content: str
    is_perturbed: bool
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
        template: str = "safe",
        **kwargs
    ):
        # TODO: Allow multiple models for multistep formalization (e.g. AF + Prover)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.template = template
        self.kwargs = kwargs

    @abstractmethod
    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Call the LLM with the given prompt."""
        pass

    def __formalization_step(self,
                             prompt,
                             system_prompt):
        
        # Try with retries
        last_error = None
        lean_code = None
        response = "NOT PROVIDED"
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt, system_prompt)
                lean_code = self._extract_lean_code(response)

                if lean_code:
                    return lean_code, last_error, attempt, response
                else:
                    last_error = "Could not extract Lean code from response"

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return lean_code, last_error, attempt, response
        
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

        # if prompt include two elements, then assume we are using the typical AF + Proving framework
        if isinstance(prompt, tuple):
            assert len(prompt)==2, "If using a template with separate AF and Proving steps you need to pass two separate prompt templates!"
            
            af_prompt = prompt[0].format(step.content, [s.content for s in context_steps], problem_statement)

            formal_statement, last_error, attempt, response = self.__formalization_step(af_prompt, None)

            metadata = {"af_attempts": attempt, 
                        "af_response": response, 
                        "af_lean_code": formal_statement if formal_statement else ""}

            if formal_statement:
                proving_prompt = prompt[1].format(formal_statement)

                lean_code, last_error, attempt, response = self.__formalization_step(proving_prompt, None)

                metadata["proving_attempts"] = attempt

                if lean_code:
                    return FormalizationResult(
                        step_id=step.step_id,
                        step_content=step.content,
                        is_perturbed=step.is_perturbed,
                        lean_code=lean_code,
                        raw_response=response,
                        success=True,
                        metadata=metadata
                    )
                
            return FormalizationResult(
                        step_id=step.step_id,
                        step_content=step.content,
                        is_perturbed=step.is_perturbed,
                        lean_code="",
                        raw_response=response,
                        success=False,
                        metadata=metadata
                    )


        elif isinstance(prompt, str):
            # Try with retries
            lean_code, last_error, attempt, response = self.__formalization_step(prompt, AUTOFORMALIZATION_SYSTEM_PROMPT)

            if lean_code:
                return FormalizationResult(
                    step_id=step.step_id,
                    step_content=step.content,
                    is_perturbed=step.is_perturbed,
                    lean_code=lean_code,
                    raw_response=response,
                    success=True,
                    metadata={"attempts": attempt + 1}
                )
            else:
                # All retries failed
                return FormalizationResult(
                    step_id=step.step_id,
                    step_content=step.content,
                    is_perturbed=step.is_perturbed,
                    lean_code="",
                    raw_response="",
                    success=False,
                    error_message=last_error,
                    metadata={"attempts": self.max_retries}
                )
        else:
            raise NotImplementedError()

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

    def __init__(self, api_key: Optional[str] = None, 
                 vllm = False,
                 **kwargs):
        super().__init__(**kwargs)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        
        if vllm:
            # Modify OpenAI's API key and API base to use vLLM's API server.
            openai_api_key = "EMPTY"
            openai_api_base = "http://localhost:8000/v1"

            self.client = OpenAI(
                # defaults to os.environ.get("OPENAI_API_KEY")
                api_key=openai_api_key,
                base_url=openai_api_base,
            )

            # ensure the chosen model is among the possible models in local VLLM server
            assert self.model in [model.id for model in self.client.models.list()], "The chosen model is not available on your local VLLM server!!!"

        else:
            self.client = OpenAI(api_key=api_key)

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Call OpenAI API."""
            
        if system_prompt:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                **self.kwargs
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
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
        "vllm": OpenAIFormalizer,
        "mock": MockFormalizer,
    }

    formalizer_class = formalizers.get(provider.lower())
    if formalizer_class is None:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {list(formalizers.keys())}"
        )
    if provider=="vllm":
        return formalizer_class(model=model, api_key=api_key, vllm=True, **kwargs)

    return formalizer_class(model=model, api_key=api_key, **kwargs)
