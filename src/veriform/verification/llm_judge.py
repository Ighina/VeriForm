"""
LLM-based judgment verifier for general reasoning statements.
"""

import re
import time
from typing import Optional, List

from .verifier import Verifier, VerificationResult, VerificationStrategy


class LLMJudgeVerifier(Verifier):
    """
    Verifies general statements using an LLM as a judge.

    The LLM evaluates whether a reasoning step is logically sound
    given the context and problem statement.
    """

    def __init__(
        self,
        llm_client,  # OpenAI or Anthropic client
        model: str,
        provider: str = "openai",
        temperature: float = 0.2,
        max_retries: int = 3
    ):
        """
        Initialize LLM judge verifier.

        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model name to use
            provider: "openai" or "anthropic"
            temperature: Sampling temperature (slightly higher for reasoning)
            max_retries: Maximum retry attempts
        """
        self.llm_client = llm_client
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_retries = max_retries

    def can_verify(self, statement: str) -> bool:
        """LLM judge can handle any statement."""
        return True

    def verify(
        self,
        statement: str,
        context: Optional[List[str]] = None,
        problem_statement: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify a statement using LLM judgment.

        Args:
            statement: The statement to verify
            context: Previous statements for context
            problem_statement: Original problem statement

        Returns:
            VerificationResult with judgment outcome
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Create judgment prompt
                prompt = self._create_judgment_prompt(
                    statement, context, problem_statement
                )

                # Get LLM judgment
                response = self._call_llm(prompt)

                # Parse the response
                is_correct, confidence, explanation = self._parse_judgment(response)

                return VerificationResult(
                    statement=statement,
                    is_correct=is_correct,
                    strategy_used=VerificationStrategy.LLM_JUDGE,
                    confidence=confidence,
                    explanation=explanation,
                    metadata={
                        "raw_response": response,
                        "attempts": attempt + 1
                    }
                )

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        return VerificationResult(
            statement=statement,
            is_correct=False,
            strategy_used=VerificationStrategy.LLM_JUDGE,
            confidence=0.0,
            explanation="Failed to get LLM judgment",
            error_message=last_error,
            metadata={"attempts": self.max_retries}
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if self.provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.llm_client.messages.create(
                model=self.model,
                system=LLM_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2048
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _create_judgment_prompt(
        self,
        statement: str,
        context: Optional[List[str]],
        problem_statement: Optional[str]
    ) -> str:
        """Create prompt for LLM judgment."""
        prompt_parts = []

        if problem_statement:
            prompt_parts.append(f"**Problem Statement:**\n{problem_statement}\n")

        if context and len(context) > 0:
            prompt_parts.append("**Previous Steps:**")
            for i, ctx in enumerate(context, 1):
                prompt_parts.append(f"{i}. {ctx}")
            prompt_parts.append("")

        prompt_parts.append(f"**Statement to Verify:**\n{statement}\n")

        prompt_parts.append(
            "Please evaluate whether this statement is correct given the context. "
            "Provide your judgment in the following format:\n\n"
            "VERDICT: [CORRECT/INCORRECT]\n"
            "CONFIDENCE: [0.0-1.0]\n"
            "EXPLANATION: [Your detailed reasoning]"
        )

        return "\n".join(prompt_parts)

    def _parse_judgment(self, response: str) -> tuple[bool, float, str]:
        """
        Parse LLM judgment response.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (is_correct, confidence, explanation)
        """
        # Extract verdict
        verdict_match = re.search(
            r'VERDICT:\s*(CORRECT|INCORRECT)',
            response,
            re.IGNORECASE
        )

        if verdict_match:
            is_correct = verdict_match.group(1).upper() == "CORRECT"
        else:
            # Fallback: look for key phrases
            response_lower = response.lower()
            if any(phrase in response_lower for phrase in ['is correct', 'is valid', 'is true']):
                is_correct = True
            elif any(phrase in response_lower for phrase in ['is incorrect', 'is invalid', 'is false']):
                is_correct = False
            else:
                # Default to incorrect if unclear
                is_correct = False

        # Extract confidence
        confidence_match = re.search(
            r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)',
            response,
            re.IGNORECASE
        )

        if confidence_match:
            confidence = float(confidence_match.group(1))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        else:
            # Default confidence based on verdict clarity
            confidence = 0.7 if verdict_match else 0.5

        # Extract explanation
        explanation_match = re.search(
            r'EXPLANATION:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )

        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            # Use the entire response as explanation
            explanation = response.strip()

        return is_correct, confidence, explanation


# System prompt for LLM judge
LLM_JUDGE_SYSTEM_PROMPT = """You are an expert reasoning evaluator. Your task is to judge whether a reasoning step is correct given the context.

Evaluate the statement carefully considering:
1. **Logical validity**: Does the conclusion follow from the premises?
2. **Factual accuracy**: Are any facts or calculations stated correctly?
3. **Contextual consistency**: Does it align with previous steps?
4. **Completeness**: Are there any logical gaps?

Provide your judgment in this exact format:

VERDICT: [CORRECT or INCORRECT]
CONFIDENCE: [A number between 0.0 and 1.0, where 1.0 is completely certain]
EXPLANATION: [Your detailed reasoning explaining why the statement is correct or incorrect]

Guidelines:
- Be thorough but concise in your explanation
- Consider edge cases and potential errors
- If a statement is partially correct, judge based on the overall correctness
- If there's ambiguity, explain it in your reasoning and adjust confidence accordingly
- Focus on logical soundness, not just surface-level correctness

Example 1:
Statement: "If all birds can fly, and penguins are birds, then penguins can fly."
VERDICT: INCORRECT
CONFIDENCE: 0.95
EXPLANATION: While the logical structure (syllogism) is valid, the premise "all birds can fly" is factually incorrect. Penguins are flightless birds, so the conclusion is false.

Example 2:
Statement: "Since the sum is 100 and we've added 60 so far, we need to add 40 more."
VERDICT: CORRECT
CONFIDENCE: 1.0
EXPLANATION: This is correct. Basic arithmetic: 100 - 60 = 40. The reasoning is sound and the calculation is accurate.
"""
