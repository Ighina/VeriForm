"""
Main agentic verifier that routes to appropriate verification strategies.
"""

from typing import Optional, List, Dict, Any

from veriform.data_collection import ReasoningStep, ReasoningChain
from veriform.autoformalization import Autoformalization, FormalizationResult
from veriform.autoformalization.lean_verifier import LeanVerifier

from .verifier import Verifier, VerificationResult, VerificationStrategy
from .classifier import StatementClassifier, StatementType
from .python_verifier import PythonVerifier
from .llm_judge import LLMJudgeVerifier


class AgenticVerifier(Verifier):
    """
    Intelligent verifier that routes statements to appropriate verification strategies.

    Routing logic:
    - COMPUTATIONAL statements → PythonVerifier (code synthesis + execution)
    - LOGICAL_MATHEMATICAL statements → Autoformalization + Lean verification
    - GENERAL statements → LLMJudgeVerifier (LLM as judge)
    """

    def __init__(
        self,
        llm_client,  # OpenAI or Anthropic client
        model: str,
        provider: str = "openai",
        autoformalization: Optional[Autoformalization] = None,
        lean_verifier: Optional[LeanVerifier] = None,
        temperature: float = 0.0,
        enable_python_verifier: bool = True,
        enable_autoformalization: bool = True,
        enable_llm_judge: bool = True,
        fallback_to_llm_judge: bool = True,
        verbose: bool = False
    ):
        """
        Initialize agentic verifier.

        Args:
            llm_client: LLM client for Python and LLM judge verifiers
            model: Model name to use
            provider: "openai" or "anthropic"
            autoformalization: Optional Autoformalization instance
            lean_verifier: Optional LeanVerifier instance
            temperature: Sampling temperature
            enable_python_verifier: Whether to enable Python verification
            enable_autoformalization: Whether to enable Lean verification
            enable_llm_judge: Whether to enable LLM judge
            fallback_to_llm_judge: Whether to fallback to LLM judge if other methods fail
            verbose: Whether to print routing decisions
        """
        self.llm_client = llm_client
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.verbose = verbose

        # Initialize verifiers
        self.python_verifier = None
        if enable_python_verifier:
            self.python_verifier = PythonVerifier(
                llm_client=llm_client,
                model=model,
                provider=provider,
                temperature=temperature
            )

        self.autoformalization = autoformalization
        self.lean_verifier = lean_verifier
        self.enable_autoformalization = enable_autoformalization and (
            autoformalization is not None and lean_verifier is not None
        )

        self.llm_judge_verifier = None
        if enable_llm_judge:
            self.llm_judge_verifier = LLMJudgeVerifier(
                llm_client=llm_client,
                model=model,
                provider=provider,
                temperature=max(temperature, 0.2)  # Slightly higher for judgment
            )

        self.fallback_to_llm_judge = fallback_to_llm_judge

        # Validate configuration
        if not any([self.python_verifier, self.enable_autoformalization, self.llm_judge_verifier]):
            raise ValueError("At least one verification strategy must be enabled")

    def can_verify(self, statement: str) -> bool:
        """Agentic verifier can handle any statement with at least one strategy."""
        return True

    def verify(
        self,
        statement: str,
        context: Optional[List[str]] = None,
        problem_statement: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify a statement by routing to the appropriate verification strategy.

        Args:
            statement: The statement to verify
            context: Previous statements for context
            problem_statement: Original problem statement

        Returns:
            VerificationResult with verification outcome
        """
        # Classify the statement
        statement_type = StatementClassifier.classify(statement)
        classification_confidence = StatementClassifier.get_classification_confidence(statement)

        if self.verbose:
            print(f"\n[AgenticVerifier] Classifying statement...")
            print(f"  Statement: {statement}")
            print(f"  Type: {statement_type.value}")
            print(f"  Classification confidence: {classification_confidence:.2f}")

        # Route to appropriate verifier
        try:
            if statement_type == StatementType.COMPUTATIONAL and self.python_verifier:
                if self.verbose:
                    print("  → Routing to PythonVerifier")
                result = self.python_verifier.verify(statement, context, problem_statement)

            elif statement_type == StatementType.LOGICAL_MATHEMATICAL and self.enable_autoformalization:
                if self.verbose:
                    print("  → Routing to Autoformalization + Lean")
                result = self._verify_with_autoformalization(
                    statement, context, problem_statement
                )

            elif statement_type == StatementType.GENERAL and self.llm_judge_verifier:
                if self.verbose:
                    print("  → Routing to LLMJudgeVerifier")
                result = self.llm_judge_verifier.verify(statement, context, problem_statement)

            else:
                # No appropriate verifier available, try fallback
                if self.fallback_to_llm_judge and self.llm_judge_verifier:
                    if self.verbose:
                        print("  → Fallback to LLMJudgeVerifier")
                    result = self.llm_judge_verifier.verify(statement, context, problem_statement)
                else:
                    raise ValueError(
                        f"No verifier available for statement type: {statement_type.value}"
                    )

            # Add classification metadata
            result.metadata["statement_type"] = statement_type.value
            result.metadata["classification_confidence"] = classification_confidence

            if self.verbose:
                print(f"  ✓ Result: {'CORRECT' if result.is_correct else 'INCORRECT'}")
                print(f"  Confidence: {result.confidence:.2f}")

            return result

        except Exception as e:
            # If verification fails and fallback is enabled, try LLM judge
            if self.fallback_to_llm_judge and self.llm_judge_verifier:
                if self.verbose:
                    print(f"  ! Verification failed: {e}")
                    print("  → Fallback to LLMJudgeVerifier")
                result = self.llm_judge_verifier.verify(statement, context, problem_statement)
                result.metadata["fallback"] = True
                result.metadata["original_error"] = str(e)
                return result
            else:
                raise

    def _verify_with_autoformalization(
        self,
        statement: str,
        context: Optional[List[str]],
        problem_statement: Optional[str]
    ) -> VerificationResult:
        """
        Verify using autoformalization and Lean.

        Args:
            statement: Statement to verify
            context: Context statements
            problem_statement: Original problem

        Returns:
            VerificationResult
        """
        # Create a ReasoningStep for the statement
        from veriform.data_collection.reasoning_step import ReasoningStep, StepType

        step = ReasoningStep(
            step_id="verify_step",
            content=statement,
            step_type=StepType.LOGICAL_DEDUCTION,
            is_perturbed=False
        )

        # Create context steps
        context_steps = []
        if context:
            for i, ctx_content in enumerate(context):
                ctx_step = ReasoningStep(
                    step_id=f"ctx_{i}",
                    content=ctx_content,
                    step_type=StepType.OTHER,
                    is_perturbed=False
                )
                context_steps.append(ctx_step)

        # Formalize the step
        formalization_result = self.autoformalization.formalize_step(
            step=step,
            context_steps=context_steps,
            problem_statement=problem_statement or ""
        )

        # If formalization failed, return error
        if not formalization_result.success:
            return VerificationResult(
                statement=statement,
                is_correct=False,
                strategy_used=VerificationStrategy.AUTOFORMALIZATION,
                confidence=0.0,
                explanation="Autoformalization failed",
                error_message=formalization_result.error_message,
                metadata={
                    "formalization_result": formalization_result.to_dict() if hasattr(formalization_result, 'to_dict') else {}
                }
            )

        # Verify the Lean code
        verification_result = self.lean_verifier.verify(formalization_result.lean_code)

        # Map to VerificationResult
        is_correct = verification_result.success and not verification_result.has_sorry

        # Calculate confidence based on verification result
        if is_correct:
            confidence = 0.95  # High confidence for successful Lean proof
        elif verification_result.success and verification_result.has_sorry:
            confidence = 0.6  # Medium confidence - proof structure is valid but incomplete
        else:
            confidence = 0.3  # Low confidence - verification failed

        explanation = f"Lean verification: {verification_result.error_message or 'Success'}"

        return VerificationResult(
            statement=statement,
            is_correct=is_correct,
            strategy_used=VerificationStrategy.AUTOFORMALIZATION,
            confidence=confidence,
            explanation=explanation,
            metadata={
                "lean_code": formalization_result.lean_code,
                "has_sorry": verification_result.has_sorry,
                "lean_error": verification_result.error_message,
                "formalization_metadata": formalization_result.metadata
            }
        )

    def verify_reasoning_step(
        self,
        step: ReasoningStep,
        context_steps: List[ReasoningStep],
        problem_statement: str
    ) -> VerificationResult:
        """
        Verify a ReasoningStep object.

        Args:
            step: The reasoning step to verify
            context_steps: Previous steps in the chain
            problem_statement: The original problem

        Returns:
            VerificationResult
        """
        context = [s.content for s in context_steps]
        return self.verify(step.content, context, problem_statement)

    def verify_reasoning_chain(
        self,
        chain: ReasoningChain
    ) -> List[VerificationResult]:
        """
        Verify all steps in a reasoning chain.

        Args:
            chain: The reasoning chain to verify

        Returns:
            List of VerificationResults, one per step
        """
        results = []

        for i, step in enumerate(chain.steps):
            context_steps = chain.steps[:i]
            result = self.verify_reasoning_step(
                step, context_steps, chain.problem_statement
            )
            results.append(result)

        return results

    def get_routing_stats(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """
        Get statistics about verification strategy routing.

        Args:
            results: List of verification results

        Returns:
            Dictionary with routing statistics
        """
        stats = {
            "total": len(results),
            "by_strategy": {},
            "by_statement_type": {},
            "correct": 0,
            "incorrect": 0,
            "average_confidence": 0.0,
            "fallback_count": 0
        }

        if not results:
            return stats

        for result in results:
            # Count by strategy
            strategy = result.strategy_used.value
            stats["by_strategy"][strategy] = stats["by_strategy"].get(strategy, 0) + 1

            # Count by statement type
            stmt_type = result.metadata.get("statement_type", "unknown")
            stats["by_statement_type"][stmt_type] = stats["by_statement_type"].get(stmt_type, 0) + 1

            # Count correct/incorrect
            if result.is_correct:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1

            # Track fallbacks
            if result.metadata.get("fallback", False):
                stats["fallback_count"] += 1

            # Sum confidence for average
            stats["average_confidence"] += result.confidence

        # Calculate average confidence
        stats["average_confidence"] /= len(results)

        # Add accuracy
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        return stats
