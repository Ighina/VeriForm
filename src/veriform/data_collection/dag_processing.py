"""
Mathematical Reasoning DAG Parser with Structured Outputs
Transforms ProcessBench reasoning chains into annotated DAGs using GPT-5, Gemini 3, and Claude 4.5
"""

import sys
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum
import json
import os

from datasets import load_dataset
import pandas as pd

# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================


class Metadata(BaseModel):
    model_config = {"extra": "forbid"}
    # add explicit fields if you know them
    notes: Optional[str] = None
    difficulty: Optional[str] = None


class ProofType(str, Enum):
    """Hilbert-style proof classification"""

    AXIOM = "axiom"  # Does not depend on previous steps
    ASSUMPTION = "assumption"  # Initial given/assumption
    DEDUCTION = "deduction"  # Derives from previous steps via inference


class StatementType(str, Enum):
    """Type of mathematical statement"""

    COMPUTATION = "computation"  # Arithmetic/algebraic calculation
    LOGICAL = "logical"  # Logical inference (if-then, therefore, etc.)
    DECLARATIVE = "declarative"  # Variable assignment, definition
    GENERAL_KNOWLEDGE = "general_knowledge"  # Known facts, theorems
    SIMPLIFICATION = "simplification"  # Algebraic simplification
    SUBSTITUTION = "substitution"  # Variable substitution
    VERIFICATION = "verification"  # Checking/verifying a result


class ReasoningNode(BaseModel):
    """A single node in the reasoning DAG"""

    node_id: str = Field(description="Unique identifier for this node (e.g., 'step_1')")
    content: str = Field(description="The actual reasoning step or statement")
    proof_type: ProofType = Field(description="Classification as axiom/assumption/deduction")
    statement_type: StatementType = Field(description="Type of mathematical statement")
    is_verifiable: bool = Field(description="Whether this step can be independently verified")
    dependencies: List[str] = Field(
        default_factory=list, description="List of node_ids this step depends on"
    )
    inference_rule: Optional[str] = Field(
        default=None,
        description="The inference rule used (e.g., 'modus ponens', 'substitution', 'arithmetic')",
    )
    verification_note: Optional[str] = Field(
        default=None, description="Explanation of why this is/isn't verifiable"
    )


class ReasoningDAG(BaseModel):
    """Complete DAG representation of mathematical reasoning"""

    nodes: List[ReasoningNode] = Field(description="All nodes in the reasoning chain")
    problem_statement: str = Field(description="The original problem being solved")
    final_answer: str = Field(description="The final answer/conclusion")
    metadata: Optional[Metadata] = Field(default=None, description="Additional metadata")


# ============================================================================
# LLM Client Implementations
# ============================================================================


class OpenAIClient:
    """OpenAI GPT-5 client with structured outputs"""

    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def parse_reasoning_chain(
        self, reasoning_chain: str, model: str = "gpt-5.2-2025-12-11"
    ) -> ReasoningDAG:
        """Parse reasoning chain using OpenAI structured outputs"""

        prompt = self._build_prompt(reasoning_chain)

        completion = self.client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format=ReasoningDAG,
        )

        return completion.choices[0].message.parsed

    def _get_system_prompt(self) -> str:
        return """You are an expert in mathematical logic and proof theory.
Your task is to analyze mathematical reasoning chains and represent them as directed acyclic graphs (DAGs).

For each step in the reasoning:
1. Classify it as axiom/assumption (no dependencies) or deduction (depends on prior steps)
2. Identify the type of statement (computation, logical inference, etc.)
3. Determine if it's verifiable (can be independently checked)
4. List all dependencies (previous steps it relies on)
5. Name the inference rule used if it's a deduction

DO NOT INCLUDE THE PROBLEM STATEMENT AMONG THE DAG STEPS.

Be precise and thorough in your analysis."""

    def _build_prompt(self, reasoning_chain: str) -> str:
        return f"""Analyze this mathematical reasoning chain and convert it into a structured DAG:

{reasoning_chain}

Create a complete DAG representation with:
- Each reasoning step as a node
- Proper dependency relationships
- Hilbert-style proof annotations
- Verifiability assessment
- Statement type classification"""


class GeminiClient:
    """Google Gemini 3 client with structured outputs"""

    def __init__(self, api_key: Optional[str] = None):
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-3-pro")  # Update with actual model name

    def parse_reasoning_chain(self, reasoning_chain: str) -> ReasoningDAG:
        """Parse reasoning chain using Gemini structured outputs"""

        prompt = self._build_full_prompt(reasoning_chain)

        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=ReasoningDAG, temperature=0.1
            ),
        )

        # Parse JSON response into Pydantic model
        return ReasoningDAG.model_validate_json(response.text)

    def _build_full_prompt(self, reasoning_chain: str) -> str:
        system_prompt = """You are an expert in mathematical logic and proof theory.
Analyze mathematical reasoning chains and represent them as directed acyclic graphs (DAGs).

For each step, classify it using Hilbert-style proof theory, identify dependencies,
and assess verifiability.

DO NOT INCLUDE THE PROBLEM STATEMENT AMONG THE DAG STEPS."""

        user_prompt = f"""Analyze this mathematical reasoning chain and convert it into a structured DAG:

{reasoning_chain}

Provide a complete JSON representation following the ReasoningDAG schema."""

        return f"{system_prompt}\n\n{user_prompt}"


class AnthropicClient:
    """Anthropic Claude 4.5 client with structured outputs"""

    def __init__(self, api_key: Optional[str] = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def parse_reasoning_chain(
        self, reasoning_chain: str, model: str = "claude-sonnet-4-5-20250929"
    ) -> ReasoningDAG:
        """Parse reasoning chain using Claude structured outputs"""

        prompt = self._build_prompt(reasoning_chain)

        response = self.client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.1,
            system=self._get_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract JSON from response and parse with Pydantic
        response_text = response.content[0].text

        # Claude returns clean JSON, parse directly
        json_data = json.loads(response_text)
        return ReasoningDAG.model_validate(json_data)

    def _get_system_prompt(self) -> str:
        return """You are an expert in mathematical logic and proof theory.
Your task is to analyze mathematical reasoning chains and represent them as directed acyclic graphs (DAGs).

For each step in the reasoning:
1. Classify it as axiom/assumption (no dependencies) or deduction (depends on prior steps)
2. Identify the type of statement (computation, logical inference, etc.)
3. Determine if it's verifiable (can be independently checked)
4. List all dependencies (previous steps it relies on)
5. Name the inference rule used if it's a deduction

DO NOT INCLUDE THE PROBLEM STATEMENT AMONG THE DAG STEPS.

Return your analysis as a JSON object matching this structure:
{
  "nodes": [
    {
      "node_id": "step_1",
      "content": "The reasoning step",
      "proof_type": "axiom|assumption|deduction",
      "statement_type": "computation|logical|declarative|etc",
      "is_verifiable": true|false,
      "dependencies": ["step_0"],
      "inference_rule": "rule name",
      "verification_note": "explanation"
    }
  ],
  "problem_statement": "original problem",
  "final_answer": "solution",
  "metadata": {}
}

Be precise and thorough. Return ONLY valid JSON."""

    def _build_prompt(self, reasoning_chain: str) -> str:
        return f"""Analyze this mathematical reasoning chain and convert it into a structured DAG:

{reasoning_chain}

Create a complete DAG representation with:
- Each reasoning step as a node with unique node_id
- Proper dependency relationships between nodes
- Hilbert-style proof annotations (axiom/assumption/deduction)
- Verifiability assessment for each step
- Statement type classification
- Inference rules for deductions

Return the result as a JSON object."""


# ============================================================================
# Unified Interface
# ============================================================================


class ReasoningDAGParser:
    """Unified interface for parsing reasoning chains with multiple LLM providers"""

    def __init__(
        self,
        provider: Literal["openai", "gemini", "anthropic"] = "anthropic",
        api_key: Optional[str] = None,
    ):
        """
        Initialize parser with specified provider

        Args:
            provider: LLM provider to use
            api_key: API key (if None, reads from environment)
        """
        if provider == "openai":
            self.client = OpenAIClient(api_key)
        elif provider == "gemini":
            self.client = GeminiClient(api_key)
        elif provider == "anthropic":
            self.client = AnthropicClient(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.provider = provider

    def parse(self, reasoning_chain: str) -> ReasoningDAG:
        """Parse reasoning chain into DAG structure"""
        return self.client.parse_reasoning_chain(reasoning_chain)

    def parse_to_json(self, reasoning_chain: str, indent: int = 2) -> str:
        """Parse and return as formatted JSON string"""
        dag = self.parse(reasoning_chain)
        return dag.model_dump_json(indent=indent)

    def validate_dag(self, dag: ReasoningDAG) -> tuple[bool, List[str]]:
        """
        Validate DAG structure

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        node_ids = {node.node_id for node in dag.nodes}

        # Check all dependencies exist
        for node in dag.nodes:
            for dep in node.dependencies:
                if dep not in node_ids:
                    errors.append(f"Node {node.node_id} depends on non-existent node {dep}")

        # Check for cycles (simplified check)
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            node = next((n for n in dag.nodes if n.node_id == node_id), None)
            if node:
                for dep in node.dependencies:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for node in dag.nodes:
            if node.node_id not in visited:
                if has_cycle(node.node_id):
                    errors.append("DAG contains a cycle")
                    break

        return len(errors) == 0, errors


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """Example of how to use the parser"""

    # Example reasoning chain from ProcessBench
    reasoning_chain = """
    Problem: Solve for x: 2x + 5 = 15

    Step 1: Start with the equation 2x + 5 = 15
    Step 2: Subtract 5 from both sides: 2x = 10
    Step 3: Divide both sides by 2: x = 5
    Step 4: Verify: 2(5) + 5 = 10 + 5 = 15 ✓
    """

    # Parse with Claude
    parser = ReasoningDAGParser(provider="openai")
    dag = parser.parse(reasoning_chain)

    # Print results
    print("Parsed DAG:")
    print(parser.parse_to_json(reasoning_chain))

    # Validate
    is_valid, errors = parser.validate_dag(dag)
    print(f"\nValidation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    # Access individual nodes
    print("\nNodes:")
    for node in dag.nodes:
        print(f"  {node.node_id}: {node.content[:50]}...")
        print(f"    Type: {node.proof_type.value}, Verifiable: {node.is_verifiable}")
        print(f"    Dependencies: {node.dependencies}")


# ============================================================================
# LLM-as-a-Judge for DAG Faithfulness Evaluation
# ============================================================================

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class FaithfulnessIssue(BaseModel):
    """A specific faithfulness issue found in the DAG"""

    issue_type: Literal[
        "content_distortion",
        "missing_content",
        "added_content",
        "incorrect_dependency",
        "missing_dependency",
        "spurious_dependency",
        "incorrect_ordering",
    ] = Field(description="Type of faithfulness issue")

    # REMOVED: default=None
    node_id: Optional[str] = Field(description="The node ID where the issue occurs")
    description: str = Field(description="Detailed description of the issue")
    severity: Literal["critical", "major", "minor"] = Field(description="Severity of the issue")

    # REMOVED: default=None
    original_content: Optional[str] = Field(
        description="Relevant content from original reasoning chain"
    )
    # REMOVED: default=None
    dag_content: Optional[str] = Field(description="Corresponding content in DAG")


class FaithfulnessEvaluation(BaseModel):
    """Complete evaluation of DAG faithfulness"""

    is_faithful: bool = Field(
        description="Whether the DAG is faithful to the original reasoning chain"
    )
    overall_score: float = Field(description="Overall score from 0.0 to 1.0")
    content_preservation_score: float = Field(description="Score from 0.0 to 1.0 for content")
    dependency_accuracy_score: float = Field(description="Score from 0.0 to 1.0 for dependencies")
    completeness_score: float = Field(description="Score from 0.0 to 1.0 for completeness")

    # REMOVED: default_factory=list
    issues: List[FaithfulnessIssue] = Field(
        description="List of specific faithfulness issues found"
    )
    summary: str = Field(description="Summary of the evaluation")

    # REMOVED: default_factory=list
    recommendations: List[str] = Field(description="Recommendations for improving the DAG")


class OpenAIJudge:
    """OpenAI GPT-5 as a judge for DAG faithfulness"""

    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def evaluate_faithfulness(
        self, original_chain: str, dag: ReasoningDAG, model: str = "gpt-5"
    ) -> FaithfulnessEvaluation:
        """Evaluate DAG faithfulness using OpenAI structured outputs"""

        prompt = self._build_evaluation_prompt(original_chain, dag)

        completion = self.client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": self._get_judge_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format=FaithfulnessEvaluation,
            temperature=0.1,
        )

        return completion.choices[0].message.parsed

    def _get_judge_system_prompt(self) -> str:
        return """You are an expert judge evaluating the faithfulness of mathematical reasoning DAG representations.

Your task is to determine whether a DAG accurately represents the original reasoning chain by checking:

1. **Content Preservation**: Is the meaning of each step preserved without distortion?
2. **Dependency Accuracy**: Are the dependency relationships correct?
   - Does each step depend on exactly the steps it should?
   - Are there missing dependencies?
   - Are there spurious (incorrect) dependencies?
3. **Completeness**: Are all reasoning steps from the original chain captured?
4. **Ordering**: Is the logical flow preserved?

Be thorough and precise. Identify specific issues with severity levels:
- **Critical**: Fundamentally breaks the reasoning (wrong dependencies on key steps, major distortions)
- **Major**: Significant but not fatal (missing important dependencies, notable content changes)
- **Minor**: Small issues that don't affect overall reasoning (minor wording changes, trivial omissions)

Provide detailed analysis with specific examples from both the original and the DAG."""

    def _build_evaluation_prompt(self, original_chain: str, dag: ReasoningDAG) -> str:
        dag_json = dag.model_dump_json(indent=2)

        return f"""Evaluate the faithfulness of this DAG representation against the original reasoning chain.

ORIGINAL REASONING CHAIN:
{original_chain}

DAG REPRESENTATION:
{dag_json}

Analyze:
1. Is each node's content faithful to the original?
2. Are all dependency relationships correct?
3. Are there any missing or extra steps?
4. Is the logical flow preserved?

Provide a comprehensive evaluation with specific issues identified."""


class GeminiJudge:
    """Google Gemini 3 as a judge for DAG faithfulness"""

    def __init__(self, api_key: Optional[str] = None):
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-3-flash-preview")

    def evaluate_faithfulness(
        self, original_chain: str, dag: ReasoningDAG
    ) -> FaithfulnessEvaluation:
        """Evaluate DAG faithfulness using Gemini structured outputs"""

        prompt = self._build_full_evaluation_prompt(original_chain, dag)

        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=FaithfulnessEvaluation,
                temperature=0.1,
            ),
        )

        return FaithfulnessEvaluation.model_validate_json(response.text)

    def _build_full_evaluation_prompt(self, original_chain: str, dag: ReasoningDAG) -> str:
        dag_json = dag.model_dump_json(indent=2)

        system_prompt = """You are an expert judge evaluating mathematical reasoning DAG representations.

Assess faithfulness by checking:
- Content preservation (no distortion)
- Dependency accuracy (correct relationships)
- Completeness (all steps captured)
- Logical ordering

Identify specific issues with severity levels and provide detailed analysis."""

        user_prompt = f"""Evaluate this DAG's faithfulness to the original reasoning chain.

ORIGINAL REASONING CHAIN:
{original_chain}

DAG REPRESENTATION:
{dag_json}

Provide a comprehensive evaluation following the FaithfulnessEvaluation schema."""

        return f"{system_prompt}\n\n{user_prompt}"


class AnthropicJudge:
    """Anthropic Claude 4.5 as a judge for DAG faithfulness"""

    def __init__(self, api_key: Optional[str] = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def evaluate_faithfulness(
        self, original_chain: str, dag: ReasoningDAG, model: str = "claude-sonnet-4-5-20250929"
    ) -> FaithfulnessEvaluation:
        """Evaluate DAG faithfulness using Claude structured outputs"""

        prompt = self._build_evaluation_prompt(original_chain, dag)

        response = self.client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.1,
            system=self._get_judge_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text
        json_data = json.loads(response_text)
        return FaithfulnessEvaluation.model_validate(json_data)

    def _get_judge_system_prompt(self) -> str:
        return """You are an expert judge evaluating the faithfulness of mathematical reasoning DAG representations.

Your task is to determine whether a DAG accurately represents the original reasoning chain by checking:

1. **Content Preservation**: Is the meaning of each step preserved without distortion?
   - Check for paraphrasing that changes meaning
   - Verify mathematical expressions are unchanged
   - Ensure no information is lost or added

2. **Dependency Accuracy**: Are the dependency relationships correct?
   - Does each step depend on exactly the steps it should?
   - Are there missing dependencies (step uses info from another but doesn't list it)?
   - Are there spurious dependencies (step lists dependencies it doesn't actually use)?

3. **Completeness**: Are all reasoning steps from the original chain captured?
   - Check for missing steps
   - Verify no steps were combined incorrectly
   - Ensure no extra steps were added

4. **Ordering**: Is the logical flow preserved?
   - Verify topological ordering makes sense
   - Check that dependencies create a valid DAG

Be thorough and precise. Identify specific issues with severity levels:
- **Critical**: Fundamentally breaks the reasoning
- **Major**: Significant but not fatal
- **Minor**: Small issues that don't affect overall reasoning

Return your evaluation as a JSON object matching this structure:
{
  "is_faithful": true|false,
  "overall_score": 0.0-1.0,
  "content_preservation_score": 0.0-1.0,
  "dependency_accuracy_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "issues": [
    {
      "issue_type": "content_distortion|missing_content|...",
      "node_id": "step_1",
      "description": "detailed description",
      "severity": "critical|major|minor",
      "original_content": "from original",
      "dag_content": "from dag"
    }
  ],
  "summary": "overall summary",
  "recommendations": ["recommendation 1", "recommendation 2"]
}

Return ONLY valid JSON."""

    def _build_evaluation_prompt(self, original_chain: str, dag: ReasoningDAG) -> str:
        dag_json = dag.model_dump_json(indent=2)

        return f"""Evaluate the faithfulness of this DAG representation against the original reasoning chain.

ORIGINAL REASONING CHAIN:
{original_chain}

DAG REPRESENTATION:
{dag_json}

Perform a detailed analysis:
1. Compare each node's content to the original steps
2. Verify all dependency relationships are correct
3. Check for missing or extra steps
4. Assess preservation of logical flow

Provide a comprehensive evaluation with:
- Overall faithfulness score and component scores
- Specific issues identified with severity levels
- Summary of findings
- Recommendations for improvement

Return the result as a JSON object."""


class DAGFaithfulnessJudge:
    """Unified interface for judging DAG faithfulness across LLM providers"""

    def __init__(
        self,
        provider: Literal["openai", "gemini", "anthropic"] = "anthropic",
        api_key: Optional[str] = None,
    ):
        """
        Initialize judge with specified provider

        Args:
            provider: LLM provider to use as judge
            api_key: API key (if None, reads from environment)
        """
        if provider == "openai":
            self.judge = OpenAIJudge(api_key)
        elif provider == "gemini":
            self.judge = GeminiJudge(api_key)
        elif provider == "anthropic":
            self.judge = AnthropicJudge(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.provider = provider

    def evaluate(self, original_chain: str, dag: ReasoningDAG) -> FaithfulnessEvaluation:
        """Evaluate DAG faithfulness"""
        return self.judge.evaluate_faithfulness(original_chain, dag)

    def evaluate_to_json(self, original_chain: str, dag: ReasoningDAG, indent: int = 2) -> str:
        """Evaluate and return as formatted JSON string"""
        evaluation = self.evaluate(original_chain, dag)
        return evaluation.model_dump_json(indent=indent)

    def print_evaluation_report(self, evaluation: FaithfulnessEvaluation, verbose: bool = True):
        """Print a human-readable evaluation report"""

        print("=" * 80)
        print("DAG FAITHFULNESS EVALUATION REPORT")
        print("=" * 80)

        # Overall assessment
        status = "✓ FAITHFUL" if evaluation.is_faithful else "✗ UNFAITHFUL"
        print(f"\nOverall Assessment: {status}")
        print(f"Overall Score: {evaluation.overall_score:.2f}/1.00")

        # Component scores
        print("\nComponent Scores:")
        print(f"  Content Preservation: {evaluation.content_preservation_score:.2f}/1.00")
        print(f"  Dependency Accuracy:  {evaluation.dependency_accuracy_score:.2f}/1.00")
        print(f"  Completeness:         {evaluation.completeness_score:.2f}/1.00")

        # Issues
        if evaluation.issues:
            print(f"\nIssues Found: {len(evaluation.issues)}")

            # Group by severity
            critical = [i for i in evaluation.issues if i.severity == "critical"]
            major = [i for i in evaluation.issues if i.severity == "major"]
            minor = [i for i in evaluation.issues if i.severity == "minor"]

            if critical:
                print(f"\n  🔴 Critical Issues: {len(critical)}")
                if verbose:
                    for issue in critical:
                        self._print_issue(issue)

            if major:
                print(f"\n  🟡 Major Issues: {len(major)}")
                if verbose:
                    for issue in major:
                        self._print_issue(issue)

            if minor:
                print(f"\n  🟢 Minor Issues: {len(minor)}")
                if verbose:
                    for issue in minor:
                        self._print_issue(issue)
        else:
            print("\n✓ No issues found")

        # Summary
        print(f"\nSummary:")
        print(f"  {evaluation.summary}")

        # Recommendations
        if evaluation.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(evaluation.recommendations, 1):
                print(f"  {i}. {rec}")

        print("\n" + "=" * 80)

    def _print_issue(self, issue: FaithfulnessIssue):
        """Print a single issue with details"""
        print(f"\n    Type: {issue.issue_type}")
        if issue.node_id:
            print(f"    Node: {issue.node_id}")
        print(f"    Description: {issue.description}")
        if issue.original_content:
            print(f"    Original: {issue.original_content[:100]}...")
        if issue.dag_content:
            print(f"    DAG: {issue.dag_content[:100]}...")


# ============================================================================
# Multi-Judge Ensemble
# ============================================================================


class EnsembleJudge:
    """Use multiple LLMs as judges and aggregate their evaluations"""

    def __init__(
        self,
        providers: List[Literal["openai", "gemini", "anthropic"]] = None,
        api_keys: Optional[dict] = None,
    ):
        """
        Initialize ensemble of judges

        Args:
            providers: List of providers to use (default: all three)
            api_keys: Dict mapping provider names to API keys
        """
        if providers is None:
            providers = ["openai", "gemini", "anthropic"]

        api_keys = api_keys or {}

        self.judges = {
            provider: DAGFaithfulnessJudge(provider, api_keys.get(provider))
            for provider in providers
        }

    def evaluate_ensemble(self, original_chain: str, dag: ReasoningDAG) -> dict:
        """
        Evaluate using all judges and return aggregated results

        Returns:
            Dict with individual evaluations and consensus metrics
        """
        evaluations = {}

        for provider, judge in self.judges.items():
            print(f"Evaluating with {provider}...")
            evaluations[provider] = judge.evaluate(original_chain, dag)

        # Compute consensus
        consensus = self._compute_consensus(evaluations)

        return {"individual_evaluations": evaluations, "consensus": consensus}

    def _compute_consensus(self, evaluations: dict) -> dict:
        """Compute consensus metrics across judges"""

        scores = {"overall": [], "content": [], "dependency": [], "completeness": []}

        faithful_votes = []
        all_issues = []

        for provider, eval_result in evaluations.items():
            scores["overall"].append(eval_result.overall_score)
            scores["content"].append(eval_result.content_preservation_score)
            scores["dependency"].append(eval_result.dependency_accuracy_score)
            scores["completeness"].append(eval_result.completeness_score)

            faithful_votes.append(eval_result.is_faithful)
            all_issues.extend(eval_result.issues)

        # Average scores
        avg_scores = {key: sum(values) / len(values) for key, values in scores.items()}

        # Majority vote on faithfulness
        is_faithful_consensus = sum(faithful_votes) > len(faithful_votes) / 2

        # Agreement rate
        agreement_rate = sum(faithful_votes) / len(faithful_votes)

        return {
            "is_faithful": is_faithful_consensus,
            "agreement_rate": agreement_rate,
            "average_scores": avg_scores,
            "score_ranges": {
                key: {"min": min(values), "max": max(values), "std": self._std_dev(values)}
                for key, values in scores.items()
            },
            "total_issues_found": len(all_issues),
            "critical_issues": len([i for i in all_issues if i.severity == "critical"]),
            "major_issues": len([i for i in all_issues if i.severity == "major"]),
            "minor_issues": len([i for i in all_issues if i.severity == "minor"]),
        }

    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def print_ensemble_report(self, results: dict):
        """Print comprehensive ensemble evaluation report"""

        print("\n" + "=" * 80)
        print("ENSEMBLE JUDGE EVALUATION REPORT")
        print("=" * 80)

        consensus = results["consensus"]

        # Consensus verdict
        status = "✓ FAITHFUL" if consensus["is_faithful"] else "✗ UNFAITHFUL"
        print(f"\nConsensus Verdict: {status}")
        print(f"Judge Agreement: {consensus['agreement_rate']:.1%}")

        # Average scores
        print("\nAverage Scores Across Judges:")
        avg = consensus["average_scores"]
        print(f"  Overall:     {avg['overall']:.3f}")
        print(f"  Content:     {avg['content']:.3f}")
        print(f"  Dependency:  {avg['dependency']:.3f}")
        print(f"  Completeness: {avg['completeness']:.3f}")

        # Score ranges
        print("\nScore Consistency:")
        for metric, ranges in consensus["score_ranges"].items():
            print(
                f"  {metric.capitalize():12} - Range: [{ranges['min']:.2f}, {ranges['max']:.2f}], "
                f"Std Dev: {ranges['std']:.3f}"
            )

        # Issues summary
        print(f"\nTotal Issues Found: {consensus['total_issues_found']}")
        print(f"  Critical: {consensus['critical_issues']}")
        print(f"  Major:    {consensus['major_issues']}")
        print(f"  Minor:    {consensus['minor_issues']}")

        # Individual judge results
        print("\nIndividual Judge Results:")
        for provider, evaluation in results["individual_evaluations"].items():
            status_icon = "✓" if evaluation.is_faithful else "✗"
            print(f"\n  {provider.upper()} {status_icon}")
            print(f"    Score: {evaluation.overall_score:.3f}")
            print(
                f"    Issues: {len(evaluation.issues)} "
                f"({len([i for i in evaluation.issues if i.severity == 'critical'])} critical)"
            )

        print("\n" + "=" * 80)


# ============================================================================
# Updated Example Usage
# ============================================================================


def example_usage_with_judge():
    """Example demonstrating parsing and judging"""

    # Example reasoning chain
    reasoning_chain = """
    Problem: Solve for x: 2x + 5 = 15

    Step 1: Start with the equation 2x + 5 = 15
    Step 2: Subtract 5 from both sides: 2x + 5 - 5 = 15 - 5, which gives 2x = 10
    Step 3: Divide both sides by 2: 2x/2 = 10/2, which gives x = 5
    Step 4: Verify by substitution: 2(5) + 5 = 10 + 5 = 15 ✓
    """

    print("=" * 80)
    print("STEP 1: PARSING REASONING CHAIN TO DAG")
    print("=" * 80)

    # Parse with Claude
    parser = ReasoningDAGParser(provider="openai")
    dag = parser.parse(reasoning_chain)

    print("\n✓ DAG created successfully")
    print(f"Nodes: {len(dag.nodes)}")
    print(f"Problem: {dag.problem_statement}")
    print(f"Answer: {dag.final_answer}")

    print("\n" + "=" * 80)
    print("STEP 2: EVALUATING DAG FAITHFULNESS")
    print("=" * 80)

    # Evaluate with single judge
    judge = DAGFaithfulnessJudge(provider="gemini")
    evaluation = judge.evaluate(reasoning_chain, dag)
    judge.print_evaluation_report(evaluation)

    # Optional: Use ensemble of judges
    print("\n" + "=" * 80)
    print("STEP 3: ENSEMBLE EVALUATION (OPTIONAL)")
    print("=" * 80)
    print("\nNote: This requires API keys for all three providers")
    print("Uncomment the code below to run ensemble evaluation:")
    print(
        """
    ensemble = EnsembleJudge(providers=["anthropic"])  # Add more providers
    results = ensemble.evaluate_ensemble(reasoning_chain, dag)
    ensemble.print_ensemble_report(results)
    """
    )


def chains_to_dag(reasoning_chains: List[str], limit=None):
    """Example of how to use the parser"""
    dags = []

    if limit is not None:
        reasoning_chains = reasoning_chains[:limit]

    for reasoning_chain in reasoning_chains:
        # Parse with Claude
        parser = ReasoningDAGParser(provider="openai")
        dag = parser.parse(reasoning_chain)

        dags.append(dag)

        # Validate
        is_valid, errors = parser.validate_dag(dag)
        if errors:
            for error in errors:
                print(f"  - {error}")
    return dags


def evaluate_dag(reasoning_chains: List[str], dags: List[str]):
    """Example demonstrating parsing and judging"""

    evaluations = []

    for reasoning_chain, dag in zip(reasoning_chains, dags):

        # Evaluate with single judge
        judge = DAGFaithfulnessJudge(provider="gemini")
        evaluation = judge.evaluate(reasoning_chain, dag)
        evaluations.append(evaluation)
        # judge.print_evaluation_report(evaluation)

    return evaluations


if __name__ == "__main__":
    task = sys.argv[1]  # 'parse' or 'evaluate'

    if task == "parse":

        ds = load_dataset("Qwen/ProcessBench", "default")

        datasets = []

        for dataset in ds:
            pandas_ds = ds[dataset].to_pandas()
            pandas_ds = pandas_ds[pandas_ds["label"] == -1]
            pandas_ds["split"] = dataset
            reasoning_chains = []
            for idx in range(len(pandas_ds)):
                problem_statement = pandas_ds.iloc[idx]["problem"]
                steps = " ".join(pandas_ds.iloc[idx]["steps"])
                reasoning_chains.append(f"problem statement: {problem_statement}\n\n{steps}")
            dags = chains_to_dag(reasoning_chains)
            dags_json = [dag.model_dump_json(indent=2) for dag in dags]
            pandas_ds["dags"] = dags_json

            datasets.append(pandas_ds)

            pd.concat(datasets).to_json("dag_processbench_final.json")
    elif task == "evaluate":
        NotImplementedError("Evaluation script not implemented yet")
    else:
        print(f"Unknown task: {task}")
