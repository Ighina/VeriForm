"""
Prompt templates for autoformalization.
"""

from typing import List
from veriform.data_collection import ReasoningStep
from templates import NLVTemplate, FormalTemplate, SafeFormalTemplate, ProverTemplate, STEP_NLV_TEMPLATE, STEPS_SAFE_TEMPLATE, PROVER_TEMPLATE, IN_CONTEXT_LEAN_TEMPLATE


AUTOFORMALIZATION_SYSTEM_PROMPT = """You are an expert in formal mathematics and the Lean theorem prover. Your task is to translate natural language mathematical reasoning steps into Lean 4 code.

Important guidelines:
1. Translate the reasoning step faithfully - if the step contains an error, the Lean code should reflect that error
2. Do NOT try to correct errors in the reasoning - preserve them in the formalization
3. Use 'sorry' for previous steps provided as context
4. Return only valid Lean 4 syntax
5. Include type annotations where helpful
6. Use appropriate mathematical notation and operators"""


def create_autoformalization_prompt(
    step: ReasoningStep,
    context_steps: List[ReasoningStep],
    problem_statement: str,
    template = "safe"
) -> str:
    """
    Create a prompt for autoformalizing a reasoning step.

    Args:
        step: The reasoning step to formalize
        context_steps: Previous steps in the reasoning chain
        problem_statement: The original problem statement

    Returns:
        A formatted prompt string
    """
    if template == "safe":
        autoformalization_template = SafeFormalTemplate(STEPS_SAFE_TEMPLATE)
        prover_template = ProverTemplate(PROVER_TEMPLATE)

        return autoformalization_template, prover_template

    else:
        prompt_parts = []

        # Add problem statement
        prompt_parts.append("## Problem Statement")
        prompt_parts.append(problem_statement)
        prompt_parts.append("")

        # Add context steps as sorry lemmas
        if context_steps:
            prompt_parts.append("## Previous Steps (assume these as lemmas)")
            for i, ctx_step in enumerate(context_steps):
                prompt_parts.append(f"Step {i + 1}: {ctx_step.content}")
            prompt_parts.append("")

        # Add current step to formalize
        prompt_parts.append("## Step to Formalize")
        prompt_parts.append(step.content)
        prompt_parts.append("")

        # Add instructions
        prompt_parts.append("## Instructions")
        prompt_parts.append(
            "Translate the step above into Lean 4 code. "
            "If the reasoning contains an error, your formalization should preserve that error. "
            "Do NOT correct mistakes - translate faithfully. "
            "Return only the Lean code, enclosed in ```lean code blocks."
        )

        return "\n".join(prompt_parts)


def create_batch_autoformalization_prompt(
    steps: List[ReasoningStep],
    problem_statement: str
) -> str:
    """
    Create a prompt for autoformalizing multiple steps in sequence.

    Args:
        steps: List of reasoning steps to formalize
        problem_statement: The original problem statement

    Returns:
        A formatted prompt string
    """
    prompt_parts = []

    # Add problem statement
    prompt_parts.append("## Problem Statement")
    prompt_parts.append(problem_statement)
    prompt_parts.append("")

    # Add all steps
    prompt_parts.append("## Reasoning Steps")
    for i, step in enumerate(steps):
        prompt_parts.append(f"Step {i + 1}: {step.content}")
    prompt_parts.append("")

    # Add instructions
    prompt_parts.append("## Instructions")
    prompt_parts.append(
        "Translate all steps above into Lean 4 code. "
        "Each step should build on previous steps. "
        "If any reasoning contains errors, preserve those errors in the formalization. "
        "Return only the Lean code, enclosed in ```lean code blocks."
    )

    return "\n".join(prompt_parts)


LEAN_CODE_EXTRACTION_PATTERNS = [
    r"```lean\s*(.*?)```",
    r"```\s*(.*?)```",
    r"<code>(.*?)</code>",
]
