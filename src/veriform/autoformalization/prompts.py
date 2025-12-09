"""
Prompt templates for autoformalization.
"""

from typing import List
from veriform.data_collection import ReasoningStep
from .templates import NLVTemplate, FormalTemplate, SafeFormalTemplate, ProverTemplate, STEP_NLV_TEMPLATE, STEPS_SAFE_TEMPLATE, PROVER_TEMPLATE, IN_CONTEXT_LEAN_TEMPLATE
from .templates import PROVER_TEMPLATE_DEEPSEEK, LEAN_WRAPPER_TEMPLATE

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
        # prover_template = ProverTemplate(PROVER_TEMPLATE)
        return autoformalization_template
        # return autoformalization_template, prover_template

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


def create_deepseek_prover_prompt(
    lean_code_with_sorry: str,
    statement: str = "",
) -> str:
    """
    Create a prompt for the DeepSeek prover model.

    Args:
        lean_code_with_sorry: The Lean code containing 'sorry' placeholders
        statement: The original problem statement (optional)
    Returns:
        A formatted prompt string
    """
    lean_code = LEAN_WRAPPER_TEMPLATE.format(lean_code=lean_code_with_sorry, statement=statement)
    prompt = PROVER_TEMPLATE_DEEPSEEK.format(lean_code_with_sorry=lean_code)
    return prompt

def parse_deepseek_prover_response(
    response: str
) -> str:
    """
    Parse the response from the DeepSeek prover model to extract Lean code.

    Args:
        response: The raw response string from the model
    Returns:
        Extracted Lean code as a string
    """
    import re
    # Pattern list: from strict to loose
    lean_patterns = [
        # Pattern 1: Strict - Code block with lean4 language identifier
        r'```lean4\s*\n(.*?)```',
        # Pattern 2: Medium - Code block with lean language identifier
        r'```lean\s*\n(.*?)```',
        # Pattern 3: Loose - Any code block with triple backticks
        r'```\s*\n(.*?)```',
        # Pattern 4: Very loose - Extract Lean statements without code blocks
        r'((?:def|theorem|variable|lemma|axiom)\s+\w+[^\n]*(?:\n(?:  |\t|[^\n])*)*)',
    ]
        
    for pattern in lean_patterns:
        matches = list(re.finditer(pattern, response, re.DOTALL))
        if matches:
            # Get the LAST match
            code = matches[-1].group(1).strip()
            # For pattern 3, verify it looks like Lean code
            if pattern == lean_patterns[2]:  # Generic code block
                if not re.search(r'\b(def|theorem|variable|lemma|axiom)\b', code):
                    continue
            return code
    
    # Last resort: check if entire response looks like Lean code
    if re.search(r'\b(def|theorem|variable)\b', response):
        # Clean up markdown artifacts
        cleaned = response.strip()
        cleaned = re.sub(r'^```.*?\n', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n```.*?$', '', cleaned, flags=re.MULTILINE)
        return cleaned.strip()
    
    # If nothing worked, raise an error
    raise ValueError(f"Could not parse Lean code from LLM output:\n{response}")