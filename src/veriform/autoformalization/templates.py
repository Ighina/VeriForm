"""Prompt template classes for flexible prompt generation."""

from abc import ABC, abstractmethod
import re
from typing import List, Dict, Any


class PromptTemplate(ABC):
    """Base class for prompt templates."""

    def __init__(self, template: str):
        self.template = template

    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the template with given parameters."""
        pass


class NLVTemplate(PromptTemplate):
    """Template for natural language verification prompts."""

    def format(self, problem: str, solution: str) -> str:
        """Format NLV template with problem and solution."""
        return self.template.format(problem=problem, solution=solution)


class FormalTemplate(PromptTemplate):
    """Template for formal verification prompts."""

    def format(self, language: str, input_text: str) -> str:
        """Format formal template with language and input."""
        try:
            return self.template.format(language=language, input=input_text)
        except KeyError:
            return self.template.format(input_text=input_text)

class SafeFormalTemplate(PromptTemplate):
    """Template for formalization in Safe-paper style"""

    def format(self, problem:str, previous_steps:List[str], current_step:str) -> str:
        templ = re.sub("<problem>", problem, self.template)
        _tmp = []
        for index, step in enumerate(previous_steps):
            step_form = f"Step {index}: {step}"
            _tmp.append(step_form)
        if _tmp:
            templ = re.sub("<previous_steps>", "\n".join(_tmp), templ)
        else:
            templ = re.sub("<previous_steps>", "No previous steps", templ)
        return re.sub("<current_step>", current_step, templ)
    
class ProverTemplate(PromptTemplate):
    """Template for formalization in Safe-paper style"""

    def format(self, input_text: str, **kwargs) -> str:
        #lean_code = re.findall("```lean(.+)```", input_text, flags=re.DOTALL)[-1]
        #lean_code=input_text.strip()
        return self.template.format(input_text=lean_code)


# Formalin step-wise template
STEP_NLV_TEMPLATE = """You are a Step-by-Step Logical Validator. Your goal is to convert a single step of natural language reasoning into a formal verification plan.

### STRICT CONSTRAINTS
1. SCOPE: You must ONLY analyze the text provided under "TARGET STEP". Ignore the final goal of the main problem except for context.
2. NO LOOKAHEAD: Do not generate plans for future steps. Do not solve the full problem. If the step says "X is 5", do not calculate what Y is.
3. FAITHFULNESS: Verify exactly what is written. If the step makes a claim, your plan must verify that specific claim, even if it seems trivial.
4. TRIVIALITY: If the step is a simple declaration (e.g., "Let x = 5"), the plan should be to "Define variable x and assign value 5".

### INPUT DATA

[CONTEXT / FULL PROBLEM DESCRIPTION]
{problem}

[TARGET STEP TO VERIFY]
{solution}

### OUTPUT FORMAT
Provide the verification plan for the TARGET STEP only.

**Verification Goal:** [Concise statement of the logical transition in this step]
**Formalization Strategy:** [How to translate this text into a proof state]
**What to verify:** [Specific claim or equality to check]
**How to verify:** [Detailed explanation: e.g., "Check if variable P_fri is defined as 18"]
**Required concepts:** [List concepts]

### YOUR RESPONSE"""

IN_CONTEXT_LEAN_TEMPLATE = """You are a translator that converts natural language mathematical verification plans into executable Lean 4 code. Your goal is to produce code that **actually verifies** whether mathematical claims are true or false by computation or proof.

## Core Requirements

1. **ALWAYS verify the claim computationally or by proof** - never assume claims as hypotheses.
2. **Use `rfl` (reflexivity) or `decide` for computational verification** when possible.
3. **ALWAYS use `import Mathlib` as the only import** - do not use specific submodule imports.
4. **Output ONLY the Lean 4 code** - no explanations, predictions, or additional content.
5. **DO NOT use `sorry`, axioms, or unproven assumptions** - all proofs must be complete.
6. **DO NOT change equations or values** - translate exactly as given.
7. **The code must succeed for TRUE claims and FAIL for FALSE claims** - this is how we verify correctness.

## Handling Incomplete Input (Token Limits)

Sometimes the input text provided to you may be truncated (cut off) due to token limits.
- **DO NOT** attempt to complete the English sentence.
- **DO NOT** repeat the English text.
- **DO** attempt to generate the valid Lean 4 code based on the mathematical intent visible so far.
- If the "What to verify" section is cut off, look at the "Verification Goal" or context to infer the intended check.

## Output Format

You must begin your response with the delimiter:
`### START LEAN CODE ###`
Followed immediately by the Lean code.

## Code Style Guide (STRICT)
1. **NO LINE WRAPPING:** Do not wrap long lines. Write code on a single line even if it is long.
2. **ATOMIC IDENTIFIERS:** Never insert a newline inside a variable name (e.g., write `totalFirstPhase`, NOT `total\nFirstPhase`).
3. **ONE STATEMENT PER LINE:** Each definition or theorem must be on its own line.

---

## Examples

### Example 1: True Arithmetic Claim

*INPUT PLAN:*

**Verification Goal:** Verify the power calculation.
**Formalization Strategy:** Define the exponentiation relation and check equality.
**What to verify:** 103 raised to the 6th power equals 1224238819633
**How to verify:** Compute 103^6 and check if it matches the target value.
**Required concepts:** Exponentiation, Nat.

### START LEAN CODE ###
```Lean

import Mathlib

def Root6 (x y : Nat) : Prop := y ^ 6 = x

-- This will SUCCEED because 103^6 actually equals 1224238819633
example : Root6 1224238819633 103 := by
  unfold Root6
  norm_num```

### Example 2: False Arithmetic Claim
*INPUT PLAN:*

**Verification Goal:** Verify the power calculation.
**Formalization Strategy:** Define the exponentiation.
**What to verify:** 103 raised to the 6th power equals 1061520150601
**How to verify:** Compute 103^6 and check for equality.

### START LEAN CODE ###
```Lean

import Mathlib

def Root6 (x y : Nat) : Prop := y ^ 6 = x

-- This will FAIL because 103^6 does not equal 1061520150601
example : Root6 1061520150601 103 := by
  unfold Root6
  norm_num```
### Example 3: Simple Equality (Truncated Input)
*INPUT PLAN:*

**Verification Goal:** Check simple addition.
**What to verify:** 15 + 27 equals 42
**How to verify:** Add 15 and 27 and com
(Note: Input cut off mid-sentence)

### START LEAN CODE ###
```Lean

import Mathlib

-- Input was truncated, but intent (equality check) is clear
example : (15 : Nat) + 27 = 42 := rfl```
### Example 4: Divisibility Property
*INPUT PLAN:*

**Verification Goal:** Verify divisibility.
**Formalization Strategy:** Use existential quantifier.
**What to verify:** 24 is divisible by 6
**How to verify:** Check if there exists a natural number k such that 6 * k = 24

### START LEAN CODE ###
```Lean

import Mathlib

-- This will SUCCEED
example : ∃ k : Nat, 6 * k = 24 := by
  use 4
  norm_num```
### Example 5: Let Binding (Lean 4 Syntax)
*INPUT PLAN:*

**Verification Goal:** Variable assignment and summation.
**What to verify:** Given x = 5 and y = 10, verify x + y equals 15
**How to verify:** Define x and y using let bindings, then sum them.

### START LEAN CODE ###
```Lean

import Mathlib

-- Note: Lean 4 uses semicolon, NOT "in" keyword
example : let x := 5; let y := 10; x + y = 15 := by
  norm_num```
### Example 6: Multi-step Calculation (Heavily Truncated)
*INPUT PLAN:*

**Verification Goal:** Verify calculation step.
**What to verify:** 100 - 20 = 80
**How to verify:** Perform subtra

### START LEAN CODE ###
```Lean

import Mathlib

-- Input truncated. Formalizing the visible equality claim.
example : 100 - 20 = 80 := rfl```
## Lean 4 Syntax Reminders
Use let x := v; (semicolon) NOT let x := v in

Use norm_num for arithmetic.

Use decide for inequalities.

Use fun x => for lambdas.

*INPUT PLAN:* 

{input_text}

### START LEAN CODE ###
"""

# Autoformalizer + Prover templates

STEPS_SAFE_TEMPLATE = """Given a question and the steps to answer it, you need to provide a Lean theorem that can verify the step.
* This Lean 4 theorem should support the step; if the Lean 4 theorem can be proven, then the step is correct and does not involve a hallucination.
* Ensure that the Lean theorems you provide ** CONFORM ** to the syntax of Lean 4, and ** AVOID USING NATURAL LANGUAGE ** to describe properties.
* Do ** NOT ** provide a proof method for the theorem; you can use "sorry" as a placeholder.
* Output the formalized theorem of the final step, and do ** NOT ** output any other content or predict next step.
* Note that each step is derived from the previous ones, so the theorem may require referencing information from the question or earlier steps.

Note that Lean 4 is not backward compatible with Lean 3.
* Type constants are now in UpperCamelCase, for example, `Nat` and `List`. Many variables in Mathlib have also changed to UpperCamelCase, such as `fintype` becoming `Fintype`.
* Lambda expressions now use `=>` as the separator. For example, `fun x => x` is the identity function, instead of `λ x, x`.

### Question:
Let \[f(x) = \left\{
\begin{array}{cl} ax+3, &\text{ if }x>2, \\
x-5 &\text{ if } -2 \le x \le 2, \\
2x-b &\text{ if } x <-2.
\end{array}
\right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).

### Step to be verified:
For example, $ax+3$ and $x-5$ must be equal when $x=2$.
This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$.
### Lean:
```lean
theorem test
  (a x: ℝ)
  (h₀: a * x + 3 = x - 5)
  (h₁: x = 3):
  (a = (-3)) := by sorry
```

### Step to be verified:
Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. 
Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$.
### Lean:
```lean
theorem test
  (b x: ℝ)
  (h₀: x - 5 = 2 * x - b)
  (h₁: x = -2):
  (b = 3) := by sorry
```

### Step to be verified:
So $a+b=-3+3=\boxed{0}$.
### Lean:
```lean
theorem test
  (a b: ℝ)
  (h₀: a = (-3))
  (h₁: b = 3):
  (a + b = 0) := by sorry
```

### Question:
Find the remainder when the sum \[75+76+77+78+79+80+81+82\]is divided by 16.

### Step to be verified:
We notice that 16 divides $78+82$ as well as $79+81$ and also 80.
### Lean:
```lean
theorem test:
  (16 ∣ 78 + 82) ∧ (16 ∣ 79 + 81) := by sorry
```

### Step to be verified:
Therefore the sum is congruent to  \[75+76+77\pmod{16}.\]
### Lean:
```lean
theorem test:
  (75 + 76 + 77 + 78 + 79 + 80 + 81 + 82) ≡ (75 + 76 + 77) [MOD 16] := by sorry
```

### Step to be verified:
Since these numbers are congruent to $-5$, $-4$, and $-3$ modulo 16, this can be computed as  \[-5-4-3\equiv-12\pmod{16}.\]
### Lean:
```lean
theorem test:
  (75 ≡ -5 [ZMOD 16]) ∧ (76 ≡ -4 [ZMOD 16]) ∧ (77 ≡ -3 [ZMOD 16]) ∧ (-5-4-3 = -12) := by sorry
```

### Step to be verified:
Finally, since $-12\equiv4\pmod{16}$ the remainder we seek is $\boxed{4}$.
### Lean:
```lean
theorem test:
  (-12 ≡ 4 [ZMOD 16]) ∧ (4 < 16) := by sorry
```

### Question
<problem>

### Steps that do not require verification:
<previous_steps>
### Step to be verified:
<current_step>
### Lean:"""

PROVER_TEMPLATE = """Complete the following Lean 4 code to prove that the given unproved statement is correct. If the statement is not correct, the resulting code should fail:

```lean
{input_text}

-- complete your proof here below: 
"""