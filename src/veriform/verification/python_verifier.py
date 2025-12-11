"""
Python code synthesis and execution for computational verification.
"""

import re
import ast
import sys
import time
import math
from io import StringIO
from typing import Optional, List, Dict, Any
from contextlib import redirect_stdout, redirect_stderr

from .verifier import Verifier, VerificationResult, VerificationStrategy
from .classifier import StatementClassifier, StatementType


class PythonVerifier(Verifier):
    """
    Verifies computational statements by synthesizing and executing Python code.

    Uses LLM to generate Python code that evaluates the statement, then
    executes the code in a sandboxed environment to get True/False result.
    """

    def __init__(
        self,
        llm_client,  # OpenAI or Anthropic client
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        timeout: int = 5,
        max_retries: int = 3
    ):
        """
        Initialize Python verifier.

        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model name to use
            provider: "openai" or "anthropic"
            temperature: Sampling temperature
            timeout: Execution timeout in seconds
            max_retries: Maximum retry attempts for code generation
        """
        self.llm_client = llm_client
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def can_verify(self, statement: str) -> bool:
        """Check if this is a computational statement."""
        return StatementClassifier.classify(statement) == StatementType.COMPUTATIONAL

    def verify(
        self,
        statement: str,
        context: Optional[List[str]] = None,
        problem_statement: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify a computational statement using Python synthesis.

        Args:
            statement: The statement to verify
            context: Previous statements for context
            problem_statement: Original problem statement

        Returns:
            VerificationResult with True/False outcome
        """
        # Generate Python code
        python_code = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                python_code = self._generate_python_code(
                    statement, context, problem_statement
                )

                if python_code:
                    # Try to execute the code
                    is_correct, output, error = self._execute_python_code(python_code)

                    if error is None:
                        # Successfully executed
                        confidence = 0.9 if is_correct else 0.85
                        explanation = f"Verified by Python execution. Output: {output}"

                        return VerificationResult(
                            statement=statement,
                            is_correct=is_correct,
                            strategy_used=VerificationStrategy.PYTHON_SYNTHESIS,
                            confidence=confidence,
                            explanation=explanation,
                            metadata={
                                "python_code": python_code,
                                "output": output,
                                "attempts": attempt + 1
                            }
                        )
                    else:
                        last_error = f"Execution error: {error}"

                else:
                    last_error = "Could not generate Python code"

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        return VerificationResult(
            statement=statement,
            is_correct=False,
            strategy_used=VerificationStrategy.PYTHON_SYNTHESIS,
            confidence=0.0,
            explanation="Failed to verify statement",
            error_message=last_error,
            metadata={"attempts": self.max_retries, "python_code": python_code or ""}
        )

    def _generate_python_code(
        self,
        statement: str,
        context: Optional[List[str]],
        problem_statement: Optional[str]
    ) -> Optional[str]:
        """
        Generate Python code to verify the statement using LLM.

        Args:
            statement: Statement to verify
            context: Context statements
            problem_statement: Original problem

        Returns:
            Python code as string, or None if generation failed
        """
        prompt = self._create_code_generation_prompt(
            statement, context, problem_statement
        )

        try:
            if self.provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PYTHON_SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
                code = response.choices[0].message.content

            elif self.provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.model,
                    system=PYTHON_SYNTHESIS_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=2048
                )
                code = response.content[0].text

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Extract Python code from response
            return self._extract_python_code(code)

        except Exception as e:
            print(f"Error generating Python code: {e}")
            return None

    def _create_code_generation_prompt(
        self,
        statement: str,
        context: Optional[List[str]],
        problem_statement: Optional[str]
    ) -> str:
        """Create prompt for Python code generation."""
        prompt_parts = []

        if problem_statement:
            prompt_parts.append(f"Problem: {problem_statement}\n")

        if context:
            prompt_parts.append("Previous steps:")
            for i, ctx in enumerate(context, 1):
                prompt_parts.append(f"{i}. {ctx}")
            prompt_parts.append("")

        prompt_parts.append(f"Statement to verify: {statement}\n")
        prompt_parts.append(
            "Generate Python code that evaluates whether this statement is correct. "
            "The code should print 'True' if the statement is correct, 'False' otherwise."
        )

        return "\n".join(prompt_parts)

    def _extract_python_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Try to find code in markdown code blocks
        patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'```python(.*?)```',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = matches[0].strip()
                if code:
                    return code

        # If no code blocks, check if the whole response is valid Python
        try:
            ast.parse(response)
            return response.strip()
        except SyntaxError:
            pass

        return None

    def _execute_python_code(
        self,
        code: str,
        timeout: Optional[int] = None
    ) -> tuple[bool, str, Optional[str]]:
        """
        Execute Python code in a sandboxed environment.

        Args:
            code: Python code to execute
            timeout: Execution timeout (uses self.timeout if None)

        Returns:
            Tuple of (is_correct, output, error_message)
        """
        if timeout is None:
            timeout = self.timeout

        # Create safe execution environment
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'True': True,
                'False': False,
                'None': None,
            },
            'math': math,
        }
        safe_locals = {}

        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            # Compile and execute code
            compiled_code = compile(code, '<string>', 'exec')

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled_code, safe_globals, safe_locals)

            # Get output
            output = stdout_capture.getvalue().strip()
            stderr_output = stderr_capture.getvalue().strip()

            if stderr_output:
                return False, output, stderr_output

            # Parse result - look for True/False in output
            output_lower = output.lower()
            if 'true' in output_lower:
                return True, output, None
            elif 'false' in output_lower:
                return False, output, None
            else:
                # Try to evaluate the last line
                lines = output.split('\n')
                if lines:
                    last_line = lines[-1].strip()
                    if last_line.lower() == 'true':
                        return True, output, None
                    elif last_line.lower() == 'false':
                        return False, output, None

                return False, output, "Could not determine True/False from output"

        except SyntaxError as e:
            return False, "", f"Syntax error: {str(e)}"
        except Exception as e:
            return False, "", f"Execution error: {str(e)}"


# System prompt for Python code generation
PYTHON_SYNTHESIS_SYSTEM_PROMPT = """You are a Python code generator for mathematical and computational verification.

Your task is to generate Python code that verifies whether a given statement is correct.

Requirements:
1. Write simple, executable Python code
2. The code must print exactly 'True' if the statement is correct, 'False' otherwise
3. Use only basic Python operations and the math module
4. Keep the code concise and focused on verification
5. Handle edge cases and potential errors gracefully

Example 1:
Statement: "15 + 7 = 22"
Code:
```python
result = 15 + 7
print(result == 22)
```

Example 2:
Statement: "The product of 12 and 5 is 60"
Code:
```python
product = 12 * 5
print(product == 60)
```

Example 3:
Statement: "25% of 80 is 20"
Code:
```python
result = 0.25 * 80
print(result == 20)
```

Always wrap your code in ```python ``` code blocks.
"""
