# Agentic Verifier - Quick Start Guide

## Overview

The Agentic Verifier is an intelligent verification system that automatically routes statements to the most appropriate verification strategy based on their content and type.

### Key Features

- **Automatic Classification**: Analyzes statements to determine their type (computational, logical/mathematical, or general)
- **Multi-Strategy Verification**: Three verification backends for different statement types
- **Fallback Support**: Automatically falls back to alternative strategies if primary method fails
- **Easy Integration**: Simple API that works with existing VeriForm components

## Installation

The agentic verifier is now part of VeriForm. No additional installation required beyond the standard VeriForm dependencies.

## Quick Start

### 1. Basic Verification

```python
import os
from openai import OpenAI
from veriform import AgenticVerifier

# Initialize
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
verifier = AgenticVerifier(
    llm_client=client,
    model="gpt-4",
    provider="openai"
)

# Verify a statement
result = verifier.verify("15 + 7 = 22")

print(f"Is correct: {result.is_correct}")
print(f"Strategy used: {result.strategy_used.value}")
print(f"Confidence: {result.confidence}")
```

### 2. With Context

```python
result = verifier.verify(
    statement="Therefore, the total is $50",
    context=[
        "Item A costs $30",
        "Item B costs $20"
    ],
    problem_statement="Calculate the total cost"
)
```

### 3. Verify Reasoning Chain

```python
from veriform.data_collection import ReasoningChain, ReasoningStep, StepType

chain = ReasoningChain(
    chain_id="example",
    problem_statement="Calculate 5 * 6 + 10",
    steps=[
        ReasoningStep(
            step_id="step_1",
            content="5 * 6 = 30",
            step_type=StepType.CALCULATION
        ),
        ReasoningStep(
            step_id="step_2",
            content="30 + 10 = 40",
            step_type=StepType.CALCULATION
        )
    ]
)

results = verifier.verify_reasoning_chain(chain)

for step, result in zip(chain.steps, results):
    status = "CORRECT" if result.is_correct else "INCORRECT"
    print(f"{step.content} -> {status}")
```

## Verification Strategies

### 1. Python Synthesis (Computational)

**For:** Arithmetic, numerical calculations, concrete computations

**Examples:**
- "15 + 7 = 22"
- "The product of 12 and 5 is 60"
- "25% of 80 is 20"

**How it works:**
1. LLM generates Python code to evaluate the statement
2. Code is executed in a sandboxed environment
3. Returns True/False based on execution result

### 2. Autoformalization + Lean (Logical/Mathematical)

**For:** Abstract mathematical reasoning, theorems, formal logic

**Examples:**
- "For all n, if n is even then n is divisible by 2"
- "Let f be a continuous function on [0, 1]"
- "Theorem: The set of prime numbers is infinite"

**How it works:**
1. Autoformalizes statement to Lean 4 code
2. Verifies using Lean theorem prover
3. Returns verification result

**Note:** Requires Lean 4 installation and properly configured autoformalization.

### 3. LLM Judge (General)

**For:** General reasoning statements, logical implications, conclusions

**Examples:**
- "Therefore, we can conclude..."
- "This follows from the previous steps"
- "Based on the evidence, the answer is..."

**How it works:**
1. Sends statement + context to LLM
2. LLM evaluates logical soundness
3. Returns judgment with confidence and explanation

## Configuration

### Using Config Object

```python
from veriform import AgenticVerifierConfig

config = AgenticVerifierConfig(
    model="gpt-4",
    provider="openai",
    temperature=0.0,
    enable_python_verifier=True,
    enable_autoformalization=True,
    enable_llm_judge=True,
    fallback_to_llm_judge=True,
    verbose=True  # Print routing decisions
)
```

### Selective Strategy Enabling

```python
# Only computational verification
verifier = AgenticVerifier(
    llm_client=client,
    model="gpt-4",
    provider="openai",
    enable_python_verifier=True,
    enable_autoformalization=False,
    enable_llm_judge=False
)

# Only LLM judge
verifier = AgenticVerifier(
    llm_client=client,
    model="gpt-4",
    provider="openai",
    enable_python_verifier=False,
    enable_autoformalization=False,
    enable_llm_judge=True
)
```

## Working with Results

### VerificationResult Structure

```python
result = verifier.verify("Some statement")

# Core fields
result.statement          # The statement that was verified
result.is_correct         # True/False
result.strategy_used      # VerificationStrategy enum
result.confidence         # 0.0 to 1.0
result.explanation        # Human-readable explanation

# Metadata
result.metadata           # Dict with additional info
result.error_message      # Error if verification failed
```

### Getting Statistics

```python
results = [verifier.verify(stmt) for stmt in statements]
stats = verifier.get_routing_stats(results)

print(f"Total: {stats['total']}")
print(f"Accuracy: {stats['accuracy']:.2%}")
print(f"Avg confidence: {stats['average_confidence']:.2f}")

# By strategy
for strategy, count in stats['by_strategy'].items():
    print(f"{strategy}: {count}")
```

## Examples

### Example 1: Simple Computational Verification

```python
from openai import OpenAI
from veriform import AgenticVerifier

client = OpenAI(api_key="your-key")
verifier = AgenticVerifier(
    llm_client=client,
    model="gpt-4",
    provider="openai",
    verbose=True  # See routing decisions
)

# Test statements
statements = [
    "15 + 7 = 22",      # Correct
    "12 * 5 = 60",      # Correct
    "100 - 35 = 55"     # Incorrect (should be 65)
]

for stmt in statements:
    result = verifier.verify(stmt)
    print(f"{stmt} -> {'✓' if result.is_correct else '✗'}")
```

### Example 2: With Anthropic Claude

```python
from anthropic import Anthropic
from veriform import AgenticVerifier

client = Anthropic(api_key="your-key")
verifier = AgenticVerifier(
    llm_client=client,
    model="claude-3-opus-20240229",
    provider="anthropic"
)

result = verifier.verify("sqrt(144) = 12")
print(f"Correct: {result.is_correct}")
```

### Example 3: Batch Verification

```python
statements = [
    "5 * 6 = 30",
    "30 + 10 = 40",
    "Therefore, 5 * 6 + 10 = 40"
]

results = []
for i, stmt in enumerate(statements):
    context = statements[:i]  # Use previous statements as context
    result = verifier.verify(stmt, context=context)
    results.append(result)

    print(f"Step {i+1}: {stmt}")
    print(f"  Status: {'CORRECT' if result.is_correct else 'INCORRECT'}")
    print(f"  Strategy: {result.strategy_used.value}")
    print()

# Overall statistics
stats = verifier.get_routing_stats(results)
print(f"Overall accuracy: {stats['accuracy']:.1%}")
```

## Running the Demo

A comprehensive demo is included that shows all features:

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"

# Run the demo
python examples/agentic_verifier_demo.py
```

The demo includes:
1. Statement classification examples
2. Python verifier examples
3. LLM judge examples
4. Full agentic routing examples
5. Reasoning chain verification

## Troubleshooting

### Issue: ImportError

**Solution:** Make sure VeriForm is installed or add to PYTHONPATH:
```python
import sys
sys.path.insert(0, 'path/to/VeriForm/src')
```

### Issue: API Key Not Found

**Solution:** Set environment variable:
```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

### Issue: Verification Always Fails

**Solution:**
1. Check API key is valid
2. Enable verbose mode to see what's happening
3. Check error_message in result
4. Enable fallback_to_llm_judge

### Issue: Classification Seems Wrong

**Solution:**
- Classification is heuristic-based
- You can manually specify the strategy by disabling others
- Or use specific verifiers directly (PythonVerifier, LLMJudgeVerifier)

## Advanced Usage

### Custom Classification

```python
from veriform.verification.classifier import StatementClassifier, StatementType

# Check classification before verifying
stmt = "15 + 7 = 22"
stmt_type = StatementClassifier.classify(stmt)
confidence = StatementClassifier.get_classification_confidence(stmt)

print(f"Type: {stmt_type.value}")
print(f"Confidence: {confidence:.2f}")

# Now verify with appropriate strategy
result = verifier.verify(stmt)
```

### Error Handling

```python
result = verifier.verify("Some statement")

if result.error_message:
    print(f"Error: {result.error_message}")

    # Check if fallback was used
    if result.metadata.get('fallback'):
        print("Fallback strategy was used")
        print(f"Original error: {result.metadata.get('original_error')}")
```

### With Autoformalization

```python
from veriform.autoformalization import get_formalizer
from veriform.autoformalization.lean_verifier import LeanVerifier

# Initialize autoformalization components
formalizer = get_formalizer(
    provider="openai",
    model="gpt-4",
    api_key="your-key"
)
lean_verifier = LeanVerifier()

# Create verifier with Lean support
verifier = AgenticVerifier(
    llm_client=client,
    model="gpt-4",
    provider="openai",
    autoformalization=formalizer,
    lean_verifier=lean_verifier,
    enable_autoformalization=True
)

# Now can verify logical/mathematical statements
result = verifier.verify(
    "For all natural numbers n, if n is even then n is divisible by 2"
)
```

## Best Practices

1. **Enable verbose mode during development** to understand routing decisions
2. **Always enable fallback_to_llm_judge** for production use
3. **Provide context when available** for better accuracy
4. **Monitor routing statistics** to understand verification patterns
5. **Handle errors gracefully** - check result.error_message
6. **Use appropriate temperature** - 0.0 for deterministic, higher for reasoning

## Next Steps

- Read the full documentation: `src/veriform/verification/README.md`
- Explore the demo: `examples/agentic_verifier_demo.py`
- Check the tests: `tests/test_agentic_verifier.py`
- Integrate with your benchmarking workflows

## Support

For issues or questions:
1. Check the README in `src/veriform/verification/`
2. Review examples in `examples/agentic_verifier_demo.py`
3. Open an issue on GitHub

---

**Happy Verifying! 🎯**
