# Agentic Verifier Module

An intelligent verification system that automatically routes statements to appropriate verification strategies based on their type and content.

## Overview

The Agentic Verifier module provides three verification strategies:

1. **Python Synthesis** - For computational/arithmetic statements
2. **Autoformalization + Lean** - For abstract logical/mathematical reasoning
3. **LLM Judge** - For general reasoning statements

The module automatically classifies statements and routes them to the most appropriate verifier, with fallback support for robustness.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AgenticVerifier                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │         StatementClassifier                        │     │
│  │  Analyzes statement → determines type              │     │
│  └────────────────────────────────────────────────────┘     │
│                          │                                   │
│                          ▼                                   │
│         ┌────────────────┬───────────────┬────────────┐     │
│         │                │               │            │     │
│    COMPUTATIONAL   LOGICAL_MATH      GENERAL          │     │
│         │                │               │            │     │
│         ▼                ▼               ▼            │     │
│  ┌──────────┐    ┌──────────────┐  ┌──────────┐     │     │
│  │  Python  │    │ Autoform +   │  │   LLM    │     │     │
│  │  Synth   │    │   Lean       │  │  Judge   │     │     │
│  └──────────┘    └──────────────┘  └──────────┘     │     │
│         │                │               │            │     │
│         └────────────────┴───────────────┘            │     │
│                          │                            │     │
│                          ▼                            │     │
│                 VerificationResult                    │     │
│          (is_correct, confidence, explanation)        │     │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. StatementClassifier

Classifies statements into three categories:

- **COMPUTATIONAL**: Arithmetic, numerical calculations
  - Examples: "15 + 7 = 22", "The product of 12 and 5 is 60"
  - Pattern matching: arithmetic operators, numbers, calculation keywords

- **LOGICAL_MATHEMATICAL**: Abstract math, theorems, formal logic
  - Examples: "For all n, if n is even then n is divisible by 2"
  - Pattern matching: mathematical terms, logical connectives, formal language

- **GENERAL**: Other reasoning statements
  - Examples: "Therefore, we can conclude...", "This follows from..."
  - Catches everything else that doesn't fit specific categories

### 2. PythonVerifier

Verifies computational statements by:
1. Using LLM to generate Python code that evaluates the statement
2. Executing code in a sandboxed environment
3. Returning True/False based on execution result

**Example:**
```python
Statement: "15 + 7 = 22"

Generated code:
result = 15 + 7
print(result == 22)

Output: True
→ Statement is CORRECT
```

### 3. LLMJudgeVerifier

Verifies general statements by:
1. Sending statement + context to LLM
2. LLM evaluates logical soundness
3. Returns judgment with confidence and explanation

**Example:**
```python
Statement: "Therefore, the total is $50"
Context: ["Item A costs $30", "Item B costs $20"]

LLM Output:
VERDICT: CORRECT
CONFIDENCE: 0.95
EXPLANATION: The arithmetic is correct: $30 + $20 = $50
```

### 4. AgenticVerifier

The main orchestrator that:
1. Classifies incoming statements
2. Routes to appropriate verifier
3. Handles fallback if primary method fails
4. Returns unified VerificationResult

## Usage

### Basic Usage

```python
from openai import OpenAI
from veriform.verification import AgenticVerifier

# Initialize LLM client
client = OpenAI(api_key="your-api-key")

# Create agentic verifier
verifier = AgenticVerifier(
    llm_client=client,
    model="gpt-4",
    provider="openai",
    verbose=True  # Print routing decisions
)

# Verify a statement
result = verifier.verify(
    statement="15 + 7 = 22",
    context=None,
    problem_statement="Verify the calculation"
)

print(f"Correct: {result.is_correct}")
print(f"Strategy: {result.strategy_used}")
print(f"Confidence: {result.confidence}")
print(f"Explanation: {result.explanation}")
```

### With Context

```python
result = verifier.verify(
    statement="Therefore, the total cost is $50",
    context=[
        "Item A costs $30",
        "Item B costs $20"
    ],
    problem_statement="Calculate the total cost"
)
```

### Verifying Reasoning Chains

```python
from veriform.data_collection import ReasoningChain, ReasoningStep, StepType

# Create a reasoning chain
chain = ReasoningChain(
    chain_id="chain_1",
    problem_statement="Calculate 5 * 6 + 10",
    steps=[
        ReasoningStep(
            step_id="step_1",
            content="First, multiply: 5 * 6 = 30",
            step_type=StepType.CALCULATION
        ),
        ReasoningStep(
            step_id="step_2",
            content="Then add: 30 + 10 = 40",
            step_type=StepType.CALCULATION
        )
    ]
)

# Verify all steps
results = verifier.verify_reasoning_chain(chain)

for step, result in zip(chain.steps, results):
    print(f"{step.content} → {'✓' if result.is_correct else '✗'}")
```

### Configuration

```python
from veriform.config import AgenticVerifierConfig

config = AgenticVerifierConfig(
    model="gpt-4",
    provider="openai",
    temperature=0.0,
    enable_python_verifier=True,
    enable_autoformalization=True,
    enable_llm_judge=True,
    fallback_to_llm_judge=True,
    verbose=True
)

verifier = AgenticVerifier(
    llm_client=client,
    model=config.model,
    provider=config.provider,
    temperature=config.temperature,
    enable_python_verifier=config.enable_python_verifier,
    enable_autoformalization=config.enable_autoformalization,
    enable_llm_judge=config.enable_llm_judge,
    fallback_to_llm_judge=config.fallback_to_llm_judge,
    verbose=config.verbose
)
```

### Selective Strategy Enabling

```python
# Only Python verification (for computational statements)
verifier = AgenticVerifier(
    llm_client=client,
    model="gpt-4",
    provider="openai",
    enable_python_verifier=True,
    enable_autoformalization=False,
    enable_llm_judge=False
)

# Only LLM judge (for general reasoning)
verifier = AgenticVerifier(
    llm_client=client,
    model="gpt-4",
    provider="openai",
    enable_python_verifier=False,
    enable_autoformalization=False,
    enable_llm_judge=True
)
```

### With Autoformalization (Requires Lean)

```python
from veriform.autoformalization import get_formalizer
from veriform.autoformalization.lean_verifier import LeanVerifier

# Initialize autoformalization
formalizer = get_formalizer(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)

lean_verifier = LeanVerifier()

# Create verifier with autoformalization enabled
verifier = AgenticVerifier(
    llm_client=client,
    model="gpt-4",
    provider="openai",
    autoformalization=formalizer,
    lean_verifier=lean_verifier,
    enable_autoformalization=True
)
```

## API Reference

### VerificationResult

```python
@dataclass
class VerificationResult:
    statement: str              # The statement that was verified
    is_correct: bool            # True if statement is correct
    strategy_used: VerificationStrategy  # Which verifier was used
    confidence: float           # Confidence score (0.0 to 1.0)
    explanation: str            # Human-readable explanation
    metadata: Dict[str, Any]    # Additional metadata
    error_message: Optional[str] # Error if verification failed
```

### VerificationStrategy

```python
class VerificationStrategy(Enum):
    PYTHON_SYNTHESIS = "python_synthesis"
    AUTOFORMALIZATION = "autoformalization"
    LLM_JUDGE = "llm_judge"
```

### StatementType

```python
class StatementType(Enum):
    COMPUTATIONAL = "computational"
    LOGICAL_MATHEMATICAL = "logical_mathematical"
    GENERAL = "general"
```

## Statistics and Monitoring

```python
# Verify multiple statements
results = [verifier.verify(stmt) for stmt in statements]

# Get routing statistics
stats = verifier.get_routing_stats(results)

print(f"Total: {stats['total']}")
print(f"Accuracy: {stats['accuracy']:.2%}")
print(f"Average confidence: {stats['average_confidence']:.2f}")

# By strategy
for strategy, count in stats['by_strategy'].items():
    print(f"{strategy}: {count}")

# By statement type
for stmt_type, count in stats['by_statement_type'].items():
    print(f"{stmt_type}: {count}")
```

## Examples

See `examples/agentic_verifier_demo.py` for comprehensive examples including:

1. Statement classification demos
2. Python verifier demos
3. LLM judge demos
4. Full agentic routing demos
5. Reasoning chain verification

Run the demo:
```bash
export OPENAI_API_KEY="your-key"
python examples/agentic_verifier_demo.py
```

## Error Handling

The module includes robust error handling:

1. **Retry logic**: Automatically retries on transient failures
2. **Fallback**: Falls back to LLM judge if primary method fails
3. **Graceful degradation**: Returns error result rather than crashing

```python
result = verifier.verify("Some statement")

if result.error_message:
    print(f"Verification failed: {result.error_message}")
    print(f"Fallback used: {result.metadata.get('fallback', False)}")
```

## Best Practices

1. **Enable verbose mode during development** to see routing decisions
2. **Use fallback to LLM judge** for robustness in production
3. **Monitor routing statistics** to understand verification patterns
4. **Adjust confidence thresholds** based on your application needs
5. **Provide context** when available for better verification accuracy

## Limitations

1. **Python verifier**: Sandboxed environment has limited libraries
2. **Autoformalization**: Requires Lean 4 installation
3. **LLM judge**: Subject to LLM limitations and biases
4. **Classification**: May misclassify edge cases or ambiguous statements

## Future Enhancements

- [ ] Add more sophisticated classification (ML-based)
- [ ] Support more programming languages (JavaScript, etc.)
- [ ] Add verification result caching
- [ ] Support batch verification for efficiency
- [ ] Add ensemble verification (combine multiple strategies)
- [ ] Fine-tuned models for specific domains
