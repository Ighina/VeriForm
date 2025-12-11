"""
Demo script for the AgenticVerifier module.

This script demonstrates how to use the agentic verifier to verify
different types of statements using appropriate strategies.
"""

import os
from openai import OpenAI
from anthropic import Anthropic

from veriform.verification import AgenticVerifier
from veriform.verification.classifier import StatementClassifier, StatementType
from veriform.config import AgenticVerifierConfig


def demo_statement_classification():
    """Demonstrate statement classification."""
    print("=" * 80)
    print("DEMO 1: Statement Classification")
    print("=" * 80)

    test_statements = [
        "15 + 7 = 22",
        "The product of 12 and 5 is 60",
        "25% of 80 is 20",
        "For all natural numbers n, if n is divisible by 4, then n is even",
        "Let f be a continuous function on [0, 1]. Then f is bounded.",
        "This conclusion follows from the previous steps because we applied the rule correctly",
        "sqrt(144) = 12",
        "If P implies Q and Q implies R, then P implies R",
    ]

    for statement in test_statements:
        stmt_type = StatementClassifier.classify(statement)
        confidence = StatementClassifier.get_classification_confidence(statement)
        print(f"\nStatement: {statement}")
        print(f"Type: {stmt_type.value}")
        print(f"Confidence: {confidence:.2f}")


def demo_python_verifier():
    """Demonstrate Python verifier for computational statements."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Python Verifier (Computational Statements)")
    print("=" * 80)

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return

    client = OpenAI(api_key=api_key)

    # Create verifier (only Python verifier enabled)
    verifier = AgenticVerifier(
        llm_client=client,
        model="gpt-4",
        provider="openai",
        enable_python_verifier=True,
        enable_autoformalization=False,
        enable_llm_judge=False,
        verbose=True
    )

    # Test computational statements
    test_cases = [
        ("15 + 7 = 22", True),  # Correct
        ("12 * 5 = 60", True),  # Correct
        ("25% of 80 is 30", False),  # Incorrect (should be 20)
        ("100 - 35 = 55", False),  # Incorrect (should be 65)
        ("sqrt(144) = 12", True),  # Correct
    ]

    print("\nVerifying computational statements:\n")

    for statement, expected in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Statement: {statement}")
        print(f"Expected: {'CORRECT' if expected else 'INCORRECT'}")

        result = verifier.verify(statement)

        print(f"Result: {'CORRECT' if result.is_correct else 'INCORRECT'}")
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Explanation: {result.explanation}")

        if result.metadata.get("python_code"):
            print(f"\nGenerated Python code:")
            print(result.metadata["python_code"])


def demo_llm_judge():
    """Demonstrate LLM judge for general statements."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: LLM Judge (General Statements)")
    print("=" * 80)

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return

    client = OpenAI(api_key=api_key)

    # Create verifier (only LLM judge enabled)
    verifier = AgenticVerifier(
        llm_client=client,
        model="gpt-4",
        provider="openai",
        enable_python_verifier=False,
        enable_autoformalization=False,
        enable_llm_judge=True,
        verbose=True
    )

    # Test general reasoning statements
    test_cases = [
        {
            "statement": "Therefore, the total cost is $50",
            "context": ["Item A costs $30", "Item B costs $20"],
            "problem": "Calculate the total cost of items A and B"
        },
        {
            "statement": "Since all birds can fly, penguins can fly",
            "context": ["Penguins are birds"],
            "problem": "Determine if penguins can fly"
        },
        {
            "statement": "This means we need to add 40 more",
            "context": ["The target sum is 100", "We have added 60 so far"],
            "problem": "How much more do we need to add?"
        },
    ]

    print("\nVerifying general reasoning statements:\n")

    for test in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Problem: {test['problem']}")
        print(f"Context: {test['context']}")
        print(f"Statement: {test['statement']}")

        result = verifier.verify(
            statement=test["statement"],
            context=test["context"],
            problem_statement=test["problem"]
        )

        print(f"\nResult: {'CORRECT' if result.is_correct else 'INCORRECT'}")
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Explanation: {result.explanation[:200]}...")


def demo_full_agentic_verifier():
    """Demonstrate full agentic verifier with all strategies."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Full Agentic Verifier (All Strategies)")
    print("=" * 80)

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return

    client = OpenAI(api_key=api_key)

    # Create full agentic verifier
    # Note: Autoformalization requires Lean setup, so we disable it for this demo
    verifier = AgenticVerifier(
        llm_client=client,
        model="gpt-4",
        provider="openai",
        enable_python_verifier=True,
        enable_autoformalization=False,  # Disabled (requires Lean)
        enable_llm_judge=True,
        fallback_to_llm_judge=True,
        verbose=True
    )

    # Mixed statement types
    test_statements = [
        "15 + 7 = 22",  # Computational
        "The result follows from the previous equation",  # General
        "50 * 2 = 100",  # Computational
        "Therefore, we can conclude that the answer is correct",  # General
        "sqrt(64) = 8",  # Computational
    ]

    print("\nVerifying mixed statement types:\n")
    results = []

    for statement in test_statements:
        print(f"\n{'─' * 80}")
        result = verifier.verify(statement)
        results.append(result)

    # Get routing statistics
    print("\n\n" + "=" * 80)
    print("ROUTING STATISTICS")
    print("=" * 80)

    stats = verifier.get_routing_stats(results)
    print(f"\nTotal statements: {stats['total']}")
    print(f"Correct: {stats['correct']}")
    print(f"Incorrect: {stats['incorrect']}")
    print(f"Accuracy: {stats['accuracy']:.2%}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")

    print("\nBy strategy:")
    for strategy, count in stats['by_strategy'].items():
        print(f"  {strategy}: {count}")

    print("\nBy statement type:")
    for stmt_type, count in stats['by_statement_type'].items():
        print(f"  {stmt_type}: {count}")


def demo_with_reasoning_chain():
    """Demonstrate verification with a complete reasoning chain."""
    print("\n\n" + "=" * 80)
    print("DEMO 5: Verifying a Reasoning Chain")
    print("=" * 80)

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return

    client = OpenAI(api_key=api_key)

    # Create verifier
    verifier = AgenticVerifier(
        llm_client=client,
        model="gpt-4",
        provider="openai",
        enable_python_verifier=True,
        enable_autoformalization=False,
        enable_llm_judge=True,
        verbose=True
    )

    # Create a reasoning chain
    from veriform.data_collection import ReasoningChain, ReasoningStep, StepType

    chain = ReasoningChain(
        chain_id="demo_chain_1",
        problem_statement="A store sells apples for $2 each and oranges for $3 each. If John buys 5 apples and 3 oranges, how much does he spend?",
        steps=[
            ReasoningStep(
                step_id="step_1",
                content="Cost of apples: 5 * $2 = $10",
                step_type=StepType.CALCULATION,
                is_perturbed=False
            ),
            ReasoningStep(
                step_id="step_2",
                content="Cost of oranges: 3 * $3 = $9",
                step_type=StepType.CALCULATION,
                is_perturbed=False
            ),
            ReasoningStep(
                step_id="step_3",
                content="Total cost: $10 + $9 = $19",
                step_type=StepType.CALCULATION,
                is_perturbed=False
            ),
        ],
        final_answer="$19"
    )

    print(f"\nProblem: {chain.problem_statement}\n")

    # Verify the entire chain
    results = verifier.verify_reasoning_chain(chain)

    # Print results
    print("\n" + "=" * 80)
    print("CHAIN VERIFICATION RESULTS")
    print("=" * 80)

    for i, (step, result) in enumerate(zip(chain.steps, results), 1):
        print(f"\nStep {i}: {step.content}")
        print(f"  Result: {'✓ CORRECT' if result.is_correct else '✗ INCORRECT'}")
        print(f"  Strategy: {result.strategy_used.value}")
        print(f"  Confidence: {result.confidence:.2f}")

    # Overall statistics
    correct = sum(1 for r in results if r.is_correct)
    total = len(results)
    print(f"\nOverall: {correct}/{total} steps correct ({correct/total:.1%})")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "AGENTIC VERIFIER DEMO" + " " * 37 + "║")
    print("╚" + "═" * 78 + "╝")

    # Demo 1: Classification
    demo_statement_classification()

    # Check for API key before running LLM demos
    if not os.getenv("OPENAI_API_KEY"):
        print("\n\nNote: Set OPENAI_API_KEY environment variable to run LLM-based demos")
        return

    # Demo 2: Python verifier
    try:
        demo_python_verifier()
    except Exception as e:
        print(f"\nERROR in Python verifier demo: {e}")

    # Demo 3: LLM judge
    try:
        demo_llm_judge()
    except Exception as e:
        print(f"\nERROR in LLM judge demo: {e}")

    # Demo 4: Full agentic verifier
    try:
        demo_full_agentic_verifier()
    except Exception as e:
        print(f"\nERROR in full agentic verifier demo: {e}")

    # Demo 5: Reasoning chain
    try:
        demo_with_reasoning_chain()
    except Exception as e:
        print(f"\nERROR in reasoning chain demo: {e}")

    print("\n\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
