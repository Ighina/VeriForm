"""
Simple standalone example of the Agentic Verifier.

This script demonstrates the agentic verifier with minimal setup.
Run with: python examples/simple_agentic_verifier_example.py
"""

import os
import sys

# Add src to path if running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from openai import OpenAI
from veriform.verification import AgenticVerifier


def main():
    """Run simple agentic verifier examples."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return

    print("=" * 80)
    print("SIMPLE AGENTIC VERIFIER EXAMPLE")
    print("=" * 80)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Create agentic verifier with verbose output
    print("\n[1] Initializing AgenticVerifier...")
    verifier = AgenticVerifier(
        llm_client=client,
        model="gpt-4",
        provider="openai",
        enable_python_verifier=True,
        enable_autoformalization=False,  # Requires Lean setup
        enable_llm_judge=True,
        fallback_to_llm_judge=True,
        verbose=True  # Shows routing decisions
    )
    print("    ✓ Verifier initialized")

    # Example 1: Computational verification
    print("\n" + "=" * 80)
    print("[2] COMPUTATIONAL VERIFICATION (Python Synthesis)")
    print("=" * 80)

    computational_statements = [
        ("15 + 7 = 22", True),
        ("12 * 5 = 60", True),
        ("100 - 35 = 55", False),  # Wrong, should be 65
        ("sqrt(144) = 12", True),
    ]

    print("\nVerifying computational statements:\n")
    for stmt, expected in computational_statements:
        print(f"\n{'─' * 80}")
        print(f"Statement: {stmt}")
        print(f"Expected: {'CORRECT' if expected else 'INCORRECT'}")

        result = verifier.verify(stmt)

        match = "✓ MATCH" if result.is_correct == expected else "✗ MISMATCH"
        print(f"Got: {'CORRECT' if result.is_correct else 'INCORRECT'} {match}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Strategy: {result.strategy_used.value}")

    # Example 2: General reasoning with context
    print("\n\n" + "=" * 80)
    print("[3] GENERAL REASONING (LLM Judge)")
    print("=" * 80)

    print("\nProblem: Calculate total cost of shopping")
    print("Context:")
    print("  1. Item A costs $30")
    print("  2. Item B costs $20")
    print("\nStatement to verify: 'Therefore, the total cost is $50'")

    result = verifier.verify(
        statement="Therefore, the total cost is $50",
        context=["Item A costs $30", "Item B costs $20"],
        problem_statement="Calculate the total cost"
    )

    print(f"\nResult: {'CORRECT' if result.is_correct else 'INCORRECT'}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Strategy: {result.strategy_used.value}")
    print(f"Explanation: {result.explanation[:150]}...")

    # Example 3: Verifying a reasoning chain
    print("\n\n" + "=" * 80)
    print("[4] REASONING CHAIN VERIFICATION")
    print("=" * 80)

    from veriform.data_collection import ReasoningChain, ReasoningStep, StepType

    chain = ReasoningChain(
        chain_id="example_1",
        problem_statement="John has 5 apples. He buys 3 more. How many does he have?",
        steps=[
            ReasoningStep(
                step_id="step_1",
                content="John starts with 5 apples",
                step_type=StepType.OTHER,
                is_perturbed=False
            ),
            ReasoningStep(
                step_id="step_2",
                content="He buys 3 more apples",
                step_type=StepType.OTHER,
                is_perturbed=False
            ),
            ReasoningStep(
                step_id="step_3",
                content="Total apples: 5 + 3 = 8",
                step_type=StepType.CALCULATION,
                is_perturbed=False
            ),
        ],
        final_answer="8 apples"
    )

    print(f"\nProblem: {chain.problem_statement}")
    print("\nVerifying each step:\n")

    results = verifier.verify_reasoning_chain(chain)

    for i, (step, result) in enumerate(zip(chain.steps, results), 1):
        status = "✓ CORRECT" if result.is_correct else "✗ INCORRECT"
        print(f"Step {i}: {step.content}")
        print(f"  → {status} (confidence: {result.confidence:.2f}, strategy: {result.strategy_used.value})")

    # Statistics
    print("\n\n" + "=" * 80)
    print("[5] VERIFICATION STATISTICS")
    print("=" * 80)

    stats = verifier.get_routing_stats(results)
    print(f"\nTotal steps verified: {stats['total']}")
    print(f"Correct: {stats['correct']}")
    print(f"Incorrect: {stats['incorrect']}")
    print(f"Accuracy: {stats['accuracy']:.1%}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")

    print("\nVerification strategies used:")
    for strategy, count in stats['by_strategy'].items():
        print(f"  • {strategy}: {count}")

    print("\nStatement types:")
    for stmt_type, count in stats['by_statement_type'].items():
        print(f"  • {stmt_type}: {count}")

    # Summary
    print("\n\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\n✓ The agentic verifier successfully:")
    print("  • Classified statements by type")
    print("  • Routed to appropriate verification strategies")
    print("  • Verified computational statements using Python synthesis")
    print("  • Verified general statements using LLM judge")
    print("  • Verified complete reasoning chains")
    print("\nNext steps:")
    print("  • See examples/agentic_verifier_demo.py for more examples")
    print("  • Read AGENTIC_VERIFIER_QUICKSTART.md for detailed guide")
    print("  • Check src/veriform/verification/README.md for full docs")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
