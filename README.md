# Veriform: Verification-Oriented Formalization Benchmarking

A framework for measuring the faithfulness of autoformalization systems in correctly handling incorrect reasoning steps from LLM Chain-of-Thoughts.

## Motivation

Current autoformalization systems exhibit **sycophancy** - they attempt to generate correct Lean proofs even when the underlying reasoning is incorrect. This defeats the purpose of formal verification, where we want the system to:
- Generate correct Lean proofs for correct reasoning steps
- Generate **incorrect** or unprovable Lean statements for incorrect reasoning steps

## Methodology

Veriform provides a benchmark generation framework that:

1. **Collects** correct reasoning steps from existing datasets
2. **Perturbs** steps with probability `p` to introduce errors (e.g., changing operators, swapping values)
3. **Autoformalizes** each (potentially incorrect) step using Lean, with previous steps as `sorry` lemmas
4. **Measures** correlation between perturbation rate `p` and verification failures

The key insight: With large enough samples, the correlation between `p` and error rate measures how faithfully the autoformalization system preserves incorrectness.

## Project Structure

```
veriform/
├── src/veriform/           # Main package
│   ├── data_collection/    # Dataset loading and management
│   ├── perturbation/       # Reasoning step perturbation strategies
│   ├── autoformalization/  # Interface to autoformalization systems
│   ├── benchmark/          # Benchmark execution and coordination
│   ├── analysis/           # Statistical analysis and visualization
│   └── utils/              # Shared utilities
├── tests/                  # Unit and integration tests
├── data/                   # Data storage
├── experiments/            # Experiment scripts and results
├── configs/                # Configuration files
└── docs/                   # Additional documentation
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from veriform.benchmark import BenchmarkRunner
from veriform.config import BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    perturbation_probabilities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    sample_size=1000,
    perturbation_strategies=["operator_swap", "value_change"],
    autoformalization_model="gpt-4"
)

# Run benchmark
runner = BenchmarkRunner(config)
results = runner.run()

# Analyze results
from veriform.analysis import analyze_faithfulness
analyze_faithfulness(results, output_dir="./results")
```

## Features

- **Multiple Perturbation Strategies**: Operator swaps, value changes, logical negations, etc.
- **Flexible Autoformalization Interface**: Support for various LLM backends
- **Statistical Analysis**: Correlation metrics, visualization, significance testing
- **Extensible Design**: Easy to add new perturbation strategies or data sources

## Citation

```bibtex
@misc{veriform2024,
  title={Veriform: Benchmarking Faithfulness in Autoformalization},
  author={Rob and collaborators},
  year={2024}
}
```

## License

MIT License
