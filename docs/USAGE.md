# Veriform Usage Guide

This guide provides detailed instructions for using Veriform to benchmark autoformalization systems.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Running Benchmarks](#running-benchmarks)
5. [Analyzing Results](#analyzing-results)
6. [Advanced Usage](#advanced-usage)

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/veriform.git
cd veriform

# Install in development mode
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- Lean 4 (optional, for actual verification)
- API keys for OpenAI or Anthropic (for real autoformalization)

## Quick Start

### 1. Run the Quickstart Example

The quickstart example uses a mock formalizer (no API keys needed):

```bash
cd experiments
python quickstart.py
```

This will:
- Load a small sample from GSM8K
- Apply perturbations at different rates
- Mock-formalize the steps
- Analyze and visualize results

### 2. Run with Real Data

Set up your API key:

```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

Run the benchmark:

```bash
python run_benchmark.py --config ../configs/default.yaml
```

## Configuration

### Configuration File Structure

Veriform uses YAML or JSON configuration files. Here's a complete example:

```yaml
# Perturbation settings
perturbation_probabilities:
  - 0.0    # No perturbations (baseline)
  - 0.2
  - 0.4
  - 0.6
  - 0.8
  - 1.0    # All steps perturbed

perturbation_strategies:
  - operator_swap      # Swap +, -, *, /
  - value_change       # Change numerical values
  - logical_negation   # Negate logical statements
  - sign_flip          # Flip signs of numbers
  - word_swap          # Swap similar words

# Dataset settings
dataset_name: gsm8k   # Options: gsm8k, math, custom
sample_size: 1000     # Number of chains to process

# Autoformalization settings
autoformalization_model: gpt-4
autoformalization_temperature: 0.0
max_retries: 3

# Lean verification settings
lean_timeout: 30
use_sorry_context: true

# Output settings
output_dir: ./experiments/outputs
save_intermediate: true

# Reproducibility
random_seed: 42
```

### Custom Datasets

To use your own dataset:

```python
from veriform.data_collection import CustomLoader, ReasoningChain

# Prepare your data
data = [
    {
        "problem": "What is 2 + 2?",
        "steps": [
            "Start with 2",
            "Add 2 more",
            "The result is 4"
        ],
        "answer": "4"
    },
    # ... more examples
]

# Create loader
loader = CustomLoader(data=data, num_samples=100)
chains = loader.load()
```

## Running Benchmarks

### Using the Command Line Script

```bash
python run_benchmark.py \
    --config path/to/config.yaml \
    --output-dir ./my_results
```

### Using Python API

```python
from veriform import BenchmarkRunner, BenchmarkConfig

# Create configuration
config = BenchmarkConfig(
    perturbation_probabilities=[0.0, 0.5, 1.0],
    dataset_name="gsm8k",
    sample_size=100,
    autoformalization_model="gpt-4",
    output_dir="./results"
)

# Run benchmark
runner = BenchmarkRunner(config)
results = runner.run()
```

### Monitoring Progress

The benchmark runner displays progress bars and logs:

```
Loading dataset: gsm8k
Loaded 100 reasoning chains

Running benchmark for probability: 0.0
Processing (p=0.0): 100%|████████████| 100/100 [02:15<00:00,  1.35s/it]
Saved run results to ./results/run_p0.00.json

Running benchmark for probability: 0.5
Processing (p=0.5): 100%|████████████| 100/100 [02:18<00:00,  1.38s/it]
...
```

## Analyzing Results

### Automatic Analysis

The benchmark runner automatically generates:
- `results.json`: Complete results data
- `summary.txt`: Text summary of key metrics
- `error_rate_vs_perturbation.png`: Main correlation plot
- `correlation_heatmap.png`: Correlation matrix
- `per_run_statistics.png`: Detailed statistics

### Manual Analysis

```python
from veriform.analysis import (
    analyze_faithfulness,
    plot_error_rate_vs_perturbation,
    create_summary_report
)

# Load results
with open("results.json") as f:
    results = json.load(f)

# Analyze faithfulness
analysis = analyze_faithfulness(results)

print(f"Faithfulness Score: {analysis['faithfulness_score']:.3f}")
print(f"Pearson r: {analysis['verification_correlation']['pearson_r']:.3f}")

# Create plots
plot_error_rate_vs_perturbation(results, show=True)

# Generate full report
create_summary_report(results, output_dir="./report")
```

### Interpreting Results

**Faithfulness Score**: Ranges from 0 to 1, where:
- **> 0.8**: Excellent faithfulness
- **0.6-0.8**: Good faithfulness
- **0.4-0.6**: Moderate faithfulness
- **< 0.4**: Poor faithfulness

**What it means**: A high faithfulness score indicates that the autoformalization system correctly produces more errors when reasoning steps are incorrect, rather than trying to "fix" them (sycophancy).

## Advanced Usage

### Custom Perturbation Strategies

Create your own perturbation strategy:

```python
from veriform.perturbation import PerturbationStrategy

class MyCustomStrategy(PerturbationStrategy):
    def can_apply(self, step):
        # Check if strategy can be applied
        return "keyword" in step.content

    def apply(self, step):
        # Apply your perturbation
        new_content = step.content.replace("keyword", "other_word")
        return self._create_perturbed_step(
            step,
            new_content,
            "my_custom_strategy"
        )

# Register and use
from veriform.perturbation import STRATEGY_REGISTRY
STRATEGY_REGISTRY["my_custom"] = MyCustomStrategy
```

### Using Different LLM Providers

```python
# OpenAI
config = BenchmarkConfig(
    autoformalization_model="gpt-4",
    openai_api_key="your-key"
)

# Anthropic
config = BenchmarkConfig(
    autoformalization_model="claude-3-opus-20240229",
    anthropic_api_key="your-key"
)

# Mock (for testing)
config = BenchmarkConfig(
    autoformalization_model="mock"
)
```

### Batch Processing

For large-scale experiments:

```python
import multiprocessing as mp

def run_config(config_path):
    config = load_config(config_path)
    runner = BenchmarkRunner(config)
    return runner.run()

# Run multiple configs in parallel
config_paths = ["config1.yaml", "config2.yaml", "config3.yaml"]
with mp.Pool(3) as pool:
    results = pool.map(run_config, config_paths)
```

### Statistical Analysis

```python
from veriform.analysis import test_significance

# Test significance
sig_results = test_significance(
    probabilities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    error_rates=[0.1, 0.2, 0.35, 0.5, 0.65, 0.8],
    alpha=0.05
)

print(sig_results["interpretation"])
# "The relationship is significant at α=0.05 level with large effect size."
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'veriform'`
- **Solution**: Install with `pip install -e .` from the project root

**Issue**: API rate limits
- **Solution**: Reduce `sample_size` or add delays between requests

**Issue**: Lean verification timeout
- **Solution**: Increase `lean_timeout` in configuration

**Issue**: Out of memory
- **Solution**: Process data in smaller batches or reduce `sample_size`

### Getting Help

- Check the [GitHub Issues](https://github.com/yourusername/veriform/issues)
- Read the [API documentation](./API.md)
- Contact the maintainers

## Best Practices

1. **Start small**: Use `sample_size=10` for initial testing
2. **Use mock mode**: Test your pipeline with `model="mock"` before using real APIs
3. **Save intermediate results**: Always set `save_intermediate=true`
4. **Set random seeds**: Ensure reproducibility with `random_seed`
5. **Version control configs**: Keep configuration files in git
6. **Document experiments**: Add metadata to track what you're testing

## Next Steps

- Read the [API Documentation](./API.md)
- Explore [Example Notebooks](../experiments/notebooks/)
- Check out [Advanced Examples](./EXAMPLES.md)
- Learn about [Contributing](../CONTRIBUTING.md)
