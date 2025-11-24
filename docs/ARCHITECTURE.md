# Veriform Architecture

This document describes the architecture and design of Veriform.

## Overview

Veriform is organized into modular components that work together to benchmark autoformalization systems:

```
Data Collection → Perturbation → Autoformalization → Verification → Analysis
```

## Module Structure

### 1. Data Collection (`src/veriform/data_collection/`)

**Purpose**: Load and parse reasoning datasets into standardized format

**Key Components**:
- `reasoning_step.py`: Core data structures
  - `ReasoningStep`: Individual reasoning step
  - `ReasoningChain`: Sequence of reasoning steps
  - `StepType`: Classification of step types

- `dataset_loaders.py`: Dataset loading implementations
  - `DatasetLoader`: Abstract base class
  - `GSM8KLoader`: Load GSM8K mathematical reasoning
  - `MATHLoader`: Load MATH competition problems
  - `CustomLoader`: Load custom datasets
  - `get_loader()`: Factory function

**Design Decisions**:
- Unified data structure across datasets
- Support for step-level metadata
- Lazy loading for memory efficiency

### 2. Perturbation (`src/veriform/perturbation/`)

**Purpose**: Introduce controlled errors into reasoning steps

**Key Components**:
- `strategies.py`: Perturbation strategies
  - `PerturbationStrategy`: Abstract base class
  - `OperatorSwapStrategy`: Swap arithmetic operators
  - `ValueChangeStrategy`: Modify numerical values
  - `LogicalNegationStrategy`: Negate logical statements
  - `SignFlipStrategy`: Flip number signs
  - `WordSwapStrategy`: Swap semantic words
  - `STRATEGY_REGISTRY`: Strategy registry

- `engine.py`: Perturbation execution
  - `PerturbationEngine`: Apply strategies with probability
  - Statistics tracking
  - Batch processing

**Design Decisions**:
- Strategy pattern for extensibility
- Probabilistic application
- Preserves original content for analysis
- Random seed support for reproducibility

### 3. Autoformalization (`src/veriform/autoformalization/`)

**Purpose**: Convert natural language reasoning to Lean code

**Key Components**:
- `formalizer.py`: LLM-based formalization
  - `Autoformalization`: Abstract base class
  - `OpenAIFormalizer`: OpenAI API implementation
  - `AnthropicFormalizer`: Anthropic API implementation
  - `MockFormalizer`: Testing implementation
  - `FormalizationResult`: Result container

- `prompts.py`: Prompt templates
  - System prompts for faithful formalization
  - Context-aware prompt generation
  - Code extraction patterns

- `lean_verifier.py`: Lean code verification
  - `LeanVerifier`: Execute Lean checker
  - `MockLeanVerifier`: Testing implementation
  - `VerificationResult`: Result container

**Design Decisions**:
- Provider-agnostic interface
- Faithful formalization emphasis (preserve errors)
- Retry logic with exponential backoff
- Context management (sorry lemmas)

### 4. Benchmark (`src/veriform/benchmark/`)

**Purpose**: Orchestrate the complete benchmarking pipeline

**Key Components**:
- `runner.py`: Main benchmark execution
  - `BenchmarkRunner`: Pipeline coordinator
  - `BenchmarkRun`: Single probability run results
  - `BenchmarkResults`: Complete benchmark results
  - Progress tracking
  - Intermediate result saving

**Design Decisions**:
- Lazy initialization of components
- Progress visualization with tqdm
- Modular execution (run per probability)
- Comprehensive result tracking

### 5. Analysis (`src/veriform/analysis/`)

**Purpose**: Statistical analysis and visualization

**Key Components**:
- `statistics.py`: Statistical analysis
  - `compute_correlation()`: Pearson, Spearman, Kendall
  - `analyze_faithfulness()`: Key faithfulness metrics
  - `test_significance()`: Hypothesis testing
  - `compute_summary_statistics()`: Descriptive stats

- `visualization.py`: Result visualization
  - `plot_error_rate_vs_perturbation()`: Main correlation plot
  - `plot_correlation_heatmap()`: Correlation matrix
  - `plot_per_run_statistics()`: Detailed run stats
  - `create_summary_report()`: Complete report generation

**Design Decisions**:
- Multiple correlation metrics
- Publication-ready visualizations
- Statistical significance testing
- Comprehensive reporting

### 6. Configuration (`src/veriform/config.py`)

**Purpose**: Centralized configuration management

**Key Components**:
- `BenchmarkConfig`: Main configuration class
- `PerturbationConfig`: Perturbation settings
- `DatasetConfig`: Dataset settings

**Design Decisions**:
- Pydantic models for validation
- Type hints for clarity
- Sensible defaults
- Environment variable support

### 7. Utilities (`src/veriform/utils/`)

**Purpose**: Shared helper functions

**Key Components**:
- Config loading/saving (YAML/JSON)
- Result loading
- Common utilities

## Data Flow

```
1. Configuration
   └─> BenchmarkConfig loaded from YAML/JSON

2. Data Loading
   └─> DatasetLoader loads ReasoningChains

3. For each perturbation probability:
   a. Perturbation
      └─> PerturbationEngine perturbs chains

   b. Autoformalization
      └─> Autoformalization converts steps to Lean

   c. Verification
      └─> LeanVerifier checks Lean code

   d. Collection
      └─> BenchmarkRun stores results

4. Analysis
   └─> Statistics computed
   └─> Visualizations generated
   └─> Report created
```

## Key Design Patterns

### 1. Strategy Pattern
Used in perturbation strategies for extensibility:
```python
class PerturbationStrategy(ABC):
    @abstractmethod
    def apply(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        pass
```

### 2. Factory Pattern
Used for creating components:
```python
def get_loader(dataset_name: str, **kwargs) -> DatasetLoader:
    return loaders[dataset_name](**kwargs)
```

### 3. Builder Pattern
Used for configuration:
```python
config = BenchmarkConfig(
    perturbation_probabilities=[0.0, 0.5, 1.0],
    dataset_name="gsm8k",
    ...
)
```

### 4. Template Method
Used in dataset loaders:
```python
class DatasetLoader(ABC):
    def load(self) -> List[ReasoningChain]:
        # Common loading logic
        pass

    @abstractmethod
    def parse_reasoning_steps(self, example) -> List[ReasoningStep]:
        # Dataset-specific parsing
        pass
```

## Extension Points

### Adding New Perturbation Strategies

1. Subclass `PerturbationStrategy`
2. Implement `can_apply()` and `apply()`
3. Register in `STRATEGY_REGISTRY`

### Adding New Datasets

1. Subclass `DatasetLoader`
2. Implement `load()` and `parse_reasoning_steps()`
3. Register in `get_loader()`

### Adding New Autoformalization Providers

1. Subclass `Autoformalization`
2. Implement `_call_llm()`
3. Register in `get_formalizer()`

### Adding New Analysis Methods

1. Add function to `statistics.py`
2. Add visualization to `visualization.py` if needed
3. Update report generation

## Testing Strategy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Mock Objects**: Use mock implementations for testing
  - `MockFormalizer`: Test without API calls
  - `MockLeanVerifier`: Test without Lean installed

## Performance Considerations

- **Lazy Loading**: Components initialized only when needed
- **Batch Processing**: Process multiple items together
- **Caching**: Cache API responses (future work)
- **Parallel Processing**: Support for parallel execution (future work)

## Error Handling

- **Retries**: Automatic retry with exponential backoff
- **Graceful Degradation**: Continue on individual failures
- **Comprehensive Logging**: Track errors and warnings
- **Validation**: Pydantic validates configurations

## Future Improvements

1. **Caching Layer**: Cache API responses
2. **Parallel Processing**: Process chains in parallel
3. **Incremental Results**: Stream results as they complete
4. **Web Interface**: Browser-based benchmark runner
5. **Database Backend**: Store results in database
6. **Distributed Execution**: Run across multiple machines

## Dependencies

### Core
- `pydantic`: Configuration validation
- `numpy`, `pandas`: Data processing
- `scipy`: Statistical analysis

### Visualization
- `matplotlib`, `seaborn`: Plotting

### LLM Providers
- `openai`: OpenAI API
- `anthropic`: Anthropic API

### Datasets
- `datasets`: HuggingFace datasets

### Utilities
- `tqdm`: Progress bars
- `pyyaml`: YAML parsing

## Versioning

Veriform follows semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Current version: 0.1.0 (Alpha)
