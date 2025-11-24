# Contributing to Veriform

Thank you for your interest in contributing to Veriform! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant code snippets or error messages

### Suggesting Features

Feature suggestions are welcome! Please:
- Check existing issues first
- Clearly describe the use case
- Explain why it would be useful
- Provide examples if possible

### Contributing Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/veriform.git
   cd veriform
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes**
   - Write clear, documented code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

6. **Run tests**
   ```bash
   pytest tests/
   ```

7. **Format code**
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

8. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

9. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all public functions/classes
- Keep functions focused and small
- Use meaningful variable names

Example:
```python
def compute_correlation(
    probabilities: List[float],
    error_rates: List[float]
) -> Dict[str, float]:
    """
    Compute correlation between perturbation probability and error rate.

    Args:
        probabilities: List of perturbation probabilities
        error_rates: List of corresponding error rates

    Returns:
        Dictionary with correlation metrics
    """
    # Implementation...
```

### Testing

- Write unit tests for new functions
- Write integration tests for new features
- Aim for >80% code coverage
- Use pytest fixtures for common setup

Example:
```python
def test_perturbation_strategy():
    """Test that operator swap works correctly."""
    strategy = OperatorSwapStrategy()
    step = ReasoningStep(
        step_id="test",
        content="2 + 2 = 4"
    )

    perturbed = strategy.apply(step)

    assert perturbed.is_perturbed
    assert perturbed.perturbation_applied == "operator_swap"
    assert "+" not in perturbed.content or perturbed.content != step.content
```

### Documentation

- Update README.md for major changes
- Add docstrings to all public APIs
- Update USAGE.md for new features
- Add examples for complex functionality

### Commit Messages

Use clear, descriptive commit messages:
- `feat: Add new perturbation strategy for ...`
- `fix: Resolve issue with ...`
- `docs: Update usage guide for ...`
- `test: Add tests for ...`
- `refactor: Improve ...`

## Areas for Contribution

### High Priority

- [ ] Additional perturbation strategies
- [ ] Support for more datasets (ProofNet, MiniF2F, etc.)
- [ ] Better Lean verification integration
- [ ] Performance optimizations
- [ ] More statistical analysis methods

### Medium Priority

- [ ] Web interface for running benchmarks
- [ ] Real-time monitoring dashboard
- [ ] Support for other proof assistants (Coq, Isabelle)
- [ ] Automated hyperparameter tuning
- [ ] Multi-language support

### Low Priority

- [ ] Additional visualization options
- [ ] Export to different formats
- [ ] Integration with MLOps tools

## Adding New Features

### Adding a Perturbation Strategy

1. Create a new class in `src/veriform/perturbation/strategies.py`:
   ```python
   class MyStrategy(PerturbationStrategy):
       def can_apply(self, step: ReasoningStep) -> bool:
           # Check if applicable
           pass

       def apply(self, step: ReasoningStep) -> Optional[ReasoningStep]:
           # Apply perturbation
           pass
   ```

2. Register in `STRATEGY_REGISTRY`

3. Add tests in `tests/unit/test_perturbation.py`

4. Update documentation

### Adding a Dataset Loader

1. Create a new class in `src/veriform/data_collection/dataset_loaders.py`:
   ```python
   class MyDatasetLoader(DatasetLoader):
       def load(self) -> List[ReasoningChain]:
           # Load data
           pass

       def parse_reasoning_steps(self, example) -> List[ReasoningStep]:
           # Parse steps
           pass
   ```

2. Register in `get_loader()` function

3. Add tests

4. Update documentation

### Adding an Analysis Method

1. Add function to `src/veriform/analysis/statistics.py`

2. Add visualization to `src/veriform/analysis/visualization.py` if needed

3. Add tests

4. Update usage guide

## Review Process

1. Maintainers will review your PR
2. Address any feedback
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged in releases

## Questions?

Feel free to:
- Open an issue for questions
- Reach out to maintainers
- Join discussions on GitHub

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Contributors will be listed in:
- GitHub contributors page
- Release notes
- ACKNOWLEDGMENTS.md (for significant contributions)

Thank you for contributing to Veriform!
