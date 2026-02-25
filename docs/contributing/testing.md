# Testing Guide

Testing guidelines for contributing to AlignTune.

## Overview

AlignTune uses [pytest](https://pytest.org/) for testing with comprehensive coverage requirements.

## Test Structure

### Directory Layout

```
tests/
 __init__.py
 core/
 test_backend_factory.py
 test_config.py
 backends/
 trl/
 unsloth/
 integration/
 test_training_pipeline.py
 fixtures/
 sample_data.py
```

### Test File Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<function_name>`

```python
# test_backend_factory.py
class TestBackendFactory:
 def test_create_sft_trainer(self):
 """Test SFT trainer creation."""
 pass
```

## Writing Tests

### Basic Test

```python
import pytest
from aligntune.core.backend_factory import create_sft_trainer

def test_create_sft_trainer():
 """Test basic SFT trainer creation."""
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 max_samples=10
 )
 assert trainer is not None
 assert trainer.config.model.name_or_path == "microsoft/DialoGPT-small"
```

### Fixtures

Use pytest fixtures for reusable test data:

```python
import pytest
from aligntune.core.sft.config import SFTConfig

@pytest.fixture
def sample_config():
 """Sample SFT configuration."""
 return SFTConfig(
 model=ModelConfig(name_or_path="test-model"),
 dataset=DatasetConfig(name="test-dataset")
 )

def test_trainer_with_config(sample_config):
 """Test trainer with sample config."""
 trainer = create_sft_trainer_from_config(sample_config)
 assert trainer.config == sample_config
```

### Parametrized Tests

Use `@pytest.mark.parametrize` for multiple test cases:

```python
@pytest.mark.parametrize("backend", ["trl", "unsloth"])
def test_backend_selection(backend):
 """Test different backend selections."""
 trainer = create_sft_trainer(
 model_name="model",
 dataset_name="dataset",
 backend=backend
 )
 assert trainer.backend == backend
```

## Test Markers

### Available Markers

```python
@pytest.mark.slow # Slow tests
@pytest.mark.integration # Integration tests
@pytest.mark.gpu # Requires GPU
@pytest.mark.unsloth # Requires Unsloth
@pytest.mark.wandb # Requires WandB
```

### Usage

```python
@pytest.mark.slow
def test_large_model_training():
 """Test training with large model."""
 pass

@pytest.mark.gpu
def test_unsloth_backend():
 """Test Unsloth backend (requires GPU)."""
 pass
```

### Running Marked Tests

```bash
# Run all tests except slow
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu

# Run integration tests
pytest -m integration
```

## Test Coverage

### Coverage Requirements

- Target: >80% coverage
- Critical paths: 100% coverage
- New features: Must include tests

### Running Coverage

```bash
# Run with coverage
pytest --cov=src/aligntune --cov-report=html

# View HTML report
open htmlcov/index.html
```

## Mocking

### External Dependencies

Mock external dependencies:

```python
from unittest.mock import Mock, patch

@patch('aligntune.core.backend_factory.load_model')
def test_model_loading(mock_load):
 """Test model loading with mock."""
 mock_load.return_value = Mock()
 trainer = create_sft_trainer(...)
 mock_load.assert_called_once()
```

## Integration Tests

### Training Pipeline

```python
@pytest.mark.integration
def test_sft_training_pipeline():
 """Test complete SFT training pipeline."""
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 max_samples=10,
 num_epochs=1
 )
 
 # Train
 results = trainer.train()
 assert "train_loss" in results
 
 # Evaluate
 metrics = trainer.evaluate()
 assert "eval_loss" in metrics
 
 # Save
 path = trainer.save_model()
 assert path is not None
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Fast Tests**: Keep unit tests fast (<1s)
3. **Clear Names**: Use descriptive test names
4. **One Assertion**: One concept per test
5. **Fixtures**: Use fixtures for setup/teardown

## Running Tests

### All Tests

```bash
pytest
```

### Specific Test

```bash
pytest tests/core/test_backend_factory.py::test_create_sft_trainer
```

### Verbose Output

```bash
pytest -v
```

### With Coverage

```bash
pytest --cov=src/aligntune --cov-report=term-missing
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Pushes to main
- Scheduled runs

## Next Steps

- [Code Style Guide](code-style.md) - Code style guidelines
- [Contributing Guide](guide.md) - General contributing guide