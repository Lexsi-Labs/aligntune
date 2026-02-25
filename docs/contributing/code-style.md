# Code Style Guide

Code style guidelines for contributing to AlignTune.

## Overview

AlignTune follows Python best practices and uses automated formatting tools to ensure consistent code style.

## Code Formatting

### Black

We use [Black](https://black.readthedocs.io/) for code formatting.

**Configuration:**
- Line length: 100 characters
- Target Python versions: 3.8, 3.9, 3.10, 3.11

**Usage:**
```bash
# Format code
black src/

# Check formatting
black --check src/
```

### isort

We use [isort](https://pycqa.github.io/isort/) for import sorting.

**Configuration:**
- Profile: black
- Line length: 100
- Multi-line output: 3

**Usage:**
```bash
# Sort imports
isort src/

# Check imports
isort --check src/
```

## Linting

### Flake8

We use [Flake8](https://flake8.pycqa.org/) for linting.

**Configuration:**
- Line length: 100
- Ignore: E501 (handled by Black)

**Usage:**
```bash
flake8 src/
```

### Ruff (Alternative)

We also support [Ruff](https://github.com/astral-sh/ruff) as a modern alternative.

**Usage:**
```bash
ruff check src/
ruff format src/
```

## Type Hints

### Type Annotations

Always use type hints for function signatures:

```python
from typing import Dict, List, Optional, Union

def create_trainer(
 model_name: str,
 dataset_name: str,
 backend: str = "auto",
 max_samples: Optional[int] = None
) -> TrainerBase:
 """Create trainer with specified parameters."""
 pass
```

### Type Checking

We use [mypy](https://mypy.readthedocs.io/) for type checking.

**Usage:**
```bash
mypy src/
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def train_model(
 model: Model,
 dataset: Dataset,
 epochs: int = 3
) -> Dict[str, Any]:
 """Train model on dataset.
 
 Args:
 model: Model to train
 dataset: Training dataset
 epochs: Number of training epochs
 
 Returns:
 Dictionary with training results
 
 Raises:
 TrainingError: If training fails
 """
 pass
```

### Inline Comments

- Use comments to explain "why", not "what"
- Keep comments up-to-date with code
- Remove commented-out code

## Naming Conventions

### Variables and Functions

- Use `snake_case` for variables and functions
- Use descriptive names
- Avoid abbreviations

```python
# Good
def create_sft_trainer(): ...
model_name = "microsoft/DialoGPT-medium"

# Bad
def createSFT(): ...
mn = "microsoft/DialoGPT-medium"
```

### Classes

- Use `PascalCase` for classes
- Use descriptive names

```python
# Good
class SFTTrainerBase: ...
class BackendFactory: ...

# Bad
class Trainer: ...
class Factory: ...
```

### Constants

- Use `UPPER_SNAKE_CASE` for constants

```python
# Good
MAX_SEQUENCE_LENGTH = 2048
DEFAULT_BATCH_SIZE = 4

# Bad
maxSequenceLength = 2048
default_batch_size = 4
```

## Code Organization

### Imports

Order imports as follows:

1. Standard library
2. Third-party packages
3. Local imports

```python
# Standard library
import logging
from typing import Dict, List

# Third-party
import torch
from transformers import AutoModel

# Local
from aligntune.core.backend_factory import create_sft_trainer
```

### File Structure

```python
"""Module docstring."""

# Imports
import ...

# Constants
DEFAULT_VALUE = ...

# Classes
class MyClass: ...

# Functions
def my_function(): ...

# Main (if applicable)
if __name__ == "__main__":
 ...
```

## Error Handling

### Exceptions

Use custom exceptions with clear messages:

```python
from aligntune.utils.errors import AlignTuneError

class ConfigurationError(AlignTuneError):
 """Configuration error."""
 pass

# Usage
if not config.model.name:
 raise ConfigurationError("Model name is required")
```

### Error Messages

- Be specific and actionable
- Include context
- Suggest solutions

```python
# Good
raise ValueError(
 f"Invalid backend '{backend}'. "
 f"Available backends: {available_backends}. "
 f"Use 'auto' for automatic selection."
)

# Bad
raise ValueError("Invalid backend")
```

## Testing

### Test Structure

- One test file per module
- Test file: `test_<module_name>.py`
- Test class: `Test<ClassName>`
- Test function: `test_<function_name>`

```python
# test_backend_factory.py
class TestBackendFactory:
 def test_create_sft_trainer(self):
 """Test SFT trainer creation."""
 pass
```

## Pre-commit Hooks

### Setup

```bash
pip install pre-commit
pre-commit install
```

### Hooks

We use pre-commit hooks for:
- Black formatting
- isort import sorting
- Flake8 linting
- mypy type checking

## Best Practices

1. **Format Before Committing**: Run Black and isort
2. **Type Hints**: Add type hints to all functions
3. **Documentation**: Document public APIs
4. **Error Handling**: Use custom exceptions
5. **Testing**: Write tests for new features

## Next Steps

- [Testing Guide](testing.md) - Testing guidelines
- [Contributing Guide](guide.md) - General contributing guide