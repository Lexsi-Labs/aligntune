# Contributing Guide

Thank you for your interest in contributing to AlignTune! This guide will help you get started.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/Lexsi-Labs/aligntune.git
cd aligntune
```

### 2. Set Up Development Environment

```bash
# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Contribution Areas

### Code Contributions

- Bug fixes
- New features
- Performance improvements
- Documentation updates

### Documentation Contributions

- Fix typos
- Improve clarity
- Add examples
- Update guides

### Testing Contributions

- Add test cases
- Improve test coverage
- Fix failing tests

## Development Workflow

### 1. Make Changes

Make your changes following the [Code Style Guide](code-style.md).

### 2. Write Tests

Add tests for your changes:

```python
# tests/test_your_feature.py
def test_your_feature():
 # Your test code
 pass
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=aligntune
```

### 4. Check Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type check
mypy src/

# Lint
ruff check src/ tests/
```

### 5. Commit Changes

```bash
git add .
git commit -m "Add: your feature description"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

## Code Style

See [Code Style Guide](code-style.md) for detailed guidelines.

## Testing

See [Testing Guide](testing.md) for testing guidelines.

## Pull Request Process

1. **Update Documentation**: Update relevant documentation
2. **Add Tests**: Add tests for new features
3. **Update Changelog**: Add entry to CHANGELOG.md
4. **Ensure Tests Pass**: All tests must pass
5. **Code Review**: Address review comments

## Questions?

- Open an issue for questions
- Join discussions on GitHub
- Check existing issues and PRs

Thank you for contributing!