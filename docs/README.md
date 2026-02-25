# AlignTune Documentation

This directory contains the complete documentation for AlignTune.

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
# From project root
pip install -r requirements-docs.txt

# Or install manually
pip install mkdocs mkdocs-material mkdocs-jupyter mkdocstrings[python] mkdocs-mermaid2-plugin pymdown-extensions
```

### Build Documentation

```bash
# First, install documentation dependencies
pip install -r requirements-docs.txt

# Serve locally (with auto-reload)
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

**Note**: Documentation dependencies are separate from runtime dependencies. Install them with `pip install -r requirements-docs.txt` from the project root.

## Documentation Structure

```
docs/
 index.md # Homepage
 getting-started/ # Getting started guides
 installation.md
 quickstart.md
 configuration.md
 backend-selection.md
 user-guide/ # User guides
 sft.md
 rl.md
 reward-functions.md
 reward-model-training.md
 evaluation.md
 model-management.md
 api-reference/ # API documentation
 overview.md
 core.md
 backend-factory.md
 configuration.md
 trainers.md
 cli/ # CLI documentation
 overview.md # CLI overview
 commands.md # CLI commands
 configuration.md # CLI configuration
 api-reference/ # API reference documentation
 overview.md # API overview
 unified-api.md # Unified API system
 core.md # Core API
 backend-factory.md # Backend factory
 configuration.md # Configuration classes
 trainers.md # Trainer classes
 reward-functions-reference.md # Reward functions reference
 reward-model-training-reference.md # Reward model training reference
 examples/ # Examples
 overview.md
 sft.md
 rl.md
 advanced.md
 advanced/ # Advanced topics
 architecture.md
 custom-backends.md
 distributed.md
 performance.md
 compatibility/ # Compatibility guides
 backend-matrix.md
 contributing/ # Contributing guides
 guide.md
 code-style.md
 testing.md
 unsloth_compatibility.md # Unsloth compatibility
```

## Adding Documentation

### New Page

1. Create markdown file in appropriate directory
2. Add entry to `mkdocs.yml` navigation
3. Follow existing documentation style

### Updating Existing Page

1. Edit the markdown file
2. Test locally with `mkdocs serve`
3. Submit PR with changes

## Documentation Style

- Use clear, concise language
- Include code examples
- Add tables for parameters
- Use admonitions for important notes
- Keep formatting consistent

## Questions?

- Check existing documentation
- Open an issue for questions
- Contribute improvements via PR