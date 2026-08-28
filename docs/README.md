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

The `nav:` section of `mkdocs.yml` is the source of truth for structure and
ordering. Current top-level layout:

```
docs/
  index.md                         # Homepage
  getting-started/                 # installation, quickstart, basic-concepts,
                                   #   configuration, backend-selection
  user-guide/                      # sft, rl, distillation, reward-functions,
                                   #   reward-model-training, evaluation,
                                   #   model-management, sample-logging,
                                   #   troubleshooting, overview
  algorithms/                      # overview + one page per algorithm
                                   #   (dpo, online-dpo, ppo, grpo, gspo, dapo,
                                   #    dr-grpo, gbmpo, counterfactual-grpo,
                                   #    pace, orpo, spin, raft)
  PARAMETERS.md                    # full parameter reference
  backends/                        # overview, trl, unsloth, comparison
  api-reference/                   # overview, core, backend-factory,
                                   #   configuration, trainers, unified-api,
                                   #   reward-functions-reference,
                                   #   reward-model-training-reference
  advanced/                        # architecture, adapters, merging,
                                   #   long-context, tokenization, distillation,
                                   #   composition, indic
  examples/                        # overview, sft, rl, distillation, advanced
  notebooks/                       # demo, lexsi-sdk, local + standalone .ipynb
  cli/                             # overview, commands, configuration,
                                   #   raw-factory-configs (+ cli-reference.md)
  compatibility/                   # backend-matrix (+ unsloth_compatibility.md)
  contributing/                    # guide, code-style, testing
  community/                       # faq
  novelty_frontiers.md             # research roadmap
  CHANGELOG.md  ISSUES.md  CODE_OF_CONDUCT.md  SECURITY.md
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