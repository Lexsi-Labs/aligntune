# Production Compositions

AlignTune can orchestrate multi-stage training pipelines, e.g. `SFT → MoA → ES → DPO → audit`: with checkpoints threaded automatically from one stage to the next, via `core/composition/` (`Composition`, `Stage`, `CompositionLoader`, `CompositionRunner`).

## Why compositions

Chaining SFT, adapter, RL, and audit stages by hand means manually wiring each stage's output checkpoint as the next stage's `init_from`, and re-running failed pipelines from scratch. A composition YAML declares the stages once; the runner threads checkpoints between them and can skip/resume around failures.

## Composition Files

A composition is a YAML file listing ordered stages, each pointing at a normal training config and optionally overriding parameters. Below is a real, working example built entirely from recipe configs that ship in this repo, a base SFT run, followed by a Mixture-of-Adapters (MoA) stage initialized from it, followed by an Evolution-Strategies (ES) router-tuning stage initialized from the MoA checkpoint:

```yaml
name: "sft_moa_es"
description: "SFT -> MoA -> ES pipeline using real repo recipe configs"

stages:
  - name: "sft"
    algo: "sft"
    config_path: "recipes/configs/sft/llama3_2_3b_rbi_compliance.yaml"

  - name: "moa"
    algo: "sft"
    config_path: "recipes/configs/sft/llama3_moa_4experts.yaml"
    init_from: "sft"
    target_params:
      train:
        epochs: 2
        learning_rate: 1e-4

  - name: "es"
    algo: "es"
    config_path: "recipes/configs/es/moa_router_tune.yaml"
    init_from: "moa"
```

`init_from` threads the checkpoint from a named earlier stage into the current one; `target_params` overrides specific fields of that stage's config without duplicating the whole file.

⚠️ **Note**: The two bundled templates in `recipes/configs/compositions/`: `full_stack.yaml` and
`minimal_stack.yaml`: describe a longer `SFT → MoA → ES → DPO → audit` pipeline, but several of the
per-stage `config_path` entries they reference (e.g. `recipes/configs/sft/llama3_supervised.yaml`,
`recipes/configs/sft/llama3_audit.yaml`, and anything under a `recipes/configs/rl/` directory) do not
currently exist in this repo, there is no `recipes/configs/rl/` directory at all. Treat those two files
as illustrative structural templates to copy and adapt with your own config paths, not as run-as-is
examples; the inline example above is the one that works unmodified against files in this repo today.

## Running Compositions

```bash
# Run a composition pipeline
aligntune compose run recipes/configs/compositions/full_stack.yaml

# Continue past a failed stage; pin device and log level
aligntune compose run recipes/configs/compositions/full_stack.yaml --device cuda --skip-failed --log-level DEBUG

# List available composition templates
aligntune compose list

# Inspect a composition's stages without running it
aligntune compose inspect recipes/configs/compositions/full_stack.yaml
```

See the [CLI Commands reference](../cli/commands.md#composition-commands) for full options.

## See Also

- [Advanced Adapters](adapters.md): MoA/Text2LoRA stages a composition can chain
- [Model Merging](merging.md)
- [CLI Commands: compose](../cli/commands.md#composition-commands)
