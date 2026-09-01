# Model Merging

AlignTune's `core/merge/` module merges multiple trained models or LoRA adapters into a single
checkpoint, without any additional training. It provides two thin wrappers:

- **`MergekitMerger`**, routes to [mergekit](https://github.com/arcee-ai/mergekit) for
  `linear`, `task_arithmetic`, and `ram`.
- **`PEFTMerger`**, a direct `peft.PeftModel.merge_and_unload()` wrapper for folding a single
  LoRA adapter into its base model, with no mergekit YAML pipeline involved.

This module can be combined with [tokenization and multilingual vocabulary adaptation](tokenization.md).
The mergekit-backed methods need `mergekit`, which is **vendored** under
[`third_party/mergekit`](https://github.com/Lexsi-Labs/aligntune/tree/main/third_party/mergekit) rather than installed from PyPI, upstream
mergekit hard-pins `pydantic`/`safetensors`/`accelerate`/etc. in a way that would downgrade this
project's own versions and break `vllm`/`unsloth`/`mcp`. It's built into the `aligntune` package
itself (not installed as a second distribution), so both

```bash
pip install -e .
# or
uv pip install -e .
```

already include it, no separate step needed.

See [`third_party/mergekit/PATCH_NOTES.md`](https://github.com/Lexsi-Labs/aligntune/blob/main/third_party/mergekit/PATCH_NOTES.md) for the two
small compatibility patches applied on top of vendored mergekit `v0.1.4`.

## Which method for which job

| Method | Mechanism | Best use case |
|---|---|---|
| `linear` | Simple weighted average | Quick merge of near-identical checkpoints (e.g. different SFT seeds) |
| `task_arithmetic` | Subtract base weights → task vectors → add/scale | Simplest multi-task merge |
| `ram` | Reinforced Agent Merging, separates shared vs. task-unique updates from **RL-trained** task vectors | Merging multiple RL-fine-tuned agents (GRPO/PPO adapters) without diluting specialization |

**`ram` is not actually implemented**, not in mergekit (checked every released
version) nor anywhere in aligntune. `merge_models(..., method="ram")` raises
`RuntimeError: Unimplemented merge method`. This is a pre-existing gap in mergekit itself, not
an aligntune bug.

## `merge_models()`

```python
from aligntune.core.backend_factory import merge_models

# Linear merge of three per-language LoRAs onto a shared base
merged_path = merge_models(
    models=["base_model", "base_model", "base_model"],
    lora_adapters=["./lora_en", "./lora_hi", "./lora_zh"],
    method="linear",
    weights=[0.33, 0.33, 0.34],
    output_path="./multilingual_merged",
)
```

Key parameters:

- `models`: model paths/HF IDs to merge. If `lora_adapters` is given, these are the **base**
  models the adapters get applied to before merging.
- `base_model`: required for every method except `linear`.
- `weights`: per-model weights (defaults to equal weights).

For merges that need per-layer weight gradients, filters, or layer ranges beyond what
`merge_models()` exposes, drop down to `merge_models_from_yaml(yaml_path, output_path)` with a
raw mergekit config.

## Direct LoRA merge (no mergekit)

When there's only **one** adapter to fold into its base, no cross-model interference to
manage, skip mergekit entirely:

```python
from aligntune.core.merge import PEFTMerger

merger = PEFTMerger()
merged_path = merger.merge_lora(
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path="./my_lora_checkpoint",
    output_path="./merged",
)
```

## Notebooks

- [`notebooks/35_merge_models.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/35_merge_models.ipynb) — linear / SLERP / TIES / DARE-TIES / task-arithmetic via Mergekit
- [`notebooks/42_merge_peft_direct_lora.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/42_merge_peft_direct_lora.ipynb) — direct LoRA-adapter merge (no Mergekit)
