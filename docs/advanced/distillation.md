# Knowledge Distillation

This page covers distillation *internals*, how `create_distill_trainer()` routes
between the supported methods, which TRL config class backs each one, implementation
gotchas, and the low-level API for building a trainer without the factory
function. For practical, copy-pasteable examples of each method, see the
[User Guide: Distillation](../user-guide/distillation.md).

## Method routing

AlignTune supports full Student-Teacher workflows by natively extracting logits and representations from the Teacher during forward passes and scoring them against the Student via Kullback-Leibler (KL) Divergence or L2 losses.

`create_distill_trainer()` picks the method via
`UnifiedDistillConfig.get_distillation_type()`, based on **which kwargs you
set**, it is not inferred from `teacher_model` alone:

| Method | How it's selected | TRL config class |
|---|---|---|
| **Standard** | default (nothing below set) | `trl.experimental.distillation.DistillationConfig` |
| **SDFT** | `teacher_model_kind` set | `trl.experimental.sdft.SDFTConfig`: its own experimental trainer, not `SFTConfig`-based despite the name |

For datasets whose hint is not already named `privileged_context`, pass a
`privileged_context_column` override (for example,
`privileged_context_column="input"` for Alpaca-shaped data).

## Implementation gotchas

- **Standard**: `alpha` is accepted but not forwarded to the trainer, see the
  `alpha` row in [Distillation Parameters](../PARAMETERS.md#distillation-parameters).
- **Standard**: the underlying TRL `lmbda` field remains available for
  compatibility but is not part of the recommended API; use `on_policy` instead.
- **Standard (`backend="unsloth"`)**: experimental, can improve
  throughput, but has additional model-loading and on-policy generation
  caveats documented in the parameter reference.
- **SDFT**: TRL's default teacher prompt template is effectively
  `"{prompt}\n\n{privileged_context}"`.

## Parameter reference

--8<-- "docs/PARAMETERS.md:distill-standard"

--8<-- "docs/PARAMETERS.md:distill-sdft"

## Building a trainer without the factory function

`create_distill_trainer()` is the recommended entry point (see the
[User Guide](../user-guide/distillation.md)), but `UnifiedDistillConfig` can be
built directly if you'd rather not go through it:

```python
from aligntune.core.distill.config import (
    UnifiedDistillConfig, DistillModelConfig, DistillDatasetConfig, DistillTrainingConfig,
)
from aligntune.core.distill.trainer_factory import create_trainer_from_config

config = UnifiedDistillConfig(
    model=DistillModelConfig(student_model="Qwen/Qwen3-0.6B", teacher_model_kind="base"),
    dataset=DistillDatasetConfig(name="tatsu-lab/alpaca", privileged_context_column="input"),
    train=DistillTrainingConfig(per_device_batch_size=4),
)
assert config.get_distillation_type().value == "sdft"

trainer = create_trainer_from_config(config)
trainer.train()
```
