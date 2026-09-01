# Knowledge Distillation Guide

AlignTune distills a larger **teacher** model into a smaller **student** model
through one factory entry point:

```python
from aligntune.core.backend_factory import create_distill_trainer
```

## Overview

AlignTune exposes two distillation methods:

1. **Standard Distillation**: the student matches an external teacher's logits.
2. **SDFT (Self-Distillation Fine-Tuning)**: self-distillation from a base,
   live, or EMA teacher state.

Both are available through the TRL and Unsloth backends when their
dependencies are installed.

## Method Selection

The factory detects the method from the arguments:

| Method | Required configuration |
|---|---|
| Standard | `teacher_model="..."` and no self-distillation flags |
| SDFT (Self-Distillation Fine-Tuning) | `teacher_model=None` and `teacher_model_kind="base"`, `"live"`, or `"ema"` |

Standard distillation supports two modes only: offline
(`on_policy=False`) uses dataset/teacher completions, while online
(`on_policy=True`) samples from the student. Intermediate offline/online
mixtures are not shown in this user guide; use the boolean switch for the
supported AlignTune workflows.

## Common Factory Pattern

```python
trainer = create_distill_trainer(
    student_model="Qwen/Qwen3-0.6B",
    teacher_model="Qwen/Qwen3-4B",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",  # or "unsloth"
    output_dir="./outputs/distillation",
    batch_size=4,
    num_epochs=1,
    learning_rate=5e-5,
    max_seq_length=1024,
    eval_strategy="no",
    report_to="none",
)

result = trainer.train()
```

CuratorKIT and DataManager normalize common prompt/completion, Alpaca, and
conversation formats. Use `column_mapping` or `processing_fn` when the source
dataset uses different names or structure.

## Standard Distillation

Use an external teacher with the default settings:

```python
trainer = create_distill_trainer(
    student_model="Qwen/Qwen3-0.6B",
    teacher_model="Qwen/Qwen3-4B",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    temperature=3.0,
    beta=1.0,
    batch_size=4,
    num_epochs=1,
)
```

Core controls are `temperature`, `beta`, `max_prompt_length`,
`max_completion_length`, `num_generations`, and `generation_batch_size`. Note:
`alpha` is accepted but not forwarded to the trainer, it has no effect.

For online standard distillation, add:

```python
on_policy=True
```

Use `backend="trl"` for the recommended implementation. `backend="unsloth"` is
experimental and can improve throughput, but has additional model-loading and
on-policy generation caveats, see
[Distillation Internals](../advanced/distillation.md) if you hit one.

## SDFT (Self-Distillation Fine-Tuning)

SDFT does not load an external teacher. It is on-policy self-distillation:

```text
student:  prompt -> completion
teacher:  prompt + privileged_context -> scores that completion
student:  matches the teacher distribution
```

The student normally generates from the plain `prompt`. The privileged
context is used to build the teacher prompt; it is not automatically shown to
the student. Set `generate_from_teacher=True` only when the rollout itself
should also use the enriched prompt.

SDFT does not require a reward function.

```python
trainer = create_distill_trainer(
    student_model="LiquidAI/LFM2.5-1.2B-Instruct",
    teacher_model=None,
    teacher_model_kind="base",  # "base", "live", or "ema"
    dataset_name="bhavyagoyal-lexsi/tat-dqa-with-retrieval-hints",
    backend="trl",  # or "unsloth"
    column_mapping={
        "prompt": "prompt",
        "privileged_context": "privileged_context",
    },
    privileged_context_column="privileged_context",
    distillation_mode="topk_logits",
    distillation_alpha=0.5,
    num_generations=2,
    max_completion_length=128,
    batch_size=2,
    num_epochs=1,
    eval_strategy="no",
    report_to="none",
)

result = trainer.train()
```

The dataset must expose these canonical fields after DataManager processing:

```python
{
    "prompt": [{"role": "user", "content": "..."}],
    "privileged_context": "A hint, correction, explanation, or retrieved context.",
}
```

If the raw dataset calls the hint `hints`, pass
`privileged_context_column="hints"` only when that column is present in the
loaded split. If the dataset is already in the canonical form above, use
`"privileged_context"` as shown. Check the processed columns before training;
rows missing either required field are filtered for SDFT.

For another dataset, set `privileged_context_column` to its raw hint,
feedback, explanation, or retrieval-context column.

Core SDFT controls are `teacher_model_kind`, `distillation_mode`,
`distillation_alpha`, `distillation_topk`, `teacher_update_rate`,
`teacher_sync_steps`, `generate_from_teacher`, `num_generations`,
`max_prompt_length`, `max_completion_length`, `generation_batch_size`,
`steps_per_generation`, `temperature`, `top_p`, `top_k`, and
`repetition_penalty`.

## Where to Go Next

- [Distillation Internals](../advanced/distillation.md): method-routing logic,
  TRL config classes, implementation gotchas, and the low-level API.
- [Distillation Parameters](../PARAMETERS.md#distillation-parameters):
  method-specific parameter tables and known limitations.
- [Evaluation Guide](evaluation.md): evaluate the trained student on task
  metrics rather than relying only on distillation loss.

For a quick smoke test, use `max_samples`, `max_steps=1`, and
`eval_strategy="no"`. Enable evaluation only when a validation split is
available.
