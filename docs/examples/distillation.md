# Distillation Examples

AlignTune exposes two distillation trainers through
`create_distill_trainer()`:

- **Standard Distillation**: an external teacher provides the target
  distribution.
- **SDFT**: self-distillation from a base or EMA teacher, using privileged
  context.

Use `backend="trl"` for the recommended implementation. The Unsloth paths are
experimental acceleration paths.

## On-Policy and Off-Policy

For Standard Distillation:

- `on_policy=False` (default) trains from dataset/teacher completions and is
  the **off-policy** mode.
- `on_policy=True` makes the student generate its own completions and is the
  **on-policy** mode. AlignTune maps this to the underlying TRL `lmbda=1.0`
  setting.

SDFT is a self-distillation rollout method. It generates student
  responses during training, so its core workflow is on-policy. Do not treat
  it as ordinary offline distillation by only changing `on_policy`.

## Standard Distillation

### Off-policy (default)

```python
from aligntune.core.backend_factory import create_distill_trainer

trainer = create_distill_trainer(
    student_model="Qwen/Qwen3-0.6B",
    teacher_model="Qwen/Qwen3-4B",
    dataset_name="Salesforce/wikitext",
    config_name="wikitext-2-raw-v1",
    column_mapping={"prompt": "text"},
    backend="trl",
    on_policy=False,
    temperature=3.0,
    alpha=0.5,
    batch_size=4,
)

trainer.train()
```

### On-policy

```python
trainer = create_distill_trainer(
    student_model="Qwen/Qwen3-0.6B",
    teacher_model="Qwen/Qwen3-4B",
    dataset_name="Salesforce/wikitext",
    config_name="wikitext-2-raw-v1",
    column_mapping={"prompt": "text"},
    backend="trl",
    on_policy=True,
    temperature=3.0,
    batch_size=4,
)

trainer.train()
```

## SDFT (Self-Distillation Fine-Tuning)

SDFT has no external teacher model. The base or EMA teacher is selected with
`teacher_model_kind`, and the teacher receives privileged context. The input
dataset must finish with canonical `prompt` and `privileged_context` columns.

```python
from aligntune.core.backend_factory import create_distill_trainer

trainer = create_distill_trainer(
    student_model="Qwen/Qwen3-0.6B",
    teacher_model=None,
    teacher_model_kind="base",
    dataset_name="bhavyagoyal-lexsi/tat-dqa-with-retrieval-hints",
    backend="trl",
    column_mapping={
        "prompt": "prompt",
        "privileged_context": "privileged_context",
    },
    privileged_context_column="privileged_context",
    max_completion_length=256,
    num_generations=2,
    batch_size=2,
)

trainer.train()
```

### Science SDFT dataset

`stalaei/sdft-science-distil` provides `prompt` for the student and
`teacher_prompt` for the teacher. AlignTune maps `teacher_prompt` to the
canonical `privileged_context` field required by SDFT.

```python
science_sdft_trainer = create_distill_trainer(
    student_model="Qwen/Qwen3-0.6B",
    teacher_model=None,
    teacher_model_kind="base",
    dataset_name="stalaei/sdft-science-distil",
    backend="trl",
    column_mapping={
        "prompt": "prompt",
        "privileged_context": "teacher_prompt",
    },
    privileged_context_column="teacher_prompt",
    max_completion_length=256,
    num_generations=2,
    batch_size=2,
)

science_sdft_trainer.train()
```

## Choosing a Method

Choose the method from the teacher and loss settings you pass to
`create_distill_trainer()`:

| Method | Select it with | Teacher | Rollouts | Extra data |
|---|---|---|---|---|
| Standard | `teacher_model="..."` | External model | Optional; set `on_policy=True` to enable | Prompt plus completion/response data |
| SDFT | `teacher_model_kind="base"`, `"live"`, or `"ema"`; no rewards | Self-teacher | Required | `prompt` plus privileged teacher context |

For SDFT, map a dataset's hint column to the canonical
`privileged_context` field. For example, if the raw teacher hint is named
`teacher_prompt`:

```python
column_mapping={
    "prompt": "prompt",
    "privileged_context": "teacher_prompt",
}
privileged_context_column="teacher_prompt"
```

Standard Distillation uses the offline dataset by default. Set
`on_policy=True` only when you want the student to generate rollouts during
training; it is not required for selecting the method.
