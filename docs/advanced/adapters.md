# PEFT Variants & Advanced Adapters (v3.3 Suite)

AlignTune currently supports several adapter capabilities in two
layers:

- **Standard LoRA** in `core/peft/`, the factory-routed PEFT variant.
- **Three experimental standalone adapter tools** in `core/adapters/`:
  Mixture of Adapters, Text2LoRA, and Doc2LoRA.

Only the standard LoRA variant is selected automatically through
`create_sft_trainer(..., lora_variant=...)`. The standalone tools require their
own APIs or recipes and are not implicitly injected into every trainer.

## Support Map

| Capability | Location | Factory `lora_variant` | Status / boundary |
|---|---|---|---|
| Standard LoRA | `core/peft/` | `standard` | TRL and Unsloth |
| Mixture of Adapters | `core/adapters/moa/` | None | **Experimental** standalone layer/recipe |
| Text2LoRA | `core/adapters/text2lora/` | None | **Experimental** standalone hypernetwork/trainer |
| Doc2LoRA | `core/adapters/text2lora/` | None | **Experimental** document-conditioned generator |

## LoRA variant (`core/peft/`)

Select via `use_peft=True, lora_variant="standard"` on `create_sft_trainer()`:
`PEFTFactory` routes to the matching adapter class:

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    use_peft=True,
    lora_variant="standard",
    lora_r=8,
    lora_alpha=16,
)
trainer.train()
```

## Advanced adapters (`core/adapters/`)

The tools in this section are experimental. They have standalone smoke
tests and reference recipes, but they are not automatically integrated into
the SFT/RL trainer pipeline. Doc2LoRA currently uses a placeholder chunk
embedding path unless a production embedding implementation is supplied.

### Mixture of Adapters (MoA)

The `MoALoraLayer` wraps standard PyTorch linear layers with *N* different LoRA experts and uses a learned router to determine the top *K* experts per token dynamically.

**Key benefits:**
- High representational capacity while remaining computationally efficient (tokens are only routed to a subset of adapters).
- Features built-in load-balancing losses to prevent the router from defaulting to a single expert.

```python
from aligntune.core.adapters import MoALoraLayer
import torch.nn as nn

moa_layer = MoALoraLayer(
    base_module=nn.Linear(64, 64),
    num_experts=4, lora_r=8, lora_alpha=16, top_k=2,
)
out = moa_layer(hidden_states)          # (batch, seq_len, hidden_dim)
lb_loss = moa_layer.get_load_balance_loss()
```

A full SFT run wired with 4 experts / top-2 gating is available as a shipped recipe:
`recipes/configs/sft/llama3_moa_4experts.yaml` (and its ES router-only-tuning counterpart,
`recipes/configs/es/moa_router_tune.yaml`).

### Text2LoRA & Doc2LoRA

Training-free adapter generation: a hypernetwork maps a task-description embedding (Text2LoRA)
or a chunked-and-pooled long document (Doc2LoRA) directly to LoRA A/B matrices, no per-task
LoRA training loop required. Useful for per-tenant or per-document adapters generated on demand.

```python
import torch
from aligntune.core.adapters import TextToLoRAHypernet, DocToLoRA

hypernet = TextToLoRAHypernet(hidden_dim=384, lora_r=16, num_target_modules=4)
embedding_model = hypernet.get_embedding_model()  # sentence-transformers model
task_embedding = torch.tensor(embedding_model.encode(["Answer banking KYC questions."]))
lora_weights = hypernet(task_embedding)            # [{"A": ..., "B": ...}, ...]

doc2lora = DocToLoRA(hypernet, chunk_size=512, num_chunks=3, pooling_strategy="mean")
doc_lora_weights = doc2lora(long_document_text)
```

See `recipes/configs/meta/text2lora_meta_training.yaml` and
`recipes/configs/meta/text2lora_doc_personalization.yaml` for the shipped reference recipes.

## Notebooks

- [`notebooks/32_moa.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/32_moa.ipynb)
- [`notebooks/34_text2lora_doc2lora.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/34_text2lora_doc2lora.ipynb)
