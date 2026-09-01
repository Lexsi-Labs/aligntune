# RAFT (Retrieval Augmented Fine-Tuning)

RAFT trains a model to answer questions using a mix of "golden" (relevant) and "distractor" (irrelevant) retrieved documents in its context, so the model learns to identify and cite the correct source rather than blindly trusting whatever is retrieved. It targets retrieval-augmented-generation (RAG) deployments where retrieval quality is imperfect.

## Overview

Each training example provides a question, an answer, a set of golden documents, and a set of distractor documents. AlignTune formats these into a single document-augmented context per example (capping how many golden/distractor documents are included) and trains on the resulting sequences. An optional citation-quality signal (`use_citation_loss`) tracks whether the model's answer draws from the golden documents.

### When to use RAFT

- Your model will be deployed behind a RAG pipeline and needs to be robust to retrieval noise.
- You want the model to learn to prefer/cite correct source documents over distractors already at fine-tuning time, instead of relying purely on prompt engineering at inference time.

## Configuration

RAFT has its own factory function (`create_raft_trainer`), separate from `create_sft_trainer`/`create_rl_trainer`, and supports both the `trl` and `unsloth` backends:

```python
from aligntune.core.backend_factory import create_raft_trainer

train_examples = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "golden_docs": [{"title": "France", "text": "...Paris is the capital..."}],
        "distractor_docs": [{"title": "Germany", "text": "...Berlin is the capital..."}],
    },
    # ...
]

trainer = create_raft_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    train_examples=train_examples,
    backend="trl",              # or "unsloth"
    max_golden_docs=3,
    max_distractor_docs=5,
    use_citation_loss=True,
)

trainer.train()
```

### Configuration Options

--8<-- "docs/PARAMETERS.md:raft"

## Related

- Data assembly: `aligntune.data.raft_dataset_builder`
- Backend implementations: `aligntune.backends.trl.raft.raft_trainer`, `aligntune.backends.unsloth.raft.raft_trainer`
- Local notebook: [`notebooks/25_raft.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/25_raft.ipynb)

## See Also

- [Algorithms Overview](overview.md)
- [RAFT Parameters](../PARAMETERS.md#raft-retrieval-augmented-fine-tuning) - Full parameter reference
