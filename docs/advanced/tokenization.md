# Tokenization & Multilingual Vocabulary Adaptation

AlignTune's `core/tokenization/` module extends a base model's tokenizer to cover a new
language or domain, without training a new model from scratch. It's built on research from
["Teaching Old Tokenizers New Words"](https://arxiv.org/abs/2512.03989) (Purason et al., 2025)
and can be combined with [model merging](merging.md).

The full pipeline is: **extend the vocabulary → (optionally) prune it → initialize the new
tokens' embeddings → fine-tune.**

## Tokenizer training and vocabulary extension

`create_tokenization_trainer()` trains or extends the tokenizer vocabulary from
the target-language/domain corpus. It does not load or fine-tune the language
model weights. AlignTune currently exposes two extension methods:

| Method | Mechanism | Best use case |
|---|---|---|
| Continued BPE | Continues the base tokenizer's own BPE merge learning on the new corpus | Adding a language with minimal wasted vocabulary, ≤2% unreachable tokens vs. 5–12% for naive extension |
| Naive extension | Trains a separate tokenizer on the new corpus, diffs it against the base vocabulary | Fast prototyping when tokenizer efficiency isn't the priority yet |

## Vocabulary pruning

After extension, **leaf-based** pruning removes redundant/unreachable tokens to shrink the
embedding table before deployment. It's the structure-safe method, naive frequency/last-N
pruning can break the BPE merge graph and create new unreachable tokens; leaf-based pruning
doesn't.

## Embedding initialization and training

New tokens need embeddings. **Fast Vocabulary Transfer (FVT)** initializes them
from the mean of their constituent subword embeddings instead of random noise.
Embedding initialization is applied when the model is loaded for SFT via
`embedding_init_method`:

- `"random"` (default): HuggingFace's standard resize, random init.
- `"mean"` / `"mean_of_constituents"`: FVT.

Use `train_embeddings=True` when the new input/output embedding rows should be
updated during SFT (with or without adapter training). `embedding_pad_to_multiple_of`
can pad the resized embedding table for hardware-friendly dimensions. These
embedding controls are currently supported by the TRL SFT path.

## `create_tokenization_trainer()`

```python
from aligntune.core.backend_factory import create_tokenization_trainer

trainer = create_tokenization_trainer(
    base_model="meta-llama/Llama-2-7b-hf",
    target_languages=["hi"],
    dataset_name="wikimedia/wikipedia",
    config_name="20231101.hi",       # HF dataset config for the target language
    num_new_tokens=20000,
    extension_method="continued_bpe",  # or "naive_extension"
    output_dir="./llama2-hindi-tokenizer",
    # Optional pruning, applied after extension:
    prune=True,
    pruning_ratio=0.1,
    pruning_method="leaf_frequency",
    # Optional: push straight to the Hub
    hub_model_id="myusername/llama2-hindi-tokenizer",
)

result = trainer.train()
```

## End-to-end pipeline

The supported workflow is:

1. Train/extend the tokenizer vocabulary.
2. Load the extended tokenizer, initialize the new embeddings, and optionally
   train the embedding rows during an SFT run.
3. Continue with normal SFT using the resulting model (optional second stage).

There is no separate `EmbeddingTrainer`; embedding initialization and embedding
updates are handled by the SFT model loader/trainer.

```python
from aligntune.core.backend_factory import create_tokenization_trainer, create_sft_trainer

# 1. Train the tokenizer vocabulary.
tokenizer_trainer = create_tokenization_trainer(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    target_languages=["hi"],
    dataset_name="wikimedia/wikipedia",
    config_name="20231101.hi",
    num_new_tokens=20000,
    extension_method="continued_bpe",  # or "naive_extension"
    output_dir="./out_tokenizer_hindi",
)
tokenizer_trainer.train()

# 2. Initialize and train the new embedding rows during SFT.
embedding_trainer = create_sft_trainer(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name_or_path="./out_tokenizer_hindi",
    dataset_name="wikimedia/wikipedia",
    subset="20231101.hi",
    dataset_text_field="completion",
    backend="trl",
    embedding_init_method="mean_of_constituents",
    train_embeddings=True,
    use_peft=True,
    output_dir="./out_hindi_embedding_sft",
    num_epochs=1,
    max_seq_length=256,
)
embedding_trainer.train()

# 3. Optional regular SFT continuation from the adapted model.
sft_trainer = create_sft_trainer(
    model_name="./out_hindi_embedding_sft",
    tokenizer_name_or_path="./out_tokenizer_hindi",
    dataset_name="your-hindi-instruction-dataset",
    backend="trl",
    output_dir="./out_hindi_sft",
    use_peft=True,
)
sft_trainer.train()
```

For a single-stage run, combine steps 2 and 3: pass the extended tokenizer to
`create_sft_trainer()`, set `embedding_init_method`, and choose whether to set
`train_embeddings=True`.

### Minimal SFT example

Fine-tune with the extended tokenizer and FVT init:

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="meta-llama/Llama-2-7b-hf",
    tokenizer_name_or_path="./llama2-hindi-tokenizer",
    dataset_name="your-hindi-instruction-dataset",
    embedding_init_method="mean_of_constituents",
    backend="trl",
)
trainer.train()
```

This mirrors the original PR's "End-to-End Example: Multilingual Hindi LLM" workflow:
continued BPE → FVT init → SFT/RL fine-tune.

## Column validation caveat

Like the rest of the `data` module, column validation only warns, it doesn't block. A dataset
missing an expected text column will proceed and fail later, rather than at data-prep time.

## Notebooks

- [`notebooks/43_tokenization_continued_bpe.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/43_tokenization_continued_bpe.ipynb)
- [`notebooks/44_tokenization_naive_extension.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/44_tokenization_naive_extension.ipynb)
- [`notebooks/45_tokenization_vocab_pruning.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/45_tokenization_vocab_pruning.ipynb)
- [`notebooks/46_tokenization_fvt_embedding_init.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/46_tokenization_fvt_embedding_init.ipynb)

See also [Model Merging](merging.md) for combining per-language LoRA adapters (trained on a
vocab-extended base) into one multilingual model with linear or task-arithmetic merging.

## Reference

The vocabulary-extension workflow is informed by:

```bibtex
@misc{purason2025teachingoldtokenizersnew,
  title         = {Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models},
  author        = {Taido Purason and Pavel Chizhov and Ivan P. Yamshchikov and Mark Fishel},
  year          = {2025},
  eprint        = {2512.03989},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2512.03989},
}
```
