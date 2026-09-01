# Indic / Regional Post-Training

AlignTune includes a supported workflow for adapting and evaluating models on
South Asian languages. The main path uses the tokenization trainer, the shared
corpus loader, and the Indic evaluation registry.

## Tokenizer training

Use `create_tokenization_trainer()` for Indic vocabulary adaptation. It loads
the base tokenizer, reads the corpus through AlignTune's `CorpusLoader`/
`LoaderResolver`, extends the vocabulary, optionally prunes it, and saves a
Hugging Face tokenizer directory.

The two supported extension methods are `continued_bpe` (recommended) and
`naive_extension`. Vocabulary pruning is optional.

```python
from aligntune.data import load_corpus
from aligntune.core.backend_factory import create_tokenization_trainer

# Inspect the same corpus interface used by TokenizationTrainer.
corpus = load_corpus(
    dataset_name="wikimedia/wikipedia",
    config_name="20231101.hi",
    split="train",
    text_column="text",
    max_samples=1000,
)
print(f"Loaded {len(corpus)} Hindi text samples")

tokenizer_trainer = create_tokenization_trainer(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    target_languages=["hi"],
    dataset_name="wikimedia/wikipedia",
    config_name="20231101.hi",
    text_column="text",
    split="train",
    max_samples=1000,
    num_new_tokens=20000,
    extension_method="continued_bpe",  # or "naive_extension"
    output_dir="./out_tokenizer_hindi",
)
result = tokenizer_trainer.train()
print(result["output_dir"])
```

CuratorKIT is used by SFT/RL dataset preparation, but it is not part of the
tokenization trainer's corpus-loading path. The tokenizer trainer uses
`CorpusLoader` and `LoaderResolver` so raw text columns can also come from
local `.txt`, `.json`, `.jsonl`, `.csv`, or `.parquet` files.

## Evaluation

Indic benchmark task registrations live in the `eval/` module and are surfaced through the CLI:

```bash
# Evaluate on all languages and benchmarks
aligntune indic-eval run --model meta-llama/Llama-2-7b

# Evaluate specific languages on a specific benchmark
aligntune indic-eval run --model meta-llama/Llama-2-7b --languages hi,ta --benchmarks milu

# List available tasks
aligntune indic-eval list
```

See the [CLI Commands reference](../cli/commands.md#indic-evaluation-commands) for the full option list.

## See Also

- [Tokenization & Multilingual](tokenization.md)
- [CLI Commands: indic-eval](../cli/commands.md#indic-evaluation-commands)
