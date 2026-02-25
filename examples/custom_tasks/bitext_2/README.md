# Bitext Banking

Evaluation task for retail banking chatbot models trained on the Bitext dataset.

## Dataset

- **Path:** `bitext/Bitext-retail-banking-llm-chatbot-training-dataset` (Hugging Face)
- **Columns:** `instruction`, `response` (instruction-following format)
- **Split:** `train`

## Task

- **Task name:** `bitext_2`
- **Type:** `generate_until` (model generates assistant response given user instruction)
- **Prompt:** ChatML-style `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n`
- **Target:** Reference `response` for metric computation
- **Few-shot:** Supported in two ways:
  - **Split-based:** `bitext_2` with `--num_fewshot N` (exemplars drawn from `train`).
  - **Fixed 2-shot:** Task `bitext_2_fewshot` uses two hand-picked banking exemplars from `utils.list_fewshot_samples()` for reproducible 2-shot eval.

## Metrics

Per-document metrics aggregated with `nanmean`:

- **bleu** – BLEU (n-gram overlap)
- **rouge1**, **rouge2**, **rougeL** – ROUGE-1/2/L
- **chrf** – chrF++ (character n-gram F-score), if `evaluate` chrF is available
- **bert_score** – BERTScore F1 (semantic similarity)

## Dependencies

- `evaluate`, `rouge_score`, `bert-score` (or equivalent via `evaluate`)

## Run

```bash
# 0-shot (default)
lm_eval --model hf \
  --model_args pretrained=path/to/sft_model \
  --tasks bitext_2 \
  --batch_size 4

# Few-shot: split-based (exemplars from train)
lm_eval --model hf \
  --model_args pretrained=path/to/sft_model \
  --tasks bitext_2 \
  --num_fewshot 2 \
  --batch_size 4

# Few-shot: fixed 2 exemplars (reproducible)
lm_eval --model hf \
  --model_args pretrained=path/to/sft_model \
  --tasks bitext_2_fewshot \
  --batch_size 4
```

Optional: `--limit N` to cap the number of eval samples.

## Suggestions

1. **Eval split:** The dataset has only `train`. For cleaner metrics, consider creating a small held-out eval set (e.g. `train[:23000]` for train, last 2.5k for eval) via `dataset_kwargs` or a custom split, and set `test_split` to that eval set so you don’t evaluate on the same data you might use for few-shot.

2. **Few-shot:** Keep **split-based** few-shot (`fewshot_split: train`) for flexibility. If you want **fixed, reproducible** exemplars (like mbpp), add `fewshot_config.samples: !function utils.list_fewshot_samples` and implement `list_fewshot_samples()` in `utils.py` returning 2–3 hand-picked `{instruction, response}` dicts with `"is_fewshot": True`.

3. **Compare 0-shot vs few-shot:** Run once with default (0-shot) and once with `--num_fewshot 2` (or 3), same `--limit`, and compare BLEU/ROUGE/BERTScore to see if few-shot helps on this task.

4. **Model path:** Use an **absolute path** for `pretrained=` when running lm_eval (e.g. `/teamspace/studios/this_studio/output/bitext-banking-sft/sft_model`) so it works regardless of current working directory.

5. **Optional description:** To add a system-style instruction before few-shot examples, set `description` in the YAML (e.g. “You are a helpful retail banking assistant.”); it will be prepended to the prompt when using few-shot.
