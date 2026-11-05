"""Helpers shared by RL trainers."""
from typing import Any


def truncate_prompts(dataset: Any, tokenizer: Any, max_prompt_length: int, column: str = "prompt") -> Any:
    """Truncate string prompts to ``max_prompt_length`` tokens.

    Unsloth caps the context at ``max_seq_length``; RL trainers that do not truncate
    prompts themselves (RLOO, Online DPO) crash in generate() when a prompt plus the
    completion budget does not fit. Chat prompts (lists of ``{"role", "content"}`` messages,
    which is what the chat-template path produces) are measured after the template is
    applied and the longest message is cut until the rendered prompt fits.
    """
    if dataset is None or not max_prompt_length or max_prompt_length <= 0:
        return dataset
    if column not in dataset.column_names:
        return dataset

    def _rendered_len(messages):
        try:
            return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        except Exception:
            return sum(len(tokenizer(str(m.get("content", "")), add_special_tokens=False)["input_ids"]) for m in messages)

    def _truncate_chat(messages):
        messages = [dict(m) for m in messages]
        for _ in range(8):
            excess = _rendered_len(messages) - max_prompt_length
            if excess <= 0:
                break
            i = max(range(len(messages)), key=lambda k: len(str(messages[k].get("content", ""))))
            ids = tokenizer(str(messages[i].get("content", "")), add_special_tokens=False)["input_ids"]
            keep = max(1, len(ids) - excess - 1)
            messages[i]["content"] = tokenizer.decode(ids[:keep], skip_special_tokens=True)
        return messages

    def _truncate(example):
        prompt = example[column]
        if isinstance(prompt, list) and prompt and all(isinstance(m, dict) for m in prompt):
            example[column] = _truncate_chat(prompt)
        elif isinstance(prompt, str):
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > max_prompt_length:
                example[column] = tokenizer.decode(ids[:max_prompt_length], skip_special_tokens=True)
        return example

    return dataset.map(_truncate, desc=f"Truncating prompts to {max_prompt_length} tokens")


def drop_rows_without_completion_tokens(dataset):
    """Drop tokenized preference rows where chosen or rejected has no completion token.

    ORPO averages a completion's log-probabilities over its tokens, so a row whose
    completion was cut off entirely (a prompt longer than ``max_length``, or a
    tokenizer boundary that swallows it) divides 0 by 0: the loss becomes NaN, the
    gradients become NaN, and every step after it trains on NaN weights.
    Returns ``(dataset, number_dropped)``.
    """
    def has_tokens(row):
        return all(any(label != -100 for label in row[side + "_labels"][1:]) for side in ("chosen", "rejected"))

    if not {"chosen_labels", "rejected_labels"} <= set(dataset.column_names):
        return dataset, 0
    keep = [i for i, row in enumerate(dataset) if has_tokens(row)]
    return (dataset.select(keep) if len(keep) < len(dataset) else dataset), len(dataset) - len(keep)
