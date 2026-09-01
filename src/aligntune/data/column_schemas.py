"""Task-scoped fallback column schemas for CuratorKIT detection failures."""

from __future__ import annotations

from typing import Dict, Iterable, Optional


TASK_FALLBACK_SCHEMAS: dict[str, list[tuple[str, dict[str, str]]]] = {
    "pretraining": [
        ("corpus", {"text": "text"}),
        ("corpus", {"content": "text"}),
        ("corpus", {"document": "text"}),
    ],
    "sft": [
        ("alpaca", {"prompt": "instruction", "response": "output"}),
        ("alpaca", {"instruction": "instruction", "output": "output"}),
        ("alpaca", {"question": "instruction", "answer": "output"}),
        ("sharegpt", {"conversations": "conversation"}),
    ],
    "dpo": [
        (
            "preference",
            {"prompt": "instruction", "chosen": "chosen", "rejected": "rejected"},
        ),
        (
            "preference",
            {
                "question": "instruction",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        ),
        (
            "preference",
            {
                "prompt": "instruction",
                "chosen_response": "chosen",
                "rejected_response": "rejected",
            },
        ),
        (
            "implicit_preference",
            {"chosen": "chosen", "rejected": "rejected"},
        ),
    ],
    "grpo": [
        (
            "grpo",
            {
                "prompt": "instruction",
                "responses": "responses",
                "reward_scores": "reward_scores",
            },
        ),
        ("prompt_only", {"prompt": "instruction"}),
        ("prompt_only", {"question": "instruction"}),
    ],
    "kto": [
        ("unpaired_preference", {"prompt": "instruction", "completion": "output", "label": "label"}),
    ],
    "distillation_offline": [
        ("alpaca", {"prompt": "instruction", "response": "output"}),
        ("alpaca", {"instruction": "instruction", "output": "output"}),
        ("alpaca", {"question": "instruction", "answer": "output"}),
        ("sharegpt", {"conversations": "conversation"}),
    ],
    "distillation_online": [
        ("prompt_only", {"prompt": "instruction"}),
        ("prompt_only", {"question": "instruction"}),
        ("prompt_only", {"instruction": "instruction"}),
    ],
    "distillation_sdft": [
        ("prompt_only", {"prompt": "instruction"}),
        ("prompt_only", {"question": "instruction"}),
        ("prompt_only", {"instruction": "instruction"}),
    ],
    "distillation_sdpo": [
        ("prompt_only", {"prompt": "instruction"}),
        ("prompt_only", {"question": "instruction"}),
        ("prompt_only", {"instruction": "instruction"}),
    ],
}


def find_task_fallback(
    task_type: str, columns: Iterable[str]
) -> Optional[tuple[str, Dict[str, str]]]:
    """Return the first complete fallback schema compatible with ``columns``."""
    available = set(columns)
    candidates = TASK_FALLBACK_SCHEMAS.get(task_type, [])
    for format_name, mapping in candidates:
        if all(source_column in available for source_column in mapping):
            return format_name, mapping
    return None
