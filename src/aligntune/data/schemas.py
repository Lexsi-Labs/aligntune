from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

class TaskType(str, Enum):
    SFT = "sft"
    PRETRAINING = "pretraining"
    DPO = "dpo"
    GRPO = "grpo"
    PPO = "ppo"
    QA = "qa"
    SUMMARIZATION = "summarization"
    CODE = "code"  # For code generation tasks
    KTO = "kto"
    TEXT_CLASSIFICATION = "text_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    VLM_SFT = "vlm_sft"
    INSTRUCTION_FOLLOWING = "instruction_following"
    # Knowledge Distillation
    DISTILLATION_OFFLINE = "distillation_offline"
    DISTILLATION_ONLINE = "distillation_online"
    DISTILLATION_SDFT = "distillation_sdft"
    DISTILLATION_SDPO = "distillation_sdpo"
    # Evolution Strategies
    ES = "es"

@dataclass
class TaskSchema:
    required_columns: List[str]
    column_heuristics: Dict[str, List[str]]

# Define the "Normalized" state for each task
TASK_SCHEMAS = {
    TaskType.PRETRAINING: TaskSchema(
        required_columns=["text"],
        column_heuristics={
            "text": ["text", "output", "content", "document", "article", "body"],
        },
    ),
    TaskType.SFT: TaskSchema(
        required_columns=["prompt", "completion"],
        column_heuristics={
            "prompt": ["instruction", "input", "question", "problem", "query", "user_input", "content", "dialogue", "source", "history", "system"],
            "completion": ["output", "answer", "solution", "response", "target", "summary", "ground_truth", "text", "destination", "label"]
        }
    ),
    # Same schema as SFT: instruction-following datasets use the same prompt/completion
    # shape. Without this entry, ColumnMapper.process() found no schema for this task
    # type and returned datasets unchanged, silently skipping column_mapping entirely.
    TaskType.INSTRUCTION_FOLLOWING: TaskSchema(
        required_columns=["prompt", "completion"],
        column_heuristics={
            "prompt": ["instruction", "input", "question", "problem", "query", "user_input", "content", "dialogue", "source", "history", "system"],
            "completion": ["output", "answer", "solution", "response", "target", "summary", "ground_truth", "text", "destination", "label"]
        }
    ),
    TaskType.DPO: TaskSchema(
        required_columns=["prompt", "chosen", "rejected"],
        column_heuristics={
            "prompt": ["instruction", "input", "question", "history", "system", "user_input"],
            "chosen": ["chosen_response", "better_response", "winner", "response_j", "positive", "good", "chosen_text"],
            "rejected": ["rejected_response", "worse_response", "loser", "response_k", "negative", "bad", "rejected_text"]
        }
    ),
    TaskType.GRPO: TaskSchema(
        required_columns=["prompt", "reference"], # Reward is often computed, reference needed for scoring
        column_heuristics={
            "prompt": ["instruction", "input", "question", "problem", "history"],
            "reference": ["answer", "solution", "output", "completion", "target", "ground_truth", "response"]
        }
    ),
    TaskType.ES: TaskSchema(
        required_columns=["prompt", "reference"], # Same as GRPO - prompt and reference for reward computation
        column_heuristics={
            "prompt": ["instruction", "input", "question", "problem", "history"],
            "reference": ["answer", "solution", "output", "completion", "target", "ground_truth", "response"]
        }
    ),
    TaskType.PPO: TaskSchema(
        required_columns=["prompt"],
        column_heuristics={
            "prompt": ["instruction", "input", "question", "problem", "history"]
        }
    ),
    TaskType.QA: TaskSchema(
        required_columns=["context", "question", "answer"],
        column_heuristics={
            "context": ["document", "passage", "background", "text", "article"],
            "question": ["query", "input", "prompt"],
            "answer": ["output", "response", "label", "ground_truth"]
        }
    ),
    TaskType.KTO: TaskSchema(
        required_columns=["prompt", "completion", "label"],
        column_heuristics={
            "prompt": ["instruction", "input", "question", "history", "system", "user_input"],
            "completion": ["output", "answer", "response", "target"],
            "label": ["kto_tag", "status", "tag", "label"]
        }
    ),
    TaskType.SUMMARIZATION: TaskSchema(
        required_columns=["document", "summary"],
        column_heuristics={
            "document": ["text", "article", "content", "input", "source", "context"],
            "summary": ["target", "output", "abstract", "highlights", "response", "completion"]
        }
    ),
    TaskType.TEXT_CLASSIFICATION: TaskSchema(
        required_columns=["text", "label"],
        column_heuristics={
            "text": ["sentence", "content", "document", "review", "input"],
            "label": ["target", "class", "category", "sentiment"]
        }
    ),
    # ========== NEW: CODE TASK TYPE ==========
    TaskType.CODE: TaskSchema(
        required_columns=["prompt", "test_cases"],
        column_heuristics={
            "prompt": [
                "text",           # MBPP uses 'text'
                "instruction",
                "input",
                "question",
                "description",
                "problem",
                "task"
            ],
            "test_cases": [
                "test_list",      # MBPP uses 'test_list'
                "tests",          # Common alternative
                "test",           # HumanEval uses 'test'
                "test_cases",
                "assertions",
                "examples"
            ],
            # Optional: reference solution (not used for evaluation, but useful for debugging)
            "completion": [
                "code",           # MBPP uses 'code'
                "canonical_solution",  # HumanEval uses this
                "solution",
                "reference",
                "answer"
            ]
        }
    ),
    # ========== END NEW ==========

    # ========== DISTILLATION TASK TYPES ==========
    TaskType.DISTILLATION_OFFLINE: TaskSchema(
        required_columns=["messages"],
        column_heuristics={
            "messages": [
                "prompt",
                "completion",
                "text",
                "conversation",
                "dialogue",
                "conversation_history"
            ]
        }
    ),

    TaskType.DISTILLATION_ONLINE: TaskSchema(
        required_columns=["messages"],
        column_heuristics={
            "messages": [
                "prompt",
                "text",
                "query",
                "question",
                "input",
                "instruction"
            ]
        }
    ),

    TaskType.DISTILLATION_SDFT: TaskSchema(
        required_columns=["prompt", "privileged_context"],
        column_heuristics={
            "prompt": [
                "question",
                "input",
                "instruction",
                "problem",
                "text"
            ],
            "privileged_context": [
                "context",
                "demonstration",
                "hint",
                "feedback",
                "few_shot",
                "example",
                "reference"
            ]
        }
    ),

    TaskType.DISTILLATION_SDPO: TaskSchema(
        required_columns=["prompt", "privileged_context"],
        column_heuristics={
            "prompt": [
                "question",
                "input",
                "instruction",
                "problem",
                "text"
            ],
            "privileged_context": [
                "context",
                "feedback",
                "hint",
                "environment_feedback",
                "explanation",
                "error_message"
            ]
        }
    ),
    # ========== END DISTILLATION ==========
}
