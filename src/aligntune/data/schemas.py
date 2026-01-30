from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

class TaskType(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    PPO = "ppo"
    QA = "qa"
    SUMMARIZATION = "summarization"
    CODE = "code"  # ← NEW: For code generation tasks

@dataclass
class TaskSchema:
    required_columns: List[str]
    column_heuristics: Dict[str, List[str]]

# Define the "Normalized" state for each task
TASK_SCHEMAS = {
    TaskType.SFT: TaskSchema(
        required_columns=["prompt", "completion"],
        column_heuristics={
            "prompt": ["instruction", "input", "question", "query", "user_input", "content", "dialogue", "source", "history", "system"],
            "completion": ["output", "answer", "response", "target", "summary", "ground_truth", "text", "destination", "label"]
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
        required_columns=["prompt", "response"], # Reward is often computed, but response is needed for offline GRPO
        column_heuristics={
            "prompt": ["instruction", "input", "question", "history"],
            "response": ["completion", "output", "answer", "target"]
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
    TaskType.SUMMARIZATION: TaskSchema(
        required_columns=["document", "summary"],
        column_heuristics={
            "document": ["text", "article", "content", "input", "source", "context"],
            "summary": ["target", "output", "abstract", "highlights", "response", "completion"]
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
}