"""
RAFT Trainer: Retrieval Augmented Fine-Tuning for AlignTune.

This module implements RAFT training, which teaches small models to:
1. Use retrieved documents as context for answering questions
2. Distinguish between relevant (golden) and irrelevant (distractor) documents
3. Minimize hallucination by grounding responses in provided sources

RAFT extends the standard SFT trainer with:
- Multi-document context handling (golden + distractors in prompt)
- Citation quality metrics (does model cite the golden doc?)
- Optional auxiliary loss for document ranking
"""

import logging
import inspect
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path

try:
    from trl import SFTTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    SFTTrainer = object

from aligntune.utils.hf_publish import HubPushMixin

logger = logging.getLogger(__name__)


@dataclass
class RaftTrainerConfig:
    """Configuration for RAFT training."""
    # Document context
    max_golden_docs: int = 3
    max_distractor_docs: int = 5
    doc_context_template: str = "[DOC {idx}] {title}: {text}"

    # Loss configuration
    use_citation_loss: bool = True
    citation_loss_weight: float = 0.1
    use_doc_ranking_loss: bool = False
    doc_ranking_loss_weight: float = 0.2

    # Training
    include_doc_ids_in_output: bool = False  # If True, expects [DOC X] in answer
    track_citation_quality: bool = True



def format_raft_example(
    example: Dict[str, Any],
    raft_config: RaftTrainerConfig,
) -> Dict[str, Any]:
    """
    Build the RAFT "text" field (documents + question + answer) for one example.

    Standalone so it can run BEFORE a RaftTrainer exists - trl's SFTTrainer
    tokenizes `train_dataset` eagerly in its own __init__, so callers need to
    format the dataset first (e.g. via `Dataset.map`) and only then construct
    the trainer. Shared by RaftTrainer._prepare_raft_example so the two paths
    can't drift apart.

    Args:
        example: Raw example with question, answer, golden_docs, distractor_docs
        raft_config: RaftTrainerConfig controlling doc limits/template

    Returns:
        Dict with a "text" field ready for SFTTrainer, plus citation metadata
    """
    question = example.get("question", "")
    answer = example.get("answer", "")
    golden_docs = example.get("golden_docs", [])[:raft_config.max_golden_docs]
    distractor_docs = example.get("distractor_docs", [])[:raft_config.max_distractor_docs]

    doc_context_parts = []
    for idx, doc in enumerate(golden_docs, start=1):
        title = doc.get("title", f"Document {idx}")
        text = doc.get("text", "")[:500]
        doc_context_parts.append(
            raft_config.doc_context_template.format(idx=idx, title=title, text=text)
        )
    for idx, doc in enumerate(distractor_docs, start=len(golden_docs) + 1):
        title = doc.get("title", f"Document {idx}")
        text = doc.get("text", "")[:500]
        doc_context_parts.append(
            raft_config.doc_context_template.format(idx=idx, title=title, text=text)
        )
    doc_context = "\n\n".join(doc_context_parts)

    if doc_context:
        full_prompt = f"Context Documents:\n{doc_context}\n\nQuestion: {question}\nAnswer: "
    else:
        full_prompt = f"Question: {question}\nAnswer: "

    return {
        "text": full_prompt + answer,
        "_raft_golden_doc_count": len(golden_docs),
        "_raft_question": question,
        "_raft_answer": answer,
    }


class RaftTrainer(HubPushMixin, SFTTrainer if TRL_AVAILABLE else object):
    """
    RAFT (Retrieval Augmented Fine-Tuning) Trainer.

    Extends SFTTrainer to handle retrieval-augmented examples where the model
    learns to ground its responses in provided documents.

    Expected dataset format:
    {
        "question": str,
        "answer": str,
        "golden_docs": [{"title": str, "text": str}, ...],
        "distractor_docs": [{"title": str, "text": str}, ...],
    }
    """

    _hub_algorithm = "raft"
    _hub_merge_dir = "./out_raft_merged"

    def __init__(self, *args, raft_config: Optional[RaftTrainerConfig] = None, **kwargs):
        """Initialize RAFT trainer."""
        if not TRL_AVAILABLE:
            raise ImportError("TRL is required for RAFT trainer")

        super().__init__(*args, **kwargs)
        self.raft_config = raft_config or RaftTrainerConfig()

        # Metrics tracking
        self.citation_metrics = {
            "citation_quality": 0.0,
            "golden_doc_cited": 0,
            "total_examples": 0,
        }

        logger.info(
            f"Initialized RAFT Trainer with config: "
            f"max_golden={self.raft_config.max_golden_docs}, "
            f"max_distractors={self.raft_config.max_distractor_docs}"
        )

    def _prepare_raft_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a single example by concatenating documents into context.

        Args:
            example: Raw example with question, answer, golden_docs, distractor_docs

        Returns:
            Modified example with documents incorporated into prompt
        """
        example.update(format_raft_example(example, self.raft_config))
        return example

    def _compute_citation_quality(
        self,
        generated_text: str,
        golden_doc_titles: List[str],
    ) -> float:
        """
        Compute citation quality: does the generated text reference golden docs?

        Args:
            generated_text: Model-generated answer
            golden_doc_titles: List of golden document titles

        Returns:
            Citation quality score [0, 1]
        """
        if not golden_doc_titles:
            return 1.0  # No golden docs = perfect (no hallucination)

        text_lower = generated_text.lower()
        cited_count = sum(1 for title in golden_doc_titles if title.lower() in text_lower)

        return min(cited_count / len(golden_doc_titles), 1.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss with RAFT components.

        Args:
            model: The language model
            inputs: Batch inputs
            return_outputs: Whether to return model outputs
            num_items_in_batch: Forwarded to transformers' Trainer.compute_loss
                (added in newer `transformers` releases). NOTE: this param
                was previously missing from the signature entirely - modern
                `transformers.Trainer` (5.13.0 here) unconditionally calls
                `self.compute_loss(model, inputs, return_outputs=True,
                num_items_in_batch=...)`, so every call into this override
                raised `TypeError: compute_loss() got an unexpected keyword
                argument 'num_items_in_batch'` and RAFT training could never
                run at all.

        Returns:
            Loss tensor (and outputs if return_outputs=True)
        """
        # Compute standard SFT loss
        outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss

        # Optional: Add citation loss (TODO: requires generation during training)
        # This is a placeholder for future enhancement
        if self.raft_config.use_citation_loss and self.raft_config.citation_loss_weight > 0:
            # In practice, citation loss would require:
            # 1. Generating outputs from the model
            # 2. Comparing generated text to golden docs
            # For now, we rely on the standard LM loss which implicitly trains citation
            pass

        if return_outputs:
            return loss, outputs
        return loss

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log RAFT-specific metrics."""
        if self.citation_metrics["total_examples"] > 0:
            avg_quality = (
                self.citation_metrics["citation_quality"]
                / self.citation_metrics["total_examples"]
            )
            metrics["raft/citation_quality"] = avg_quality
            metrics["raft/golden_docs_cited"] = (
                self.citation_metrics["golden_doc_cited"]
                / self.citation_metrics["total_examples"]
            )

        super()._log_metrics(metrics)


def raft_trainer_from_config(
    config: Dict[str, Any],
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    **kwargs
) -> RaftTrainer:
    """
    Factory function to create RaftTrainer from config dict.

    Args:
        config: Configuration dictionary
        model: Pretrained model
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        **kwargs: Additional arguments to pass to trainer

    Returns:
        Configured RaftTrainer instance
    """
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    # Extract RAFT config
    raft_cfg = RaftTrainerConfig(
        max_golden_docs=config.get("max_golden_docs", 3),
        max_distractor_docs=config.get("max_distractor_docs", 5),
        doc_context_template=config.get(
            "doc_context_template",
            "[DOC {idx}] {title}: {text}"
        ),
        use_citation_loss=config.get("use_citation_loss", True),
        citation_loss_weight=config.get("citation_loss_weight", 0.1),
    )

    # Build training arguments
    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "./outputs"),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        warmup_steps=config.get("warmup_steps", 500),
        weight_decay=float(config.get("weight_decay", 0.01)),
        logging_steps=config.get("logging_steps", 100),
        eval_strategy=config.get("eval_strategy", "no"),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 500),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        seed=config.get("seed", 42),
        **kwargs
    )

    trainer = RaftTrainer(
        model=model,
        # trl 1.7.1's SFTTrainer.__init__ has no `tokenizer` parameter at all
        # (renamed to `processing_class`); passing `tokenizer=` here raised
        # TypeError: __init__() got an unexpected keyword argument 'tokenizer'.
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        raft_config=raft_cfg,
    )

    return trainer
