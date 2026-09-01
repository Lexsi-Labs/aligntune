"""
Unsloth RAFT Trainer: Retrieval Augmented Fine-Tuning for AlignTune.

This module implements RAFT training, which teaches small models to:
1. Use retrieved documents as context for answering questions
2. Distinguish between relevant (golden) and irrelevant (distractor) documents
3. Minimize hallucination by grounding responses in provided sources

RAFT extends the standard SFT trainer with:
- Multi-document context handling (golden + distractors in prompt)
- Citation quality metrics (does model cite the golden doc?)
- Optional auxiliary loss for document ranking

This is an Unsloth-accelerated port of
`aligntune.backends.trl.raft.raft_trainer.RaftTrainer`. All RAFT-specific
logic (document context construction, prompt formatting, citation metrics,
loss composition) is preserved exactly as in the TRL version - the only
change is that the model backbone is loaded via Unsloth's
`FastLanguageModel` (routed through
`aligntune.core.model_loader.build_model(..., use_unsloth=True)` when the
underlying model loader is invoked, mirroring
`aligntune.backends.unsloth.sft.sft.UnslothSFTTrainer.setup_model`) before
being handed to TRL's `SFTTrainer`.
"""

import logging
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

try:
    import unsloth  # noqa: F401 - import-only availability check
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

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

    Standalone so it can run BEFORE an UnslothRaftTrainer exists - trl's
    SFTTrainer tokenizes `train_dataset` eagerly in its own __init__, so
    callers need to format the dataset first (e.g. via `Dataset.map`) and
    only then construct the trainer. Shared by
    UnslothRaftTrainer._prepare_raft_example so the two paths can't drift
    apart.

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


class UnslothRaftTrainer(HubPushMixin, SFTTrainer if TRL_AVAILABLE else object):
    """
    Unsloth-accelerated RAFT (Retrieval Augmented Fine-Tuning) Trainer.

    Extends TRL's SFTTrainer to handle retrieval-augmented examples where the
    model learns to ground its responses in provided documents. The model
    handed to this trainer is expected to already be an Unsloth-optimized
    model (e.g. loaded via `unsloth_raft_trainer_from_config`, which routes
    loading through `FastLanguageModel.from_pretrained` /
    `core.model_loader.build_model(..., use_unsloth=True)`); the RAFT
    training logic itself is backend-agnostic and identical to the TRL
    version.

    Expected dataset format:
    {
        "question": str,
        "answer": str,
        "golden_docs": [{"title": str, "text": str}, ...],
        "distractor_docs": [{"title": str, "text": str}, ...],
    }
    """

    _hub_algorithm = "raft"
    _hub_backend = "unsloth"
    _hub_merge_dir = "./out_raft_merged"

    def __init__(self, *args, raft_config: Optional[RaftTrainerConfig] = None, **kwargs):
        """Initialize Unsloth RAFT trainer."""
        if not TRL_AVAILABLE:
            raise ImportError("TRL is required for RAFT trainer")
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth is required for UnslothRaftTrainer")

        super().__init__(*args, **kwargs)
        self.raft_config = raft_config or RaftTrainerConfig()

        # Metrics tracking
        self.citation_metrics = {
            "citation_quality": 0.0,
            "golden_doc_cited": 0,
            "total_examples": 0,
        }

        logger.info(
            f"Initialized Unsloth RAFT Trainer with config: "
            f"max_golden={self.raft_config.max_golden_docs}, "
            f"max_distractors={self.raft_config.max_distractor_docs}"
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL and Unsloth are both available."""
        try:
            from trl import SFTTrainer  # noqa: F401
            import unsloth  # noqa: F401
            return True
        except ImportError:
            return False

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


def _build_unsloth_raft_model(
    model_name_or_path: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    dtype: Optional[Any] = None,
):
    """
    Load the RAFT backbone through Unsloth, mirroring the loading path used by
    `aligntune.backends.unsloth.sft.sft.UnslothSFTTrainer.setup_model`, which
    routes through `aligntune.core.model_loader.build_model(...,
    use_unsloth=True)` -> `unsloth.FastLanguageModel.from_pretrained`.

    `unsloth_raft_trainer_from_config` historically received an
    already-instantiated `model`/`tokenizer` pair (same as the TRL
    `raft_trainer_from_config`), so this helper is only invoked when the
    caller does not supply one - existing call sites that pre-load their own
    model are unaffected.

    Args:
        model_name_or_path: Base model to fine-tune (e.g.
            "unsloth/Qwen2.5-0.5B-Instruct")
        max_seq_length: Maximum sequence length for the Unsloth model
        load_in_4bit: Whether to load the base model in 4-bit precision
        dtype: Optional explicit torch dtype (None lets Unsloth auto-detect)

    Returns:
        (model, tokenizer) tuple, with the tokenizer's pad token set.
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is required to build an Unsloth RAFT model")
    if not model_name_or_path:
        raise ValueError(
            "model_name_or_path (or a pre-built model/tokenizer pair) is "
            "required to build an Unsloth RAFT model"
        )

    from unsloth import FastLanguageModel

    logger.info(f"Loading Unsloth RAFT backbone: {model_name_or_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def unsloth_raft_trainer_from_config(
    config: Dict[str, Any],
    model=None,
    tokenizer=None,
    train_dataset=None,
    eval_dataset=None,
    **kwargs
) -> UnslothRaftTrainer:
    """
    Factory function to create UnslothRaftTrainer from config dict.

    Args:
        config: Configuration dictionary. In addition to the keys read by
            the TRL `raft_trainer_from_config`, this reads
            `model_name_or_path` (or `model_name`), `max_seq_length`, and
            `load_in_4bit` to build the Unsloth model when `model`/
            `tokenizer` are not already supplied.
        model: Pretrained (ideally Unsloth-loaded) model. If None, one is
            built via `_build_unsloth_raft_model` (routes through
            `FastLanguageModel.from_pretrained` /
            `core.model_loader.build_model(..., use_unsloth=True)`).
        tokenizer: Tokenizer matching `model`. If None, loaded alongside the
            model.
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        **kwargs: Additional arguments to pass to trainer

    Returns:
        Configured UnslothRaftTrainer instance
    """
    from trl import SFTConfig

    if model is None or tokenizer is None:
        model, tokenizer = _build_unsloth_raft_model(
            model_name_or_path=config.get("model_name_or_path") or config.get("model_name"),
            max_seq_length=config.get("max_seq_length", 2048),
            load_in_4bit=config.get("load_in_4bit", False),
        )

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
    training_args_kwargs = dict(
        output_dir=config.get("output_dir", "./outputs/unsloth_raft"),
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
        run_name=config.get("run_name", "unsloth_raft"),
        # Unsloth's compiled SFTTrainer.__init__ auto-fills args.max_length
        # from the Unsloth-loaded model's own max_seq_length attribute
        # whenever max_length isn't explicitly tracked as set (regardless of
        # what's passed here), then - since it also forces padding_free=True
        # internally - raises "max_length is not enforced" for any non-None
        # max_length when packing=False. Enabling packing satisfies that
        # check via its other branch (`not args.packing` becomes False)
        # instead of fighting Unsloth's max_length auto-fill.
        packing=True,
        packing_strategy="bfd",
    )
    # kwargs takes precedence over the defaults above (e.g. caller-supplied
    # run_name/output_dir), and avoids TrainingArguments(...) raising
    # "got multiple values for keyword argument" if a caller also forwards
    # one of these names through **kwargs.
    training_args_kwargs.update(kwargs)
    # Build SFTConfig directly rather than a plain TrainingArguments: Unsloth's
    # compiled SFTTrainer.__init__ takes whatever args object it's given,
    # round-trips it through vars() into a dict, and reconstructs an
    # SFTConfig from that dict - so a plain TrainingArguments carries fields
    # (e.g. push_to_hub_token) that no longer exist on this trl version's
    # SFTConfig, raising "unexpected keyword argument 'push_to_hub_token'".
    training_args = SFTConfig(**training_args_kwargs)

    trainer = UnslothRaftTrainer(
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
