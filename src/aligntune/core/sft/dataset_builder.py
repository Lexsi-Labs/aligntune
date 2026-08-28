"""
Centralized SFT Dataset Builder Utility.

This module handles dataset loading, parsing, chat-templating,
task formatting, packing, and validation for all SFT trainers.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datasets import Dataset, DatasetDict, load_dataset

from .config import SFTConfig, TaskType
from ..dataset_adapters import schema_detector
from ..rl.caching import DatasetCache
from ..rl.registries import DatasetRegistry
from ...data.classification import (
    ClassificationLabelEncoder,
    normalize_text_classification,
    normalize_token_classification,
)

logger = logging.getLogger(__name__)


def _apply_chat_template_safe(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> Optional[str]:
    """Apply tokenizer chat template with broad compatibility."""
    if not tokenizer or not hasattr(tokenizer, "apply_chat_template"):
        return None
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError:
        # Some tokenizers don't support add_generation_prompt or other kwargs.
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            return None
    except Exception:
        return None


def _format_dataset_for_task(
    dataset: Dataset,
    tokenizer: Any,
    task_type: TaskType,
    dataset_config: Any,
    label_encoder: Optional[ClassificationLabelEncoder] = None,
    fit_labels: bool = True,
) -> Tuple[Dataset, Optional[ClassificationLabelEncoder]]:
    """Format dataset using the central schema detector and apply chat templates if needed."""
    logger.info(f"Formatting dataset for task: {task_type.value}")
    
    # Classification has a separate schema from causal-LM SFT. Do this before
    # the generic prompt/completion adapters so labels cannot be discarded.
    if task_type == TaskType.TEXT_CLASSIFICATION:
        normalized, encoder = normalize_text_classification(
            dataset,
            column_mapping=getattr(dataset_config, "column_mapping", None),
            text_column=getattr(dataset_config, "text_column", "text"),
            label_column=getattr(dataset_config, "label_column", "label"),
            encoder=None if fit_labels else label_encoder,
        )
        labels_sample = normalized["labels"][: min(100, len(normalized))]
        if fit_labels and len(set(labels_sample)) < 2:
            raise ValueError("Text classification data must contain at least two label classes.")
        return normalized, encoder

    if task_type == TaskType.TOKEN_CLASSIFICATION:
        normalized, encoder = normalize_token_classification(
            dataset,
            column_mapping=getattr(dataset_config, "column_mapping", None),
            tokens_column=getattr(dataset_config, "tokens_column", "tokens"),
            tags_column=getattr(dataset_config, "tags_column", "ner_tags"),
            encoder=None if fit_labels else label_encoder,
        )
        return normalized, encoder

    # 1. Use adapters to normalize fields to prompt/completion.
    # The adapter automatically handles column_mapping from dataset_config if provided
    custom_mappings = getattr(dataset_config, "column_mapping", None)
    
    # Map task type to the adapter training type
    if task_type == TaskType.TEXT_CLASSIFICATION:
        training_type = "text_classification"
    elif task_type == TaskType.TOKEN_CLASSIFICATION:
        training_type = "token_classification"
    else:
        training_type = "sft"
        
    dataset = schema_detector.adapt_dataset(
        dataset, 
        training_type=training_type,
        custom_mappings=custom_mappings
    )
    
    # 2. If it's a classification task, the adapter already gave us `text`/`tokens` and `labels`. No templating needed.
    if task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
        if len(dataset) > 0 and task_type == TaskType.TEXT_CLASSIFICATION:
            labels_sample = [dataset[i]['labels'] for i in range(min(100, len(dataset)))]
            unique_labels = set(labels_sample)
            logger.info(f"✓ Unique labels: {unique_labels}")
            if len(unique_labels) == 1:
                logger.error(f"❌ ERROR: All labels are {list(unique_labels)[0]}! Dataset is corrupted!")
                raise ValueError("Dataset has only one label class!")
        return dataset, None
        
    # 3. For SFT / Generation / Chat tasks, apply the tokenizer chat template
    can_template = tokenizer is not None and hasattr(tokenizer, "apply_chat_template")
    system_prompt = getattr(dataset_config, "system_prompt", None)
    
    def apply_template(example):
        # Already a conversation (e.g. UltraChat or custom mapped `messages`)
        if "messages" in example and isinstance(example["messages"], list):
            msgs = example["messages"]
            if system_prompt and (not msgs or msgs[0].get("role") != "system"):
                msgs = [{"role": "system", "content": system_prompt}] + msgs
            
            templated = _apply_chat_template_safe(tokenizer, msgs) if can_template else None
            
            if templated:
                return {"text": templated}
            else:
                # Fallback to simple formatting
                return {"text": "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs)}
        
        # Standard prompt/completion (from Alpaca/Dolly/Custom)
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": str(prompt)})
        msgs.append({"role": "assistant", "content": str(completion)})
        
        templated = _apply_chat_template_safe(tokenizer, msgs) if can_template else None
        if templated:
            return {"text": templated}
            
        # Fallback
        return {"text": f"### Instruction:\n{prompt}\n\n### Response:\n{completion}"}

    # Only keep the 'text' column for generation tasks
    formatted_dataset = dataset.map(
        apply_template,
        remove_columns=dataset.column_names,
        desc=f"Applying templates for {task_type.value}"
    )
    
    return formatted_dataset, None


def _apply_document_packing(dataset: Dataset, tokenizer: Any, config: SFTConfig) -> Dataset:
    """Pack short documents into fixed-length sequences for long-context SFT."""
    from ....core.long_context.packing import DocumentPacker

    context_length: int = getattr(config.model, "max_seq_length", 32_768)
    text_column: str = getattr(config.dataset, "text_column", None) or "text"

    logger.info(
        f"Long-context document packing enabled: context_length={context_length}, "
        f"text_column='{text_column}', n_docs={len(dataset)}"
    )

    if text_column not in dataset.column_names:
        logger.warning(f"Column '{text_column}' not found. Skipping packing.")
        return dataset

    packer = DocumentPacker(
        tokenizer=tokenizer,
        context_length=context_length,
        eos_between_docs=True,
    )

    packed_dataset = packer.pack_documents(dataset, text_column=text_column)
    logger.info(f"Packing complete: {len(dataset)} documents -> {len(packed_dataset)} sequences.")
    return packed_dataset


def build_sft_datasets(config: SFTConfig, tokenizer: Any, task_type: TaskType) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load, split, format, and pack datasets based on the SFT config.
    Returns a tuple of (train_dataset, eval_dataset).
    """
    logger.info("=" * 80)
    logger.info(f"Building SFT Datasets: {config.dataset.name}")
    logger.info("=" * 80)
    
    # Initialize cache
    cache_root = getattr(config, 'caching', {}).get("root", "./cache") if hasattr(config, 'caching') else "./cache"
    dataset_cache = DatasetCache(cache_root=cache_root)
    
    dataset_config = config.dataset
    dataset_name = dataset_config.name
    
    # Splitting logic fallback
    target_split = getattr(dataset_config, 'train_split', None) or getattr(dataset_config, 'split', "train")
    
    dataset_subset = None
    if hasattr(dataset_config, 'subset') and dataset_config.subset:
        dataset_subset = dataset_config.subset
    elif hasattr(dataset_config, 'config') and dataset_config.config:
        dataset_subset = dataset_config.config
        
    eval_dataset_raw = None
    
    # Support callers that already loaded a Dataset/DatasetDict. This avoids
    # treating an in-memory object as a filesystem/HF dataset identifier.
    if isinstance(dataset_name, DatasetDict):
        dataset_dict = dataset_name
        if target_split not in dataset_dict:
            raise ValueError(
                f"DatasetDict does not contain requested split '{target_split}'. "
                f"Available splits: {list(dataset_dict.keys())}"
            )
        dataset = dataset_dict[target_split]
        eval_dataset_raw = dataset_dict.get("validation") or dataset_dict.get("test")
    elif isinstance(dataset_name, Dataset):
        dataset = dataset_name
    elif dataset_name in DatasetRegistry.list_loaders():
        logger.info(f"Using registered dataset loader: {dataset_name}")
        dataset = DatasetRegistry.load_dataset(
            name=dataset_name,
            split=target_split,
            max_samples=dataset_config.max_samples,
            **{k: v for k, v in dataset_config.__dict__.items() if k not in ['name', 'split', 'train_split', 'max_samples'] and v is not None}
        )
    else:
        try:
            dataset_dict = load_dataset(dataset_name, dataset_subset) if dataset_subset else load_dataset(dataset_name)
            
            if isinstance(dataset_dict, dict) and target_split in dataset_dict:
                dataset = dataset_dict[target_split]
                eval_dataset_raw = dataset_dict.get("validation") or dataset_dict.get("test")
            else:
                dataset = load_dataset(dataset_name, dataset_subset, split=target_split) if dataset_subset else load_dataset(dataset_name, split=target_split)
        except ValueError as e:
            if "Config name is missing" in str(e):
                raise ValueError(f"Dataset '{dataset_name}' requires a config/subset name.")
            raise

    logger.info(f"Original dataset size: {len(dataset)} samples")

    label_encoder: Optional[ClassificationLabelEncoder] = None

    def process_split(ds: Dataset, is_eval: bool = False) -> Dataset:
        nonlocal label_encoder
        if task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            if not is_eval:
                ds = ds.shuffle(seed=42)
        
        # Size caps
        if dataset_config.percent and dataset_config.percent < 1.0:
            ds_size = int(len(ds) * dataset_config.percent)
            ds = ds.select(range(ds_size))
        elif dataset_config.max_samples:
            cap = max(1000, int(dataset_config.max_samples * 0.2)) if is_eval else dataset_config.max_samples
            ds = ds.select(range(min(cap, len(ds))))
        
        # Formatting
        ds, split_encoder = _format_dataset_for_task(
            ds,
            tokenizer,
            task_type,
            dataset_config,
            label_encoder=label_encoder,
            fit_labels=not is_eval,
        )
        if split_encoder is not None and not is_eval:
            label_encoder = split_encoder

        # Packing
        long_context_packing = (
            getattr(config.train, "long_context_packing", False)
            and task_type not in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]
        )
        if long_context_packing:
            ds = _apply_document_packing(ds, tokenizer, config)
            
        return ds

    train_dataset = process_split(dataset, is_eval=False)

    if eval_dataset_raw:
        logger.info("Found existing validation split.")
        eval_dataset = process_split(eval_dataset_raw, is_eval=True)
    else:
        try:
            eval_percent = getattr(config.train, 'eval_percent', 0.1) if hasattr(config, 'train') else 0.1
            test_size = eval_percent
            if len(train_dataset) < 1000 and len(train_dataset) >= 200:
                test_size = 0.1
            elif len(train_dataset) < 200:
                test_size = min(0.2, max(0.1, 50 / max(1, len(train_dataset))))

            split_ds = train_dataset.train_test_split(test_size=test_size, seed=42, shuffle=True)
            train_dataset = split_ds["train"]
            eval_dataset = split_ds["test"]
            logger.info("Created eval split from train.")
        except Exception as e:
            eval_dataset = None
            logger.warning(f"Could not create eval split: {str(e)}")

    logger.info(f"Dataset Setup Complete -> Train: {len(train_dataset)} | Eval: {len(eval_dataset) if eval_dataset else 0}")
    return train_dataset, eval_dataset
