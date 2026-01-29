"""
TRL SFT Backend Implementation with Task Type Support.

This module provides a pure TRL backend for Supervised Fine-Tuning (SFT),
using TRL's SFTTrainer for reliable and battle-tested training.
Based on the official TRL library patterns and examples.
Includes all logging, debugging, and miscellaneous features from legacy implementation.

Enhanced with comprehensive task type support:
- Instruction Following
- Supervised Fine-Tuning
- Text Generation
- Chat Completion
- Text Classification
- Token Classification
"""

import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from torch.utils.data import DataLoader

from ....core.sft.trainer_base import SFTTrainerBase
from ....core.rl.config import UnifiedConfig
from ....core.rl.registries import DatasetRegistry, RewardRegistry
from ....core.rl.caching import DatasetCache
from ....core.sft.evaluator import EnhancedEvaluator
from ....core.dataset_adapters import schema_detector
from ....core.sft.config import TaskType
from ....core.precision_handler import PrecisionHandler

logger = logging.getLogger(__name__)


class TaskFormatter:
    """Handles task-specific dataset formatting for TRL backend."""
    
    @staticmethod
    def format_instruction_following(examples: Dict, dataset_cfg) -> Dict[str, List[str]]:
        """Format dataset for instruction following with structured prompts."""
        texts = []
        for i in range(len(examples[dataset_cfg.instruction_column])):
            instruction = examples[dataset_cfg.instruction_column][i]
            response = examples[dataset_cfg.response_column][i]
            context = examples.get(dataset_cfg.context_column, [""] * len(examples[dataset_cfg.instruction_column]))[i]
            
            if context:
                text = (
                    f"<|im_start|>system\nContext: {context}<|im_end|>\n"
                    f"<|im_start|>user\n{instruction}<|im_end|>\n"
                    f"<|im_start|>assistant\n{response}<|im_end|>"
                )
            else:
                text = (
                    f"<|im_start|>user\n{instruction}<|im_end|>\n"
                    f"<|im_start|>assistant\n{response}<|im_end|>"
                )
            texts.append(text)
        return {"text": texts}
    
    @staticmethod
    def format_supervised_fine_tuning(examples: Dict, dataset_cfg) -> Dict[str, List[str]]:
        """Format dataset for standard supervised fine-tuning."""
        texts = []
        for i in range(len(examples.get("instruction", examples.get("text", [])))):
            instruction = examples.get("instruction", [""])[i] if "instruction" in examples else ""
            input_text = examples.get("input", [""])[i] if "input" in examples else ""
            output = examples.get("response", examples.get("output", [""]))[i]
            
            if instruction and output:
                if input_text:
                    text = (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Input:\n{input_text}\n\n"
                        f"### Response:\n{output}"
                    )
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            else:
                text = examples.get("text", [""])[i]
            
            texts.append(text)
        return {"text": texts}
    
    @staticmethod
    def format_text_generation(examples: Dict, dataset_cfg) -> Dict[str, List[str]]:
        """Format dataset for text generation (simple text continuation)."""
        texts = []
        text_field = dataset_cfg.text_column if hasattr(dataset_cfg, 'text_column') else "text"
        
        for text in examples.get(text_field, []):
            if text and isinstance(text, str) and text.strip():
                texts.append(text.strip())
        
        if not texts:
            texts = ["[NO_VALID_TEXT]"]
        
        return {"text": texts}
    
    @staticmethod
    def format_chat_completion(examples: Dict, dataset_cfg) -> Dict[str, List[str]]:
        """Format dataset for chat completion with conversation format."""
        texts = []
        messages_field = dataset_cfg.messages_column if hasattr(dataset_cfg, 'messages_column') else "messages"
        
        for messages in examples.get(messages_field, []):
            if isinstance(messages, list):
                conversation = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conversation.append(f"{role}: {content}")
                texts.append("\n".join(conversation))
            else:
                texts.append(str(messages) if messages else "")
        return {"text": texts}
    
    @staticmethod
    def format_classification(examples: Dict, dataset_cfg) -> Dict:
        """Format dataset for text classification."""
        # Get the label column name
        label_col = dataset_cfg.label_column if hasattr(dataset_cfg, 'label_column') else 'label'
        
        # CRITICAL: Create 'labels' (plural) as the output, remove 'label' (singular)
        result = {
            'text': examples[dataset_cfg.text_column],
            'labels': examples[label_col]  # This becomes 'labels'
        }
        
        return result
    
    @staticmethod
    def format_token_classification(examples: Dict, dataset_cfg) -> Dict:
        """Format dataset for token classification (NER, POS tagging)."""
        return {
            'tokens': examples[dataset_cfg.tokens_column],
            'labels': examples[dataset_cfg.tags_column]
        }


class TRLSFTTrainer(SFTTrainerBase):
    """SFT trainer using pure TRL SFTTrainer with comprehensive task type support."""
    
    # All supported task types
    SUPPORTED_TASKS = [
        TaskType.INSTRUCTION_FOLLOWING,
        TaskType.SUPERVISED_FINE_TUNING,
        TaskType.TEXT_GENERATION,
        TaskType.CHAT_COMPLETION,
        TaskType.TEXT_CLASSIFICATION,
        TaskType.TOKEN_CLASSIFICATION
    ]
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task_type = self._get_task_type()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.dataset = None
        self.training_history = []
        self.logging_manager = None
        self.evaluator = None
        self.eval_dataset = None
        self.eval_dataset = None  # Already exists - no need to add
        self.custom_evaluator = None  # ADD THIS LINE (for BaseEvaluator)
        
        # Validate task type
        self._validate_task_type()
        
        logger.info(f"Initialized TRLSFTTrainer for task: {self.task_type.value}")
    
    def _get_task_type(self) -> TaskType:
        """Extract task type from config."""
        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'task_type'):
            task_type = self.config.dataset.task_type
        elif hasattr(self.config, 'train') and hasattr(self.config.train, 'task_type'):
            task_type = self.config.train.task_type
        else:
            task_type = TaskType.SUPERVISED_FINE_TUNING
        
        if isinstance(task_type, str):
            task_type = TaskType(task_type.lower())
        
        return task_type
    
    def _validate_task_type(self):
        """Validate that the task type is supported."""
        if self.task_type not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Task type {self.task_type.value} is not supported by TRL backend. "
                f"Supported tasks: {[t.value for t in self.SUPPORTED_TASKS]}"
            )
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL is available."""
        try:
            from trl import SFTTrainer, SFTConfig, ModelConfig
            from transformers import (
                AutoModelForCausalLM, 
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
                AutoTokenizer
            )
            return True
        except ImportError:
            return False
    
    # Replace lines 465-476 in setup_model:
    def setup_model(self) -> None:
        """Setup model using standard Transformers with task type awareness."""
        logger.info("=" * 80)
        logger.info(f"Setting up TRL SFT model: {self.config.model.name_or_path}")
        logger.info(f"Task Type: {self.task_type.value}")
        logger.info("=" * 80)
        
        # === UNIFIED PRECISION HANDLING ===
        precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
        precision = PrecisionHandler.validate_precision(precision)
        PrecisionHandler.log_precision_info(precision, "TRL SFT")
        dtype = PrecisionHandler.get_torch_dtype(precision)
        
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
                AutoTokenizer
            )
        except ImportError as e:
            raise ImportError("Transformers not available.") from e
        
        # Load tokenizer first
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name_or_path,
            trust_remote_code=True,
        )
    
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad token to eos token")
        
        # Load model based on task type
        logger.info(f"Loading model for task: {self.task_type.value}")
        
        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            num_labels = getattr(self.config.model, 'num_labels', 2)
            logger.info(f"Loading classification model with {num_labels} labels")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model.name_or_path,
                num_labels=num_labels,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        
        elif self.task_type == TaskType.TOKEN_CLASSIFICATION:
            num_labels = getattr(self.config.model, 'num_labels', 9)
            logger.info(f"Loading token classification model with {num_labels} labels")
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model.name_or_path,
                num_labels=num_labels,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        
        else:
            # Causal LM for generation tasks
            logger.info("Loading causal language model")
            
            # Handle quantization configuration
            quantization_config = None
            quantization_dict = getattr(self.config.model, 'quantization', {})
            
            if quantization_dict.get('load_in_4bit', False) or quantization_dict.get('load_in_8bit', False):
                try:
                    from transformers import BitsAndBytesConfig
                    
                    load_in_4bit = quantization_dict.get('load_in_4bit', False)
                    load_in_8bit = quantization_dict.get('load_in_8bit', False)
                    
                    if load_in_4bit:
                        compute_dtype_str = quantization_dict.get('bnb_4bit_compute_dtype', 'bfloat16')
                        compute_dtype = torch.bfloat16 if compute_dtype_str == 'bfloat16' else torch.float16
                        
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_quant_type=quantization_dict.get('bnb_4bit_quant_type', 'nf4'),
                            bnb_4bit_use_double_quant=quantization_dict.get('bnb_4bit_use_double_quant', False),
                        )
                        logger.info("✅ 4-bit quantization enabled for memory optimization")
                    elif load_in_8bit:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        logger.info("✅ 8-bit quantization enabled for memory optimization")
                except ImportError:
                    logger.warning("⚠️  BitsAndBytes not available. Install with: pip install bitsandbytes")
                    quantization_config = None
            
            # Get max_memory from config if specified
            max_memory = getattr(self.config.model, 'max_memory', None)
            
            model_kwargs = {
                'torch_dtype': dtype,
                'trust_remote_code': True,
                'device_map': 'auto' if torch.cuda.is_available() else None,
                'low_cpu_mem_usage': True,
            }
            
            # Add max_memory if specified
            if max_memory:
                model_kwargs['max_memory'] = max_memory
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
                logger.info(f"Quantization config: {quantization_config}")
            else:
                # Only set these if not using quantization
                model_kwargs['load_in_4bit'] = False
                model_kwargs['load_in_8bit'] = False
                logger.info("No quantization configured - loading full precision model")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name_or_path,
                **model_kwargs
            )
        
            # Move to GPU only if not using device_map='auto' (which handles it automatically)
            if not quantization_config and torch.cuda.is_available() and model_kwargs.get('device_map') != 'auto':
                self.model = self.model.cuda()
                logger.info("Model moved to GPU")
            elif quantization_config:
                logger.info("Model loaded with quantization (device_map='auto')")
                # CRITICAL: When using quantization, PEFT adapters MUST be enabled
                # TRL SFTTrainer will handle this, but we need to ensure PEFT is enabled in config
                if not getattr(self.config.model, 'peft_enabled', False):
                    logger.warning("⚠️  Quantization requires PEFT adapters. Enabling PEFT automatically.")
                    self.config.model.peft_enabled = True
        
        # Setup tokenizer for specific tasks
        self._setup_tokenizer_for_task()
        
        logger.info("=" * 80)
        logger.info("TRL SFT model setup completed successfully")
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        logger.info("=" * 80)
    
    def _setup_tokenizer_for_task(self):
        """Setup tokenizer with task-specific configurations."""
        # If a custom chat template (jinja) is provided in config, set it so
        # `tokenizer.apply_chat_template(...)` can be used during SFT formatting.
        cfg_template = getattr(self.config.dataset, "chat_template", None)
        if cfg_template and isinstance(cfg_template, str):
            if ("{%" in cfg_template or "{{" in cfg_template) and (
                not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None
            ):
                try:
                    self.tokenizer.chat_template = cfg_template
                    logger.info("Applied custom chat_template from config to tokenizer")
                except Exception as e:
                    logger.debug(f"Failed to set tokenizer.chat_template from config: {e}")

        if self.task_type == TaskType.CHAT_COMPLETION:
            if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{ message['role'] }}: {{ message['content'] }}\n"
                    "{% endfor %}Assistant:"
                )
                logger.info("Added chat template to tokenizer")
        
        elif self.task_type == TaskType.INSTRUCTION_FOLLOWING:
            # If the tokenizer already supports chat templating, prefer using that
            # instead of introducing ChatML-style tokens.
            has_chat_template = (
                hasattr(self.tokenizer, "apply_chat_template") and
                hasattr(self.tokenizer, "chat_template") and
                self.tokenizer.chat_template is not None
            )
            if not has_chat_template:
                special_tokens = ["<|im_start|>", "<|im_end|>"]
                num_added = self.tokenizer.add_special_tokens({
                    'additional_special_tokens': special_tokens
                })
                if num_added > 0:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    logger.info(f"Added {num_added} special tokens for instruction following")
    
    def _apply_chat_template_safe(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> Optional[str]:
        """Apply tokenizer chat template with broad compatibility."""
        if not self.tokenizer or not hasattr(self.tokenizer, "apply_chat_template"):
            return None
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except TypeError:
            # Some tokenizers don't support add_generation_prompt or other kwargs.
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                return None
        except Exception:
            return None

    def _format_with_chat_template(self, examples: Dict[str, Any], dataset_cfg) -> Dict[str, List[str]]:
        """
        Build a `text` field using the model's chat template when available.
        Falls back to legacy string formatting if templating fails.
        """
        system_prompt = getattr(dataset_cfg, "system_prompt", None)
        texts: List[str] = []

        def _prepend_system_if_missing(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
            if system_prompt and (not msgs or msgs[0].get("role") != "system"):
                return [{"role": "system", "content": system_prompt}] + msgs
            return msgs

        if self.task_type == TaskType.CHAT_COMPLETION:
            messages_field = dataset_cfg.messages_column if hasattr(dataset_cfg, "messages_column") else "messages"
            for messages in examples.get(messages_field, []):
                if isinstance(messages, list):
                    msgs = _prepend_system_if_missing(messages)
                    templated = self._apply_chat_template_safe(msgs, add_generation_prompt=False)
                    if templated is not None:
                        texts.append(templated)
                        continue
                    # Fallback: plain "role: content" formatting
                    conversation = [f"{m.get('role','user')}: {m.get('content','')}" for m in msgs]
                    texts.append("\n".join(conversation))
                else:
                    texts.append(str(messages) if messages else "")
            return {"text": texts}

        if self.task_type in [TaskType.INSTRUCTION_FOLLOWING, TaskType.SUPERVISED_FINE_TUNING]:
            # Determine columns (prefer config, fall back to common names)
            instr_col = getattr(dataset_cfg, "instruction_column", "instruction")
            resp_col = getattr(dataset_cfg, "response_column", "response")
            ctx_col = getattr(dataset_cfg, "context_column", "context")
            input_col = getattr(dataset_cfg, "input_column", "input")

            instructions = examples.get(instr_col, examples.get("instruction", []))
            # response could be in response/output/completion depending on dataset
            responses = examples.get(resp_col, examples.get("response", examples.get("output", [])))

            n = min(len(instructions), len(responses))
            for i in range(n):
                instruction = instructions[i]
                response = responses[i]
                context = ""
                if ctx_col in examples and i < len(examples.get(ctx_col, [])):
                    context = examples.get(ctx_col, [""])[i] or ""
                input_text = ""
                if input_col in examples and i < len(examples.get(input_col, [])):
                    input_text = examples.get(input_col, [""])[i] or ""

                sys_parts: List[str] = []
                if system_prompt:
                    sys_parts.append(str(system_prompt).strip())
                if context and str(context).strip():
                    sys_parts.append(f"Context: {str(context).strip()}")

                messages: List[Dict[str, str]] = []
                if sys_parts:
                    messages.append({"role": "system", "content": "\n\n".join(sys_parts)})

                user_content = str(instruction) if instruction is not None else ""
                if input_text and str(input_text).strip():
                    user_content = f"{user_content}\n\n{str(input_text).strip()}"
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": str(response) if response is not None else ""})

                templated = self._apply_chat_template_safe(messages, add_generation_prompt=False)
                if templated is not None:
                    texts.append(templated)
                else:
                    # Fallback to legacy formatting (inline; avoid recomputing full batch)
                    if self.task_type == TaskType.INSTRUCTION_FOLLOWING:
                        if context and str(context).strip():
                            texts.append(
                                f"<|im_start|>system\nContext: {str(context).strip()}<|im_end|>\n"
                                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                                f"<|im_start|>assistant\n{str(response) if response is not None else ''}<|im_end|>"
                            )
                        else:
                            texts.append(
                                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                                f"<|im_start|>assistant\n{str(response) if response is not None else ''}<|im_end|>"
                            )
                    else:
                        if user_content and response is not None:
                            texts.append(f"### Instruction:\n{user_content}\n\n### Response:\n{str(response)}")
                        else:
                            texts.append(str(examples.get("text", [""])[i]) if i < len(examples.get("text", [""])) else "")

            if not texts:
                texts = ["[NO_VALID_TEXT]"]
            return {"text": texts}

        # Default fallback (shouldn't be hit for tasks we call this for)
        return {"text": examples.get("text", ["[NO_VALID_TEXT]"])}
    
    def setup_data(self) -> None:
        """Setup datasets for SFT training with task-aware formatting."""
        logger.info("=" * 80)
        logger.info(f"Setting up TRL SFT datasets: {self.config.dataset.name}")
        logger.info(f"Task Type: {self.task_type.value}")
        logger.info("=" * 80)
        
        # Initialize dataset cache
        cache_root = getattr(self.config, 'caching', {}).get("root", "./cache") if hasattr(self.config, 'caching') else "./cache"
        self.dataset_cache = DatasetCache(cache_root=cache_root)
        logger.info(f"Dataset cache initialized at: {self.dataset_cache.cache_root}")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset_config = self.config.dataset
        
        from datasets import load_dataset
        
        # Handle dataset config/subset
        load_kwargs = {'split': dataset_config.split or "train"}
        dataset_name = dataset_config.name
        dataset_subset = None
        
        if hasattr(dataset_config, 'subset') and dataset_config.subset:
            dataset_subset = dataset_config.subset
        elif hasattr(dataset_config, 'config') and dataset_config.config:
            dataset_subset = dataset_config.config
        
        # Check if dataset has a registered custom loader
        if dataset_name in DatasetRegistry.list_loaders():
            logger.info(f"Using registered dataset loader: {dataset_name}")
            dataset = DatasetRegistry.load_dataset(
                name=dataset_name,
                split=dataset_config.split or "train",
                max_samples=dataset_config.max_samples,
                **{k: v for k, v in dataset_config.__dict__.items() if k not in ['name', 'split', 'max_samples'] and v is not None}
            )
        else:
            # Load dataset
            try:
                if dataset_subset:
                    dataset = load_dataset(dataset_name, dataset_subset, **load_kwargs)
                else:
                    dataset = load_dataset(dataset_name, **load_kwargs)
            except ValueError as e:
                if "Config name is missing" in str(e):
                    raise ValueError(
                        f"Dataset '{dataset_name}' requires a config/subset name. "
                        f"Add 'subset' or 'config' to your DatasetConfig."
                    )
                raise
        
        logger.info(f"Original dataset size: {len(dataset)} samples")
        
        # 🔹 CRITICAL FIX: Shuffle dataset BEFORE selecting samples
        # This ensures balanced class distribution for classification tasks
        if self.task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            logger.info("Shuffling dataset to ensure balanced sampling...")
            dataset = dataset.shuffle(seed=42)
        
        # Apply dataset size limits
        if dataset_config.percent and dataset_config.percent < 1.0:
            dataset_size = int(len(dataset) * dataset_config.percent)
            dataset = dataset.select(range(dataset_size))
            logger.info(f"Using {dataset_config.percent*100}% of dataset: {dataset_size} samples")
        elif dataset_config.max_samples:
            dataset = dataset.select(range(min(dataset_config.max_samples, len(dataset))))
            logger.info(f"Using max_samples limit: {len(dataset)} samples")
        
        # Apply column mappings if needed
        if hasattr(dataset_config, 'column_mapping') and dataset_config.column_mapping:
            valid_mappings = {k: v for k, v in dataset_config.column_mapping.items() if k in dataset.column_names}
            if valid_mappings:
                dataset = dataset.rename_columns(valid_mappings)
                logger.info(f"Renamed columns: {valid_mappings}")
        
        # Task-specific formatting
        dataset = self._format_dataset_for_task(dataset, dataset_config)
        
        # Create evaluation split
        try:
            test_size = 0.1
            if len(dataset) >= 1000:
                test_size = 0.1
            elif len(dataset) >= 200:
                test_size = 0.1
            else:
                test_size = min(0.2, max(0.1, 50 / max(1, len(dataset))))

            split_ds = dataset.train_test_split(test_size=test_size, seed=42, shuffle=True)
            self.dataset = split_ds["train"]
            self.eval_dataset = split_ds["test"]
            logger.info(f"Created eval split. Train: {len(self.dataset)} | Eval: {len(self.eval_dataset)}")
        except Exception as e:
            self.dataset = dataset
            self.eval_dataset = None
            logger.warning(f"Could not create eval split: {str(e)}")
        
        logger.info("=" * 80)
        logger.info(f"TRL SFT dataset setup completed: {len(self.dataset)} samples")
        logger.info("=" * 80)
    
    def _format_dataset_for_task(self, dataset, dataset_config):
        """Format dataset based on task type."""
        logger.info(f"Formatting dataset for task: {self.task_type.value}")
        
        # Select appropriate formatter
        can_template = self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template")

        if self.task_type == TaskType.INSTRUCTION_FOLLOWING:
            formatter = self._format_with_chat_template if can_template else TaskFormatter.format_instruction_following
        elif self.task_type == TaskType.SUPERVISED_FINE_TUNING:
            formatter = self._format_with_chat_template if can_template else TaskFormatter.format_supervised_fine_tuning
        elif self.task_type == TaskType.TEXT_GENERATION:
            formatter = TaskFormatter.format_text_generation
        elif self.task_type == TaskType.CHAT_COMPLETION:
            formatter = self._format_with_chat_template if can_template else TaskFormatter.format_chat_completion
        elif self.task_type == TaskType.TEXT_CLASSIFICATION:
            formatter = TaskFormatter.format_classification
        elif self.task_type == TaskType.TOKEN_CLASSIFICATION:
            formatter = TaskFormatter.format_token_classification
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # Apply formatting
        try:
            # For classification, we need to keep the columns AND rename properly
            if self.task_type == TaskType.TEXT_CLASSIFICATION:
                # Don't remove columns, just map them
                formatted_dataset = dataset.map(
                    lambda examples: formatter(examples, dataset_config),
                    batched=True,
                    remove_columns=[col for col in dataset.column_names if col not in ['text', 'label', 'labels']],
                    desc=f"Formatting for {self.task_type.value}"
                )
                
                # NOW remove the old 'label' column if it exists (keep only 'labels')
                if 'label' in formatted_dataset.column_names and 'labels' in formatted_dataset.column_names:
                    formatted_dataset = formatted_dataset.remove_columns(['label'])
                    logger.info("Removed duplicate 'label' column, keeping 'labels'")
            
            elif self.task_type == TaskType.TOKEN_CLASSIFICATION:
                formatted_dataset = dataset.map(
                    lambda examples: formatter(examples, dataset_config),
                    batched=True,
                    remove_columns=[],
                    desc=f"Formatting for {self.task_type.value}"
                )
            
            else:
                # For generation tasks, remove all columns
                formatted_dataset = dataset.map(
                    lambda examples: formatter(examples, dataset_config),
                    batched=True,
                    remove_columns=dataset.column_names,
                    desc=f"Formatting for {self.task_type.value}"
                )
            
            # Verify
            sample = formatted_dataset[0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            if 'text' in sample:
                logger.info(f"Sample text (first 200 chars): {sample['text'][:200]}")
            if 'labels' in sample:
                logger.info(f"Sample label: {sample['labels']}")
            
            # CRITICAL: For classification, check label distribution
            if self.task_type == TaskType.TEXT_CLASSIFICATION:
                labels_sample = [formatted_dataset[i]['labels'] for i in range(min(100, len(formatted_dataset)))]
                unique_labels = set(labels_sample)
                label_counts = {label: labels_sample.count(label) for label in unique_labels}
                logger.info(f"✓ Label distribution (first 100): {label_counts}")
                logger.info(f"✓ Unique labels: {unique_labels}")
                
                if len(unique_labels) == 1:
                    logger.error(f"❌ ERROR: All labels are {list(unique_labels)[0]}! Dataset is corrupted!")
                    raise ValueError("Dataset has only one label class!")
            
            return formatted_dataset
            
        except Exception as e:
            logger.error(f"Error formatting dataset: {e}")
            raise
    
    def setup_rewards(self) -> None:
        """SFT doesn't use reward functions."""
        logger.info("SFT training doesn't use reward functions")
        self.reward_functions = []
    
    def train(self) -> Dict[str, Any]:
        """Train using TRL SFTTrainer with task-aware setup."""
        logger.info(f"Starting TRL SFT training for task: {self.task_type.value}")
        
        # Setup model and data if not already done
        if self.model is None:
            self.setup_model()
        if self.dataset is None:
            self.setup_data()
        
        try:
            from trl import SFTTrainer
            from transformers import TrainingArguments, Trainer
            from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification
            import evaluate
        except ImportError as e:
            raise ImportError("TRL required for SFT training. Install with: pip install trl") from e
        
        # Create training arguments
        # Get optimizer and scheduler configurations
        from ....core.optimization import get_optimizer_for_config, get_scheduler_for_config
        # === UNIFIED PRECISION HANDLING ===
        precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
        precision_args = PrecisionHandler.get_training_args_precision(precision)

        # Use config-specified optimizer or default to adamw_torch
        optimizer_name = getattr(self.config.train, 'optimizer', 'adamw_torch')
        scheduler_name = getattr(self.config.train, 'lr_scheduler', 'cosine')

        # Create optimizer configuration
        optimizer_config = get_optimizer_for_config(
            optimizer_name,
            self.config.train.learning_rate or 2e-4,
            self.config.train.weight_decay or 0.01
        )

        # Calculate max_steps - use config value or calculate from epochs
        max_steps = self.config.train.max_steps
        if max_steps is None:
            # Calculate from epochs if max_steps not specified
            epochs = getattr(self.config.train, 'epochs', 3) or 3
            dataset_size = len(self.dataset) if hasattr(self, 'dataset') and self.dataset else 1000
            batch_size = self.config.train.per_device_batch_size or 2
            grad_accum = self.config.train.gradient_accumulation_steps or 4
            max_steps = (dataset_size * epochs) // (batch_size * grad_accum)
            max_steps = max(max_steps, 10)  # Minimum 10 steps
            logger.info(f"Calculated max_steps={max_steps} from {epochs} epochs, {dataset_size} samples")

        # Calculate warmup steps
        warmup_steps = getattr(self.config.train, 'warmup_steps', 0)
        if hasattr(self.config.train, 'warmup_ratio') and self.config.train.warmup_ratio:
            warmup_steps = int(max_steps * self.config.train.warmup_ratio)

        # Create scheduler configuration
        scheduler_config = get_scheduler_for_config(
            scheduler_name,
            max_steps,
            warmup_steps
        )
        
        def kwargs_to_str(kwargs_dict):
            """Convert optimizer/scheduler kwargs dict to string format."""
            # Exclude lr and weight_decay since TrainingArguments already sets them
            filtered = {k: v for k, v in kwargs_dict.items() if k not in ['lr', 'learning_rate', 'weight_decay']}
            if not filtered:
                return None
            return ",".join(f"{k}={f'({','.join(map(str, v))})' if isinstance(v, tuple) else v}" for k, v in filtered.items())
        
        optim_args_str=kwargs_to_str(optimizer_config['optimizer_kwargs'])
        # print(lr_scheduler_kwargs_str)
        # print(scheduler_config)
        filtered_scheduler_kwargs = {
            k: v for k, v in scheduler_config['lr_scheduler_kwargs'].items() 
            if k not in ['num_training_steps', 'num_warmup_steps']
        }

        # Build TrainingArguments kwargs, conditionally including optim_args
        training_kwargs = {
            "output_dir": self.config.logging.output_dir,
            "max_steps": max_steps,
            "per_device_train_batch_size": self.config.train.per_device_batch_size or 2,
            "gradient_accumulation_steps": self.config.train.gradient_accumulation_steps or 4,
            "learning_rate": self.config.train.learning_rate or 2e-4,
            "weight_decay": self.config.train.weight_decay or 0.01,
            "warmup_steps": warmup_steps,
            "logging_steps": getattr(self.config.logging, 'log_interval', 10),
            "save_steps": getattr(self.config.train, 'save_interval', 500),
            "eval_steps": getattr(self.config.train, 'eval_interval', 500),
            "eval_strategy": "no",
            "save_strategy": "steps",
            "load_best_model_at_end": False,
            "report_to": [],
            "run_name": self.config.logging.run_name,
            "seed": 42,
            **precision_args,
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "optim": optimizer_name,
            "lr_scheduler_type": scheduler_name,
            "lr_scheduler_kwargs": filtered_scheduler_kwargs,
        }
        
        # Only include optim_args if it's not None/empty and valid.
        # For 8-bit optimizers (adamw_8bit, paged_adamw_8bit, etc.), bitsandbytes handles kwargs internally,
        # so we skip optim_args to avoid Transformers parsing issues.
        # Also skip if the string doesn't contain "=" (would cause parsing errors).
        should_skip_optim_args = (
            not optim_args_str or  # None or empty string
            not optim_args_str.strip() or  # Whitespace only
            "=" not in optim_args_str or  # Invalid format (no key=value pairs)
            any(x in optimizer_name.lower() for x in ["8bit", "8_bit", "bnb", "paged"])  # 8-bit optimizers
        )
        if not should_skip_optim_args:
            training_kwargs["optim_args"] = optim_args_str
        
        training_args = TrainingArguments(**training_kwargs)
        
        # Create trainer based on task type
        if self.task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            self._setup_classification_trainer(training_args)
        else:
            # Generation tasks use SFTTrainer
            # Setup PEFT config if enabled (required for quantized models)
            peft_config = None
            if getattr(self.config.model, 'peft_enabled', False):
                try:
                    from peft import LoraConfig
                    
                    # Get LoRA parameters from config
                    lora_r = getattr(self.config.model, 'lora_rank', 16)
                    lora_alpha = getattr(self.config.model, 'lora_alpha', 32)
                    lora_dropout = getattr(self.config.model, 'lora_dropout', 0.1)
                    target_modules = getattr(self.config.model, 'target_modules', None)
                    
                    # If target_modules not specified, use default for Llama models
                    if target_modules is None:
                        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    
                    peft_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    logger.info(f"✅ PEFT LoRA config created: r={lora_r}, alpha={lora_alpha}, modules={target_modules}")
                except ImportError:
                    logger.warning("⚠️  PEFT not available. Install with: pip install peft")
                    peft_config = None
            
            # Generation tasks use SFTTrainer
            self.trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
                peft_config=peft_config,  # Pass PEFT config to SFTTrainer
            )
        
        # Record training start
        start_time = time.time()
        logger.info("=" * 80)
        logger.info(f"Starting TRL SFT Training - Task: {self.task_type.value}")
        logger.info("=" * 80)
        
        # Start training
        train_result = self.trainer.train()
        
        # Record training end
        end_time = time.time()
        training_duration = end_time - start_time
        
        logger.info(f"Training completed in {training_duration:.2f} seconds")
        
        # Log training metrics
        if self.logging_manager and hasattr(train_result, 'metrics'):
            self.logging_manager.log_metrics(train_result.metrics)
        
        # Save model
        self.trainer.save_model()
        
        # Add to training history
        self.training_history.append({
            'timestamp': time.time(),
            'duration': training_duration,
            'steps': getattr(train_result, 'global_step', 0),
            'task_type': self.task_type.value,
            'model_path': self.config.logging.output_dir,
        })
        
        logger.info("=" * 80)
        logger.info("TRL SFT training completed successfully!")
        logger.info("=" * 80)
        
        # Return training results
        return {
            'training_time': training_duration,
            'final_loss': getattr(train_result, 'train_loss', 0.0),
            'model_path': self.config.logging.output_dir,
            'steps': getattr(train_result, 'global_step', 0),
            'epochs': getattr(train_result, 'epoch', 0),
            'metrics': getattr(train_result, 'metrics', {}),
            'task_type': self.task_type.value
        }
    
    def _setup_classification_trainer(self, training_args):
        """Setup trainer for classification tasks - FIXED VERSION."""
        from transformers import Trainer, DataCollatorWithPadding
        import evaluate
        
        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            # CRITICAL: Verify dataset has both text and labels BEFORE tokenization
            logger.info(f"Dataset columns before tokenization: {self.dataset.column_names}")
            logger.info(f"Sample before tokenization: {self.dataset[0]}")
            
            # Ensure we have the required columns
            if 'text' not in self.dataset.column_names or 'labels' not in self.dataset.column_names:
                raise ValueError(
                    f"Dataset must have 'text' and 'labels' columns. "
                    f"Found: {self.dataset.column_names}"
                )
            
            def tokenize_function(examples):
                """Tokenize text and preserve labels - FIXED"""
                # Get texts - handle both list and single values
                texts = examples['text']
                if not isinstance(texts, list):
                    texts = [texts]
                
                # Clean texts
                texts = [str(t) if t is not None else "" for t in texts]
                
                # Tokenize WITHOUT max_length padding here
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=False,  # FIX: Don't pad here, let collator handle it
                    max_length=self.config.model.max_seq_length,
                    return_tensors=None
                )
                
                # Get labels - handle both list and single values
                labels = examples['labels']
                if not isinstance(labels, list):
                    labels = [labels]
                
                # FIX: Ensure labels are integers and valid
                try:
                    tokenized['labels'] = [int(label) for label in labels]
                except (ValueError, TypeError) as e:
                    logger.error(f"Label conversion error: {e}")
                    logger.error(f"Problematic labels: {labels}")
                    raise
                
                return tokenized
            
            # Store original column names
            original_train_cols = self.dataset.column_names.copy()
            
            # Tokenize train dataset - REMOVE original columns
            logger.info("Tokenizing train dataset...")
            self.dataset = self.dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=original_train_cols,
                desc="Tokenizing train dataset"
            )
            
            # Tokenize eval dataset if it exists
            if self.eval_dataset:
                original_eval_cols = self.eval_dataset.column_names.copy()
                logger.info("Tokenizing eval dataset...")
                self.eval_dataset = self.eval_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=original_eval_cols,
                    desc="Tokenizing eval dataset"
                )
            
            # Verify tokenization
            logger.info(f"✓ Train dataset columns after tokenization: {self.dataset.column_names}")
            sample = self.dataset[0]
            logger.info(f"✓ Sample keys: {list(sample.keys())}")
            logger.info(f"✓ Sample input_ids length: {len(sample['input_ids'])}")
            logger.info(f"✓ Sample label: {sample['labels']}")
            logger.info(f"✓ Sample label type: {type(sample['labels'])}")
            
            # FIX: Verify label distribution and validity
            sample_size = min(100, len(self.dataset))
            all_labels = [self.dataset[i]['labels'] for i in range(sample_size)]
            unique_labels = set(all_labels)
            label_counts = {label: all_labels.count(label) for label in unique_labels}
            
            logger.info(f"✓ First {sample_size} labels distribution: {label_counts}")
            logger.info(f"✓ Unique labels: {unique_labels}")
            logger.info(f"✓ Expected labels: {set(range(self.config.model.num_labels))}")
            
            # Validate labels
            expected_labels = set(range(self.config.model.num_labels))
            if not unique_labels.issubset(expected_labels):
                raise ValueError(
                    f"Found unexpected labels: {unique_labels - expected_labels}. "
                    f"Expected labels in range [0, {self.config.model.num_labels})"
                )
            
            if len(unique_labels) == 1:
                logger.error(f"❌ ERROR: All labels are {list(unique_labels)[0]}!")
                raise ValueError("Dataset has only one label class after tokenization!")
            
            # FIX: Use DataCollatorWithPadding that handles padding dynamically
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,  # Pad dynamically to longest in batch
                max_length=self.config.model.max_seq_length,
                pad_to_multiple_of=8  # Optimize for GPU
            )
            
            logger.info("✓ Data collator created with dynamic padding")
            
            def compute_metrics(eval_pred):
                """Compute classification metrics"""
                predictions, labels = eval_pred
                
                # Handle logits
                if len(predictions.shape) > 1:
                    predictions = np.argmax(predictions, axis=1)
                
                # Log predictions distribution
                unique_preds = np.unique(predictions)
                unique_labels = np.unique(labels)
                logger.info(f"Predictions distribution: {np.bincount(predictions)}")
                logger.info(f"Labels distribution: {np.bincount(labels.astype(int))}")
                
                metrics = {}
                
                # Accuracy
                accuracy_metric = evaluate.load("accuracy")
                metrics.update(accuracy_metric.compute(
                    predictions=predictions,
                    references=labels
                ))
                
                # F1 score
                try:
                    f1_metric = evaluate.load("f1")
                    metrics.update(f1_metric.compute(
                        predictions=predictions,
                        references=labels,
                        average='weighted'
                    ))
                except Exception as e:
                    logger.warning(f"Could not compute F1: {e}")
                
                return metrics
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
            
            logger.info("✓ Classification trainer created successfully")


        elif self.task_type == TaskType.TOKEN_CLASSIFICATION:
            # Tokenize for token classification
            def tokenize_and_align_labels(examples):
                tokenized = self.tokenizer(
                    examples['tokens'],
                    truncation=True,
                    is_split_into_words=True,
                    max_length=self.config.model.max_seq_length
                )
                
                labels = []
                for i, label in enumerate(examples['labels']):
                    word_ids = tokenized.word_ids(batch_index=i)
                    label_ids = []
                    previous_word_idx = None
                    
                    for word_idx in word_ids:
                        if word_idx is None:
                            label_ids.append(-100)
                        elif word_idx != previous_word_idx:
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(-100)
                        previous_word_idx = word_idx
                    
                    labels.append(label_ids)
                
                tokenized["labels"] = labels
                return tokenized
            
            self.dataset = self.dataset.map(tokenize_and_align_labels, batched=True)
            if self.eval_dataset:
                self.eval_dataset = self.eval_dataset.map(tokenize_and_align_labels, batched=True)
            
            data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
            
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=2)
                
                true_predictions = [
                    [p for (p, l) in zip(pred, label) if l != -100]
                    for pred, label in zip(predictions, labels)
                ]
                true_labels = [
                    [l for (p, l) in zip(pred, label) if l != -100]
                    for pred, label in zip(predictions, labels)
                ]
                
                flat_preds = [item for sublist in true_predictions for item in sublist]
                flat_labels = [item for sublist in true_labels for item in sublist]
                
                accuracy = evaluate.load("accuracy")
                return accuracy.compute(predictions=flat_preds, references=flat_labels)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    
    def save_model(self, output_dir: Optional[str] = None) -> str:
        """Save the trained model and tokenizer.
        
        Args:
            output_dir: Directory to save to (defaults to config output_dir)
        
        Returns:
            Path where model was saved
        """
        if output_dir is None:
            output_dir = Path(self.config.logging.output_dir) / "sft_model"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.trainer.save_model(output_dir)
            logger.info(f"Model saved to {output_dir}")
        else:
            logger.warning("No trainer available to save model")
        
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to {output_dir}")
        else:
            logger.warning("No tokenizer available to save")
        
        # Save task type info
        config_path = output_dir / "training_config.yaml"
        config_dict = {
            'task_type': self.task_type.value,
            'model_name': self.config.model.name_or_path
        }
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
        logger.info(f"Training config saved to {config_path}")
        
        # Store for push_to_hub
        self._last_save_path = str(output_dir)
        
        return str(output_dir)
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute training step (handled by TRL trainer)."""
        return {"loss": 0.0}
    
    def create_data_loader(self) -> DataLoader:
        """Create data loader for training."""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call setup_data() first.")
        
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self.dataset,
            batch_size=self.config.train.per_device_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Collate function for data loader."""
        if not batch:
            return {}
        
        # Extract text from batch
        texts = []
        for item in batch:
            if isinstance(item, dict):
                if "text" in item:
                    texts.append(item["text"])
                elif "input_ids" in item:
                    return item
                else:
                    for key, value in item.items():
                        if isinstance(value, str) and len(value.strip()) > 0:
                            texts.append(value)
                            break
            elif isinstance(item, str):
                texts.append(item)
        
        if not texts:
            raise ValueError("No valid text found in batch")
        
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.model.max_seq_length,
            return_tensors="pt"
        )
    
    def get_sample_outputs(self) -> list:
        """Get sample outputs for logging."""
        if not hasattr(self, 'dataset') or self.dataset is None:
            return []
        
        # Classification tasks don't generate samples
        if self.task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            return []
        
        sample_outputs = []
        for i in range(min(3, len(self.dataset))):
            sample = self.dataset[i]
            
            if "text" in sample:
                text = sample["text"]
            elif "messages" in sample:
                messages = sample["messages"]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                text = str(sample.get("input", ""))
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=getattr(self.config.train, 'temperature', 0.7),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            sample_outputs.append({
                "input": text[:100] + "..." if len(text) > 100 else text,
                "output": response,
                "model": "trl_sft",
                "task_type": self.task_type.value
            })
        
        return sample_outputs
    
    def cleanup(self) -> None:
        """Cleanup TRL resources."""
        super().cleanup()
        
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        
        logger.info("TRL SFT trainer cleanup completed")
    
    def evaluate(self, eval_config: Optional[Dict] = None,eval_dataset=None) -> Dict[str, Any]:
        """Evaluate the model performance with enhanced metrics."""
        logger.info(f"Running TRL SFT evaluation for task: {self.task_type.value}")
        
        if not self.trainer:
            raise ValueError("Training not setup. Call train() first.")
        
        if eval_dataset:
            self.eval_dataset=eval_dataset
        
        # Run evaluation
        eval_results = {}
        if hasattr(self.trainer, 'evaluate'):
            if getattr(self, 'eval_dataset', None) is not None:
                eval_results = self.trainer.evaluate()
            else:
                logger.warning("No eval_dataset available; skipping evaluation.")
            
            if self.logging_manager:
                self.logging_manager.log_metrics(eval_results)
        
        # Add task type to results
        eval_results['task_type'] = self.task_type.value
        
        # Calculate perplexity from loss (if available)
        if "eval_loss" in eval_results:
            try:
                eval_results["eval_perplexity"] = float(
                    torch.exp(torch.tensor(float(eval_results["eval_loss"]))).item()
                )
            except Exception as e:
                logger.debug(f"Could not calculate perplexity: {e}")
        
        # Add comprehensive quality metrics using SFTEvaluator for generation tasks
        if self.task_type in [TaskType.INSTRUCTION_FOLLOWING, TaskType.TEXT_GENERATION, TaskType.CHAT_COMPLETION]:
            try:
                from aligntune.core.sft.evaluator import SFTEvaluator
                
                if self.model and self.tokenizer and hasattr(self, 'eval_dataset') and self.eval_dataset:
                    evaluator = SFTEvaluator(self.config)
                    quality_metrics = evaluator.evaluate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        dataset=self.eval_dataset,
                        config=self.config
                    )
                    
                    # Merge quality metrics (avoid duplicates)
                    for key, value in quality_metrics.items():
                        if key not in eval_results:  # Don't override existing metrics
                            eval_results[key] = value
                    
                    logger.info(f"Added {len(quality_metrics)} quality metrics")
            except Exception as e:
                logger.warning(f"Could not compute quality metrics: {e}")
        
        # Generate qualitative samples for generation tasks
        if self.task_type not in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            try:
                samples = self._generate_samples()
                eval_results['qualitative_samples'] = samples
            except Exception as e:
                logger.warning(f"Could not generate qualitative samples: {str(e)}")
                eval_results['qualitative_samples'] = []
        
        logger.info(f"TRL SFT evaluation completed: {len(eval_results)} metrics computed")
        return eval_results
    
    def _generate_samples(self) -> List[Dict[str, str]]:
        """Generate qualitative samples for manual inspection."""
        samples = []
        self.model.eval()
        
        # Task-specific prompts
        if self.task_type == TaskType.INSTRUCTION_FOLLOWING:
            sample_prompts = [
                "<|im_start|>user\nWhat is artificial intelligence?<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nExplain machine learning.<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nWrite a Python function.<|im_end|>\n<|im_start|>assistant\n"
            ]
        elif self.task_type == TaskType.CHAT_COMPLETION:
            sample_prompts = [
                "Human: Hello!\nAssistant:",
                "Human: How are you?\nAssistant:",
                "Human: Explain quantum computing.\nAssistant:"
            ]
        else:
            sample_prompts = [
                "The future of artificial intelligence is",
                "In a world where technology advances rapidly,",
                "The most important skill for the 21st century is"
            ]
        
        for i, prompt in enumerate(sample_prompts[:3]):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=getattr(self.config.train, 'temperature', 0.7),
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()
                
                samples.append({
                    'prompt': prompt,
                    'generated_response': response,
                    'task_type': self.task_type.value
                })
                
                logger.info(f"Generated sample {i+1}/3")
                
            except Exception as e:
                logger.warning(f"Error generating sample {i}: {str(e)}")
                samples.append({
                    'prompt': prompt,
                    'generated_response': f"Error: {str(e)}",
                    'task_type': self.task_type.value
                })
        
        return samples
    
    def generate_samples(self, num_samples: int = 5) -> List[Dict[str, str]]:
        """Public method to generate samples."""
        if self.task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            logger.warning("Sample generation not supported for classification tasks")
            return []
        
        try:
            samples = self._generate_samples()
            # Return limited number and ensure all have 'response' key
            return [s for s in samples[:num_samples] if 'generated_response' in s or 'response' in s]
        except Exception as e:
            logger.error(f"Sample generation failed: {e}")
            return []
    
    def run_zero_shot_evaluation(self, test_prompts=None) -> Dict[str, Any]:
        """Run zero-shot evaluation before training with enhanced metrics."""
        logger.info(f"Running zero-shot evaluation for task: {self.task_type.value}")
        
        # Classification tasks don't support zero-shot evaluation
        if self.task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            logger.warning("Zero-shot evaluation not supported for classification tasks")
            return {}
        
        if test_prompts is None:
            # Task-specific default prompts
            if self.task_type == TaskType.INSTRUCTION_FOLLOWING:
                test_prompts = [
                    "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n",
                    "<|im_start|>user\nExplain ML.<|im_end|>\n<|im_start|>assistant\n",
                    "<|im_start|>user\nHow does training work?<|im_end|>\n<|im_start|>assistant\n"
                ]
            elif self.task_type == TaskType.CHAT_COMPLETION:
                test_prompts = [
                    "Human: Hello!\nAssistant:",
                    "Human: How are you?\nAssistant:",
                    "Human: Tell me about AI.\nAssistant:"
                ]
            else:
                test_prompts = [
                    "What is artificial intelligence?",
                    "Explain machine learning briefly.",
                    "How does a computer work?"
                ]
        
        results = []
        self.model.eval()
        
        for prompt in test_prompts:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
                results.append({
                    'prompt': prompt,
                    'response': response,
                    'response_length': len(response.split()),
                    'coherence_score': self._calculate_coherence_score(response)
                })
                
            except Exception as e:
                logger.warning(f"Zero-shot evaluation failed for prompt: {e}")
                results.append({
                    'prompt': prompt,
                    'response': f"Error: {str(e)}",
                    'response_length': 0,
                    'coherence_score': 0.0
                })
        
        # Calculate aggregate metrics
        avg_length = sum(r['response_length'] for r in results) / len(results) if results else 0
        avg_coherence = sum(r['coherence_score'] for r in results) / len(results) if results else 0
        successful_responses = len([r for r in results if not r['response'].startswith('Error')])
        
        zero_shot_metrics = {
            'task_type': self.task_type.value,
            'zero_shot_results': results,
            'avg_response_length': avg_length,
            'avg_coherence_score': avg_coherence,
            'successful_responses': successful_responses,
            'success_rate': successful_responses / len(results) if results else 0
        }
        
        if self.logging_manager:
            self.logging_manager.log_metrics({
                'zero_shot_avg_length': avg_length,
                'zero_shot_coherence': avg_coherence,
                'zero_shot_success_rate': successful_responses / len(results) if results else 0
            })
        
        logger.info(f"Zero-shot evaluation completed. Success rate: {successful_responses}/{len(results)}")
        return zero_shot_metrics
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate a simple coherence score for generated text."""
        if not text or len(text.strip()) == 0:
            return 0.0
        
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        # Check for repetition
        unique_words = len(set(words))
        repetition_penalty = unique_words / len(words)
        
        # Check for reasonable length
        length_score = min(1.0, len(words) / 20.0) if len(words) < 20 else max(0.5, 40.0 / len(words))
        
        # Simple readability check
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        readability_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.7
        
        # Combine scores
        coherence_score = (repetition_penalty + length_score + readability_score) / 3.0
        return min(1.0, max(0.0, coherence_score))
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get enhanced training statistics and information."""
        stats = {
            'config': {
                'model_name': self.config.model.name_or_path,
                'task_type': self.task_type.value,
                'dataset_name': self.config.dataset.name,
                'epochs': self.config.train.num_epochs if hasattr(self.config.train, 'num_epochs') else None,
                'max_steps': self.config.train.max_steps,
                'learning_rate': self.config.train.learning_rate,
                'batch_size': self.config.train.per_device_batch_size,
                'use_peft': self.config.model.use_peft if hasattr(self.config.model, 'use_peft') else False,
                'precision': self.config.model.precision if hasattr(self.config.model, 'precision') else 'auto',
            },
            'dataset_info': {
                'train_size': len(self.dataset) if hasattr(self, 'dataset') and self.dataset else 0,
                'val_size': len(self.eval_dataset) if hasattr(self, 'eval_dataset') and self.eval_dataset else 0,
            },
            'model_info': {
                'loaded': self.model is not None,
                'device': str(next(self.model.parameters()).device) if self.model else 'unknown',
                'vocab_size': len(self.tokenizer) if self.tokenizer else 0,
                'has_peft': hasattr(self.model, 'peft_config') if self.model else False,
            },
            'training_history': self.training_history,
        }
        
        return stats
    
    def save_config(self, path: str):
        """Save enhanced configuration to YAML file."""
        config_dict = {
            'model_name': self.config.model.name_or_path,
            'task_type': self.task_type.value,
            'max_seq_length': self.config.model.max_seq_length,
            'learning_rate': self.config.train.learning_rate,
            'epochs': self.config.train.num_epochs if hasattr(self.config.train, 'num_epochs') else None,
            'max_steps': self.config.train.max_steps,
            'batch_size': self.config.train.per_device_batch_size,
            'dataset_name': self.config.dataset.name,
            'use_peft': self.config.model.use_peft if hasattr(self.config.model, 'use_peft') else False,
            'precision': str(self.config.model.precision) if hasattr(self.config.model, 'precision') else 'auto',
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"TRL SFT configuration saved to {path}")
    
    def run_experiment(self):
        """Run a complete experiment with enhanced features."""
        logger.info(f"Starting TRL SFT experiment: {self.config.logging.run_name}")
        logger.info(f"Task Type: {self.task_type.value}")
        
        try:
            # Setup model and data
            self.setup_model()
            self.setup_data()
            
            # Run zero-shot evaluation if configured
            zero_shot_results = None
            if hasattr(self.config, 'evaluation') and getattr(self.config.evaluation, 'run_zero_shot', False):
                logger.info("Running pre-training zero-shot evaluation...")
                zero_shot_results = self.run_zero_shot_evaluation()
            
            # Start training
            logger.info("Starting training...")
            start_time = time.time()
            train_results = self.train()
            end_time = time.time()
            training_duration = end_time - start_time
            
            # Evaluate after training
            logger.info("Running post-training evaluation...")
            eval_results = self.evaluate()
            
            # Combine results
            results = {
                'mode': 'train_and_eval',
                'task_type': self.task_type.value,
                'train_results': train_results,
                'eval_results': eval_results,
                'training_stats': self.get_training_stats(),
                'config': {
                    'model': self.config.model.__dict__ if hasattr(self.config.model, '__dict__') else str(self.config.model),
                    'dataset': self.config.dataset.__dict__ if hasattr(self.config.dataset, '__dict__') else str(self.config.dataset),
                    'train': self.config.train.__dict__ if hasattr(self.config.train, '__dict__') else str(self.config.train),
                }
            }
            
            # Add zero-shot results if available
            if zero_shot_results:
                results['zero_shot_evaluation'] = zero_shot_results
            
            # Save results
            output_dir = Path(self.config.logging.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results_path = output_dir / "experiment_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
            
            logger.info(f"TRL SFT experiment completed. Results saved to {results_path}")
            return results
            
        except Exception as e:
            logger.error(f"TRL SFT experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            if self.logging_manager:
                self.logging_manager.finish()