"""
Classification Trainer - FINAL COMPLETE VERSION

Copy this to:
/teamspace/studios/this_studio/Finetunehub/src/aligntune/backends/trl/sft/Classification_trainer.py
"""

import torch
from typing import Optional
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification
)
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """
    Complete Classification Trainer for text and token classification.
    
    Supports:
    - Text Classification (sentiment, topic classification, etc.)
    - Token Classification (NER, POS tagging, etc.)
    
    Uses standard Transformers Trainer (NOT TRL) for proper classification.
    """
    
    def __init__(self, config):
        """Initialize trainer with config."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.task_type = config.dataset.task_type
        
        logger.info(f"✓ ClassificationTrainer initialized for {self.task_type.value}")
        print(f"✓ ClassificationTrainer initialized for {self.task_type.value}")
    
    def setup_model(self):
        """Load model and tokenizer based on task type."""
        from aligntune.core.sft.config import TaskType
        
        print(f"\nLoading model: {self.config.model.name_or_path}")
        logger.info(f"Loading model: {self.config.model.name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name_or_path,
            use_fast=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Ensure pad_token_id is set (required for data collator batching)
        # For Llama and similar tokenizers, we need to set it on both tokenizer and config
        pad_token_id = None
        if hasattr(self.tokenizer, 'pad_token_id'):
            pad_token_id = self.tokenizer.pad_token_id
        
        if pad_token_id is None:
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                pad_token_id = self.tokenizer.eos_token_id
            elif self.tokenizer.pad_token is not None:
                # If pad_token was added, get its ID from the tokenizer
                try:
                    pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                    if pad_token_id == self.tokenizer.unk_token_id:  # If it's the unk token, use eos instead
                        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                            pad_token_id = self.tokenizer.eos_token_id
                except Exception as e:
                    # Fallback: use eos_token_id if available
                    if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                        pad_token_id = self.tokenizer.eos_token_id
                    else:
                        logger.warning(f"Could not get pad_token_id from tokenizer: {e}")
        
        # Set pad_token_id on tokenizer (multiple ways to ensure it sticks)
        if pad_token_id is not None:
            # Set directly on tokenizer
            if hasattr(self.tokenizer, 'pad_token_id'):
                self.tokenizer.pad_token_id = pad_token_id
            # Also set on tokenizer's special_tokens_map if it exists
            if hasattr(self.tokenizer, 'special_tokens_map'):
                if 'pad_token' not in self.tokenizer.special_tokens_map:
                    self.tokenizer.special_tokens_map['pad_token'] = self.tokenizer.pad_token
            # Set on tokenizer config if it exists
            if hasattr(self.tokenizer, 'init_kwargs') and 'pad_token_id' in self.tokenizer.init_kwargs:
                self.tokenizer.init_kwargs['pad_token_id'] = pad_token_id
            # Also try setting via __setattr__ to ensure it persists
            try:
                object.__setattr__(self.tokenizer, 'pad_token_id', pad_token_id)
            except:
                pass
        
        # Final check: ensure pad_token_id is set (critical for batching)
        final_pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if final_pad_token_id is None:
            logger.error("pad_token_id is still None after setup! This will cause batching errors.")
            raise ValueError("pad_token_id must be set for batching. Tokenizer configuration issue.")
        
        # Determine precision and quantization settings
        use_8bit = False
        torch_dtype = None
        device_map = "cpu"  # Keep on CPU initially
        
        # Check if precision is set and configure accordingly
        if hasattr(self.config.model, 'precision') and self.config.model.precision:
            precision_value = self.config.model.precision.value if hasattr(self.config.model.precision, 'value') else str(self.config.model.precision)
            if precision_value == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif precision_value == "fp16":
                torch_dtype = torch.float16
        
        # For 4B+ models, use lower precision dtype to reduce memory
        # NOTE: 8-bit quantization (BitsAndBytes) requires PEFT adapters for fine-tuning
        # Since we're not using PEFT for classification, we skip 8-bit quantization
        # and rely on BF16/FP16 precision instead
        model_size_estimate = "4B" if "4B" in self.config.model.name_or_path or "4b" in self.config.model.name_or_path.lower() else "unknown"
        use_8bit = False  # Disable 8-bit quantization for classification (requires PEFT)
        quantization_config = None
        
        # Ensure we use lower precision dtype for 4B models
        if torch_dtype is None and torch.cuda.is_available():
            # Default to BF16 if supported, else FP16
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                logger.info("Using BF16 precision for 4B model to reduce memory usage (8-bit quantization disabled - requires PEFT)")
            else:
                torch_dtype = torch.float16
                logger.info("Using FP16 precision for 4B model to reduce memory usage (8-bit quantization disabled - requires PEFT)")
        
        # Load appropriate model based on task
        # Load on CPU first with appropriate precision/quantization to avoid OOM
        model_kwargs = {
            "num_labels": self.config.model.num_labels,
            "ignore_mismatched_sizes": True,
            "device_map": device_map,  # Keep on CPU
            "torch_dtype": torch_dtype if torch_dtype else None
        }
        
        # Add quantization config if using 8-bit
        if use_8bit and quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Remove None values
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        
        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model.name_or_path,
                **model_kwargs
            )
            logger.info(f"Loaded AutoModelForSequenceClassification with {self.config.model.num_labels} labels")
            
        else:  # TOKEN_CLASSIFICATION
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model.name_or_path,
                **model_kwargs
            )
            logger.info(f"Loaded AutoModelForTokenClassification with {self.config.model.num_labels} labels")
        
        # Ensure model stays on CPU (device_map should handle this, but double-check)
        if self.model is not None and not use_8bit:  # 8-bit models handle device_map automatically
            try:
                # Only move to CPU if not already there and not using device_map
                current_device = next(self.model.parameters()).device
                if current_device.type != "cpu":
                    self.model = self.model.to("cpu")
                    logger.info("Model moved to CPU to avoid OOM during setup")
                else:
                    logger.info("Model already on CPU (via device_map)")
            except Exception as move_error:
                logger.warning(f"Could not verify/move model to CPU: {move_error}")
        
        # Resize token embeddings if we added tokens
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # CRITICAL: Set pad_token_id on model config (required for DataCollatorWithPadding)
        # This is often the missing piece that causes "Cannot handle batch sizes > 1" errors
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_token_id is not None and hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'pad_token_id'):
                self.model.config.pad_token_id = pad_token_id
            # Also try setting on model.config if it's a dict-like object
            try:
                setattr(self.model.config, 'pad_token_id', pad_token_id)
            except:
                pass
            logger.info(f"Set model.config.pad_token_id = {pad_token_id}")
        
        # Enable gradient checkpointing to reduce memory usage (helps with OOM)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled to reduce memory usage")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")
        
        # Apply PEFT (LoRA) if enabled
        if self.config.model.peft_enabled:
            from peft import LoraConfig, get_peft_model
            
            # Get target modules from config or auto-detect
            target_modules = self.config.model.target_modules
            if target_modules is None:
                # Auto-detect based on model architecture
                if "distilbert" in self.config.model.name_or_path.lower():
                    target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
                elif "bert" in self.config.model.name_or_path.lower():
                    target_modules = ["query", "key", "value", "dense"]
                elif "roberta" in self.config.model.name_or_path.lower():
                    target_modules = ["query", "key", "value", "dense"]
                else:
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                
                logger.info(f"Auto-detected target modules: {target_modules}")
            
            # Create PEFT config
            peft_config = LoraConfig(
                r=self.config.model.lora_rank,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
                bias=self.config.model.bias,
                task_type="SEQ_CLS" if self.task_type == TaskType.TEXT_CLASSIFICATION else "TOKEN_CLS",
                target_modules=target_modules
            )
            
            # Apply PEFT
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        print(f"✓ Model loaded successfully")
    
    def setup_data(self):
        """Load and prepare dataset."""
        from aligntune.core.sft.config import TaskType
        
        print(f"\nLoading dataset: {self.config.dataset.name}")
        logger.info(f"Loading dataset: {self.config.dataset.name}")
        
        # Load dataset with trust_remote_code for compatibility
        try:
            if self.config.dataset.subset:
                dataset = load_dataset(
                    self.config.dataset.name,
                    self.config.dataset.subset,
                    split=self.config.dataset.split,
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    self.config.dataset.name,
                    split=self.config.dataset.split,
                    trust_remote_code=True
                )
        except Exception as e:
            logger.error(f"Failed to load dataset with trust_remote_code: {e}")
            # Fallback without trust_remote_code
            if self.config.dataset.subset:
                dataset = load_dataset(
                    self.config.dataset.name,
                    self.config.dataset.subset,
                    split=self.config.dataset.split
                )
            else:
                dataset = load_dataset(
                    self.config.dataset.name,
                    split=self.config.dataset.split
                )
        
        # Limit samples if specified
        if self.config.dataset.max_samples:
            max_samples = min(self.config.dataset.max_samples, len(dataset))
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited dataset to {max_samples} samples")
        
        # Apply column mapping if provided (must be done before splitting)
        if self.config.dataset.column_mapping:
            logger.info(f"Applying column mapping: {self.config.dataset.column_mapping}")
            # Rename columns: old_name -> new_name
            dataset = dataset.rename_columns(self.config.dataset.column_mapping)
            logger.info(f"Renamed columns: {self.config.dataset.column_mapping}")
        
        # Split into train/eval (90/10)
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
        
        # Tokenize based on task type
        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            def tokenize_function(examples):
                """Tokenize text classification data."""
                result = self.tokenizer(
                    examples[self.config.dataset.text_column],
                    truncation=True,
                    padding=False,
                    max_length=self.config.model.max_seq_length
                )
                # Get labels and convert to 0-indexed if needed (PyTorch expects 0-indexed)
                labels = examples[self.config.dataset.label_column]
                if isinstance(labels, list) and len(labels) > 0:
                    min_label = min(labels)
                    if min_label > 0:
                        # Labels are 1-indexed, convert to 0-indexed
                        labels = [l - 1 for l in labels]
                result['labels'] = labels
                return result
        
        else:  # TOKEN_CLASSIFICATION
            def tokenize_function(examples):
                """Tokenize token classification data with label alignment."""
                tokenized = self.tokenizer(
                    examples[self.config.dataset.tokens_column],
                    truncation=True,
                    is_split_into_words=True,
                    padding=False,
                    max_length=self.config.model.max_seq_length
                )
                
                # Align labels with tokens
                labels = []
                for i, label in enumerate(examples[self.config.dataset.tags_column]):
                    word_ids = tokenized.word_ids(batch_index=i)
                    label_ids = []
                    previous_word_idx = None
                    
                    for word_idx in word_ids:
                        if word_idx is None:
                            # Special tokens get -100
                            label_ids.append(-100)
                        elif word_idx != previous_word_idx:
                            # First token of a word gets the label
                            label_ids.append(label[word_idx])
                        else:
                            # Other tokens of the same word get -100
                            label_ids.append(-100)
                        previous_word_idx = word_idx
                    
                    labels.append(label_ids)
                
                tokenized['labels'] = labels
                return tokenized
        
        # Apply tokenization
        self.train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        self.eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset"
        )
        
        print(f"✓ Dataset loaded - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")
        logger.info(f"Dataset loaded - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")
    
    def setup_training(self):
        """Setup trainer with appropriate settings."""
        from aligntune.core.sft.config import TaskType
        
        print("\nConfiguring trainer...")
        logger.info("Configuring trainer...")
        
        # Clear GPU cache before creating trainer to avoid OOM errors
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cache_error:
            logger.warning(f"Could not clear GPU cache: {cache_error}")
        
        # Training arguments
        # For quick testing: use max_steps=10, then revert to full training
        max_steps_for_testing = None  # Set to None for full training, or 10 for quick testing
        training_args_dict = {
            "output_dir": self.config.logging.output_dir,
            "run_name": self.config.logging.run_name,
            "per_device_train_batch_size": self.config.train.per_device_batch_size,
            "per_device_eval_batch_size": self.config.train.per_device_batch_size,
            "learning_rate": self.config.train.learning_rate,
            "weight_decay": self.config.train.weight_decay,
            "warmup_ratio": self.config.train.warmup_ratio,
            "eval_strategy": "steps",
            "eval_steps": self.config.train.eval_interval,
            "save_strategy": "steps",
            "save_steps": self.config.train.save_interval,
            "load_best_model_at_end": self.config.train.load_best_model_at_end,
            "metric_for_best_model": self.config.train.metric_for_best_model,
            "greater_is_better": self.config.train.greater_is_better,
            "logging_steps": 10,
            "logging_dir": f"{self.config.logging.output_dir}/logs",
            # Use precision from model config, default to BF16 if CUDA supports it (for memory efficiency)
            "fp16": (hasattr(self.config.model, 'precision') and 
                    hasattr(self.config.model.precision, 'value') and
                    self.config.model.precision.value == "fp16") if hasattr(self.config.model, 'precision') else False,
            "bf16": (hasattr(self.config.model, 'precision') and 
                    hasattr(self.config.model.precision, 'value') and
                    self.config.model.precision.value == "bf16") if hasattr(self.config.model, 'precision') else 
                    (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),  # Auto-enable BF16 if supported
            "report_to": "none",  # Disable wandb/tensorboard
            "save_total_limit": 2,  # Keep only 2 checkpoints
            "push_to_hub": False
        }
        
        # Add gradient_accumulation_steps if specified in config (helps with OOM)
        if hasattr(self.config.train, 'gradient_accumulation_steps') and self.config.train.gradient_accumulation_steps is not None:
            training_args_dict["gradient_accumulation_steps"] = self.config.train.gradient_accumulation_steps
        # Set either max_steps (for quick testing) or num_train_epochs (for full training)
        # Don't pass num_train_epochs at all when max_steps is set (TrainingArguments doesn't like both)
        if max_steps_for_testing is not None:
            training_args_dict["max_steps"] = max_steps_for_testing
            # Don't include num_train_epochs when max_steps is set
        else:
            training_args_dict["num_train_epochs"] = self.config.train.epochs
            # Don't include max_steps when num_train_epochs is set
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Ensure pad_token_id is set before creating data collator (required for batching)
        if self.tokenizer is not None:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is None:
                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                elif self.tokenizer.pad_token is not None:
                    try:
                        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                        if pad_token_id != self.tokenizer.unk_token_id:  # Make sure it's not the unk token
                            self.tokenizer.pad_token_id = pad_token_id
                        else:
                            # Fallback: use eos_token_id if available
                            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    except Exception as e:
                        # Fallback: use eos_token_id if available
                        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        else:
                            logger.warning(f"Could not set pad_token_id: {e}")
            
            # Final check: ensure pad_token_id is set (critical for batching)
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is None:
                logger.error("pad_token_id is still None before creating data collator! This will cause batching errors.")
                raise ValueError("pad_token_id must be set for batching. Tokenizer configuration issue.")
        
        # Choose data collator based on task
        # Explicitly pass pad_token_id to ensure it's used
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_token_id is None:
            raise ValueError("pad_token_id must be set before creating data collator")
        
        # Create a wrapper class to ensure pad_token_id is always set before batching
        class SafeDataCollatorWithPadding(DataCollatorWithPadding):
            def __call__(self, features):
                # Ensure pad_token_id is set before batching (critical fix)
                if self.tokenizer is not None:
                    if getattr(self.tokenizer, 'pad_token_id', None) is None:
                        # Try to set it from eos_token_id
                        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        # If still None, try to get it from pad_token
                        if getattr(self.tokenizer, 'pad_token_id', None) is None and self.tokenizer.pad_token is not None:
                            try:
                                pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                                if pad_id != self.tokenizer.unk_token_id:
                                    self.tokenizer.pad_token_id = pad_id
                                elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                            except:
                                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                return super().__call__(features)
        
        class SafeDataCollatorForTokenClassification(DataCollatorForTokenClassification):
            def __call__(self, features):
                # Ensure pad_token_id is set before batching (critical fix)
                if self.tokenizer is not None:
                    if getattr(self.tokenizer, 'pad_token_id', None) is None:
                        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        if getattr(self.tokenizer, 'pad_token_id', None) is None and self.tokenizer.pad_token is not None:
                            try:
                                pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                                if pad_id != self.tokenizer.unk_token_id:
                                    self.tokenizer.pad_token_id = pad_id
                                elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                            except:
                                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                return super().__call__(features)
        
        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            data_collator = SafeDataCollatorWithPadding(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=8  # Optimize for tensor cores
            )
        else:  # TOKEN_CLASSIFICATION
            data_collator = SafeDataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=8
            )
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            """Compute classification metrics."""
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=-1)
            
            # For token classification, flatten and filter
            if self.task_type == TaskType.TOKEN_CLASSIFICATION:
                predictions = predictions.flatten()
                labels = labels.flatten()
                # Remove ignored indices (-100)
                mask = labels != -100
                predictions = predictions[mask]
                labels = labels[mask]
            
            # Calculate metrics
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
                'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
                'recall': recall_score(labels, predictions, average='weighted', zero_division=0)
            }
        
        # Clear CUDA cache before creating trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create trainer with error handling for OOM
        try:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                tokenizer=self.tokenizer
            )
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_error:
            # Check if it's actually an OOM error
            is_oom = isinstance(oom_error, torch.cuda.OutOfMemoryError) or "out of memory" in str(oom_error).lower()
            
            if is_oom:
                # Clear cache and try to provide helpful error message
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                error_msg = (
                    f"CUDA out of memory when creating trainer. "
                    f"This usually means other processes are using GPU memory. "
                    f"Try: (1) Free GPU memory by stopping other processes, "
                    f"(2) Reduce batch_size in config, (3) Use a smaller model. "
                    f"Original error: {str(oom_error)[:200]}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from oom_error
            else:
                # Re-raise if it's not an OOM error
                raise
        
        print("✓ Trainer configured")
        logger.info("Trainer configured successfully")
    
    def train(self):
        """Train the model."""
        print("\n" + "=" * 80)
        print("TRAINING")
        print("=" * 80)
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
        # Clear CUDA cache before training to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache before training")
        
        logger.info("Starting training...")
        try:
            train_result = self.trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.error(f"CUDA OOM during training: {e}")
                print(f"\n❌ CUDA Out of Memory error during training.")
                print("   Suggestions:")
                print("   - Reduce batch_size in your config")
                print("   - Increase gradient_accumulation_steps to maintain effective batch size")
                print("   - Reduce max_seq_length")
                print("   - Use a smaller model")
                raise
            else:
                raise
        
        print("\n✓ Training completed")
        logger.info("Training completed")
        return train_result
    
    def evaluate(self):
        """Evaluate the model."""
        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)
        
        # Ensure pad_token is set before evaluation (required for batching)
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Also ensure pad_token_id is set (critical for batching)
        if self.tokenizer is not None:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            
            if pad_token_id is None:
                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    pad_token_id = self.tokenizer.eos_token_id
                elif self.tokenizer.pad_token is not None:
                    try:
                        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                        if pad_token_id == self.tokenizer.unk_token_id:  # If it's the unk token, use eos instead
                            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                                pad_token_id = self.tokenizer.eos_token_id
                    except Exception as e:
                        # Fallback: use eos_token_id if available
                        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                            pad_token_id = self.tokenizer.eos_token_id
                        else:
                            logger.warning(f"Could not get pad_token_id in evaluate(): {e}")
            
            # Set pad_token_id if we found one
            if pad_token_id is not None:
                if hasattr(self.tokenizer, 'pad_token_id'):
                    self.tokenizer.pad_token_id = pad_token_id
                try:
                    object.__setattr__(self.tokenizer, 'pad_token_id', pad_token_id)
                except:
                    pass
            
            # Final check: ensure pad_token_id is set (critical for batching)
            final_pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            if final_pad_token_id is None:
                error_msg = "pad_token_id is still None in evaluate()! Cannot handle batch sizes > 1 without padding token."
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        logger.info("Starting evaluation...")
        eval_result = self.trainer.evaluate()
        
        # Calculate perplexity from loss (if available)
        if "eval_loss" in eval_result:
            try:
                eval_result["eval_perplexity"] = float(
                    torch.exp(torch.tensor(float(eval_result["eval_loss"]))).item()
                )
            except Exception as e:
                logger.debug(f"Could not calculate perplexity: {e}")
        
        print("\nEvaluation Results:")
        for key, value in eval_result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        logger.info(f"Evaluation completed: {len(eval_result)} metrics computed")
        return eval_result
    
    def save_model(self, output_dir: Optional[str] = None) -> str:
        """Save the trained model.
        
        Args:
            output_dir: Directory to save to (defaults to config output_dir)
        
        Returns:
            Path where model was saved
        """
        if output_dir is None:
            save_path = f"{self.config.logging.output_dir}/{self.config.logging.run_name}/final_model"
        else:
            save_path = output_dir
        
        print(f"\nSaving model to {save_path}...")
        logger.info(f"Saving model to {save_path}")
        
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"✓ Model saved")
        logger.info("Model saved successfully")
        
        # Store for push_to_hub
        self._last_save_path = save_path
        
        return save_path
    
    def run_experiment(self):
        """
        Run complete training experiment.
        
        This is the main method that backend_factory calls!
        """
        print("\n" + "=" * 80)
        print(f"CLASSIFICATION TRAINING - {self.task_type.value.upper()}")
        print("=" * 80)
        
        logger.info(f"Starting classification experiment: {self.task_type.value}")
        
        try:
            # Setup
            self.setup_model()
            self.setup_data()
            self.setup_training()
            
            # Train
            train_result = self.train()
            
            # Evaluate
            eval_result = self.evaluate()
            
            # Save
            model_path = self.save_model()
            
            # Return complete results
            results = {
                'task_type': self.task_type.value,
                'train_loss': train_result.training_loss,
                'eval_results': eval_result,
                'model_path': model_path,
                'steps': train_result.global_step
            }
            
            print("\n" + "=" * 80)
            print("✅ EXPERIMENT COMPLETE")
            print("=" * 80)
            print(f"  Task: {results['task_type']}")
            print(f"  Train Loss: {results['train_loss']:.4f}")
            print(f"  Eval Accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
            print(f"  Eval F1: {eval_result.get('eval_f1', 0):.4f}")
            print(f"  Model: {model_path}")
            print("=" * 80)
            
            logger.info(f"Experiment completed successfully: {results}")
            return results
        
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


# Export the trainer
__all__ = ['ClassificationTrainer']