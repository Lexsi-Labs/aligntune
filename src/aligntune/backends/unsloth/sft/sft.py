"""
Enhanced Unsloth SFT Backend Implementation with Task Type Support.

This module provides task-aware training using Unsloth optimizations,
supporting multiple task types:
- Instruction Following
- Supervised Fine-Tuning
- Text Generation
- Chat Completion

Note: Classification tasks are not supported by Unsloth backend.
"""

import logging
import time
import yaml
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader

from aligntune.core.sft.trainer_base import SFTTrainerBase
from aligntune.core.registry import TaskType
from aligntune.core.sft.evaluator import EnhancedEvaluator
from aligntune.core.dataset_adapters import schema_detector
from aligntune.core.precision_handler import PrecisionHandler
logger = logging.getLogger(__name__)





class UnslothSFTTrainer(SFTTrainerBase):
    """Enhanced SFT trainer using Unsloth with task type support."""
    
    # Supported task types for Unsloth
    SUPPORTED_TASKS = [
        TaskType.SFT,
        TaskType.PRETRAINING,
        TaskType.INSTRUCTION_FOLLOWING,
    ]
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task_type = self._get_task_type()
        self.training_config = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.training_history = []
        self.logging_manager = None
        self.evaluator = None
        self.unsloth_model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.eval_dataset = None  # Already exists - no need to add
        self.custom_evaluator = None  # ADD THIS LINE (for BaseEvaluator)
        
        # Validate task type
        self._validate_task_type()
        
        logger.info(f"Initialized UnslothSFTTrainer for task: {self.task_type.value}")
    
    def _get_task_type(self) -> TaskType:
        """Extract task type from config."""
        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'task_type'):
            task_type = self.config.dataset.task_type
        elif hasattr(self.config, 'train') and hasattr(self.config.train, 'task_type'):
            task_type = self.config.train.task_type
        else:
            # Default to supervised fine-tuning
            task_type = TaskType.SFT
        
        # Convert string to enum if needed
        if isinstance(task_type, str):
            task_type = TaskType(task_type.lower())
        
        return task_type
    
    def _validate_task_type(self):
        """Validate that the task type is supported by Unsloth."""
        if self.task_type not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Task type {self.task_type.value} is not supported by Unsloth backend. "
                f"Supported tasks: {[t.value for t in self.SUPPORTED_TASKS]}. "
                f"Use TRL backend for classification tasks."
            )
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth is available."""
        try:
            import unsloth
            from unsloth import FastLanguageModel
            from trl import SFTTrainer, SFTConfig, ModelConfig
            return True
        except ImportError:
            return False
    
    def setup_model(self) -> None:
        """Setup Unsloth-optimized model with task-aware configuration."""
        from ....core.model_loader import build_model
        
        apply_peft = False
        if hasattr(self.config.model, 'peft'):
            apply_peft = getattr(self.config.model.peft, 'enabled', False)
            
        self.unsloth_model, self.tokenizer = build_model(
            config=self.config,
            task_type=self.task_type,
            use_unsloth=True,
            apply_peft=apply_peft
        )
        self.model = self.unsloth_model
    
    # setup_data() inherited from SFTTrainerBase - uses unified DataManager

    def setup_dataset(self) -> None:
        """Alias for compatibility."""
        self.setup_data()

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step (required by abstract base class)."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_model() first.")
        
        # The actual training step is handled by TRL's SFTTrainer
        return {"loss": 0.0, "learning_rate": 0.0}
    
    def setup_trainer(self) -> None:
        """Setup TRL SFTTrainer with task-aware configuration."""
        try:
            from trl import SFTTrainer, SFTConfig, ModelConfig

            logger.info(f"Setting up TRL SFTTrainer for task: {self.task_type.value}")

            # Get training parameters
            from ....core.sft.training_args_builder import build_sft_config

            # Call base class method for optimizer/scheduler normalization (unified with TRL approach)
            optim_scheduler = self._setup_optimizer_scheduler(dataset_for_estimation=self.train_dataset)

            # Extract scheduler kwargs (filter out computed values)
            filtered_scheduler_kwargs = {
                k: v for k, v in optim_scheduler['scheduler_config']['lr_scheduler_kwargs'].items()
                if k not in ['num_training_steps', 'num_warmup_steps']
            }

            # Build extra_kwargs for build_sft_config
            extra_kwargs = {
                "optim": optim_scheduler['optimizer_name'],
                "lr_scheduler_type": optim_scheduler['scheduler_name'],
                "lr_scheduler_kwargs": filtered_scheduler_kwargs,
                "dataloader_pin_memory": False,  # Unsloth specific
            }

            # Handle optimizer args string
            optim_args_str = optim_scheduler['optim_args_str']
            should_skip_optim_args = (
                not optim_args_str or
                not optim_args_str.strip() or
                "=" not in optim_args_str or
                any(x in optim_scheduler['optimizer_name'].lower() for x in ["8bit", "8_bit", "bnb", "paged"])
            )
            if not should_skip_optim_args:
                extra_kwargs["optim_args"] = optim_args_str

            self.training_config = build_sft_config(
                config=self.config,
                train_dataset=self.train_dataset,
                eval_dataset=getattr(self, 'eval_dataset', None),
                output_dir=getattr(self.config.logging, 'output_dir', './output') if hasattr(self.config, 'logging') else './output',
                **extra_kwargs  # Pass all normalized optimizer/scheduler args
            )

            # Get max_seq_length
            max_seq_len = getattr(self.config.model, 'max_seq_length', 2048) if hasattr(self.config.model, 'max_seq_length') else 2048

            # Task-specific trainer settings
            packing = False  # Generally disable packing for better quality

            logger.info("Initializing Unsloth SFTTrainer...")

            def prepare_dataset(dataset):
                """Convert DataManager's SFT messages to Unsloth's text field."""
                if "text" in dataset.column_names:
                    return dataset.select_columns(["text"])
                if "messages" not in dataset.column_names:
                    raise ValueError(
                        "Unsloth SFT expects DataManager output with either "
                        "a 'messages' column or a pretraining 'text' column"
                    )

                names = ["messages", "chat_template_kwargs"]
                dataset = dataset.select_columns(
                    [name for name in names if name in dataset.column_names]
                )

                def formatting_prompts_func(examples):
                    conversations = examples["messages"]
                    template_kwargs = examples.get(
                        "chat_template_kwargs", [{} for _ in conversations]
                    )
                    texts = []
                    for conversation, kwargs in zip(conversations, template_kwargs):
                        try:
                            text = self.tokenizer.apply_chat_template(
                                conversation,
                                tokenize=False,
                                add_generation_prompt=False,
                                **(kwargs or {}),
                            )
                        except TypeError:
                            text = self.tokenizer.apply_chat_template(
                                conversation,
                                tokenize=False,
                                add_generation_prompt=False,
                            )
                        texts.append(text)
                    return {"text": texts}

                return dataset.map(
                    formatting_prompts_func,
                    batched=True,
                    remove_columns=dataset.column_names,
                    desc="Formatting SFT messages for Unsloth",
                )

            train_dataset = prepare_dataset(self.train_dataset)
            eval_dataset = (
                prepare_dataset(self.eval_dataset)
                if self.eval_dataset is not None
                else None
            )

            # prepare_dataset() above unconditionally normalizes both the
            # pretraining-text and messages cases down to a column literally
            # named "text" - but self.training_config.dataset_text_field may
            # still hold whatever field name the user configured (e.g.
            # "completion", when DataManager's SFT normalizer renamed the raw
            # dataset's own "text" column to avoid colliding with this one).
            # Unsloth's internal tokenization reads args.dataset_text_field,
            # not the "text" kwarg below, so a stale value there raised
            # KeyError on the now-nonexistent original field. Keep both in
            # sync with what prepare_dataset() actually produced.
            self.training_config.dataset_text_field = "text"

            # Create trainer
            self.trainer = SFTTrainer(
                model=self.unsloth_model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=self.training_config,
                dataset_text_field="text",
                max_seq_length=max_seq_len,
                dataset_num_proc=2,
                packing=packing,
                callbacks=self.get_hf_callbacks()
            )
            
            logger.info(f"TRL SFTTrainer setup completed for {self.task_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise
    
    def train(self) -> Dict[str, Any]:
        """Execute training with Unsloth optimizations."""
        try:
            logger.info(f"Starting Unsloth SFT training for task: {self.task_type.value}")
            start_time = time.time()
            
            # Setup components
            self.setup_model()
            self.setup_data()
            self.setup_trainer()
            
            # Suppress Unsloth's informational warning about num_items_in_batch
            # This is a known limitation with Qwen2 models and gradient accumulation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*num_items_in_batch.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*Qwen2ForCausalLM does not accept.*", category=UserWarning)
                # Start training
                training_result = self.trainer.train()
            
            # Save model
            output_dir = self.config.logging.output_dir if hasattr(self.config, 'logging') else './output'
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            training_time = time.time() - start_time
            
            # Compile results
            results = {
                "task_type": self.task_type.value,
                "training_time": training_time,
                "final_loss": training_result.training_loss,
                "total_steps": training_result.global_step,
                "model_path": output_dir,
                "training_history": self.training_history,
            }
            
            # Trigger deployment export if configured (e.g. export_format: gguf in config)
            self._auto_export()
            
            logger.info(f"Unsloth SFT training completed in {training_time:.2f} seconds")
            logger.info(f"Task: {self.task_type.value}, Final loss: {training_result.training_loss:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

if __name__ == "__main__":
    from aligntune.core.registry import TaskType
    from types import SimpleNamespace
    
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace(
        name_or_path="EleutherAI/pythia-14m",
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["query_key_value"],
        device_map="cpu",
    )
    cfg.dataset = SimpleNamespace(
        name="imdb",
        task_type="sft",
        split="train[:1%]"
    )
    cfg.train = SimpleNamespace(
        learning_rate=2e-4,
        weight_decay=0.01,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=5,
    )
    cfg.logging = SimpleNamespace(output_dir="./output/sft_test")
    
    print("Testing Unsloth SFT setup_model and setup_data...")
    trainer = UnslothSFTTrainer(cfg)
    try:
        trainer.setup_model()
        print("Model initialized successfully!")
        trainer.setup_data()
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Test note: {e}")
