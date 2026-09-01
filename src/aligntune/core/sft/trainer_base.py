"""
Abstract base trainer for SFT training, inheriting from UnifiedTrainerBase.
"""

import logging
import torch
import yaml
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config import SFTConfig
from .logging_utils import SFTLogger
from .evaluator import SFTEvaluator
from ..trainer_base import UnifiedTrainerBase, TrainingState
from ..callbacks import CallbackHandler, TrainerControl, TrainerCallback
from ...eval import BaseEvaluator
from ...eval.metrics import RougeMetric, BleuMetric, PerplexityMetric
from ..registry import TaskType

logger = logging.getLogger(__name__)


class SFTTrainerBase(UnifiedTrainerBase):
    """
    SFT-specific base trainer. Inherits lifecycle, saving, and callbacks from UnifiedTrainerBase.
    """
    
    def __init__(self, config: SFTConfig, callbacks: Optional[List[TrainerCallback]] = None):
        """Initialize SFT trainer."""
        super().__init__(config, callbacks)
        
        # Initialize SFT-specific logging
        self.logger = SFTLogger(config.logging)
        
        # Initialize SFT-specific evaluator
        self.evaluator = SFTEvaluator(config)
        
        self.custom_evaluator = None
        
        logger.info(f"Initialized {self.__class__.__name__} for {getattr(config.dataset, 'task_type', 'SFT')} task")
    
    @abstractmethod
    def setup_model(self) -> None:
        """Setup model, tokenizer, and optimization. Handled by core model_loader."""
        pass
    
    def setup_data(self) -> None:
        """
        Unified setup_data implementation for SFT trainers.
        Uses DataManager to load and process datasets.
        Override this method only if you need custom dataset handling.
        """
        logger.info("Setting up SFT datasets with unified DataManager...")

        task_type = self._get_task_type()
        dataset_config = self._extract_dataset_config()
        params = self._extract_dataset_params(dataset_config)

        logger.info(f"Loading dataset: {params['dataset_name']} (split: {params['split']}, task: {task_type})")

        from aligntune.data.manager import DataManager

        manager = DataManager(
            task_type=task_type,
            system_prompt=params['system_prompt'],
            tokenizer=self.tokenizer,
            enable_thinking=params['enable_thinking'],
            column_mapping=params['column_mapping'],
            processing_fn=params['processing_fn'],
            processing_batched=params['processing_batched'],
            max_samples=params['max_samples'],
            max_length=getattr(self.config.model, 'max_seq_length', 1024),
            expected_format=params['format_type'],
            keep_columns=params['keep_columns'],
            val_split_ratio=params['val_split_ratio'],
            test_split_ratio=params['test_split_ratio'],
            seed=params['split_seed'],
            curator_schema_gate=params['curator_schema_gate'],
            curator_clean=params['curator_clean'],
            curator_dedup=params['curator_dedup'],
            curator_use_tiktoken=params['curator_use_tiktoken'],
            curator_max_tokens=params['curator_max_tokens'],
        )

        dataset_dict = manager.load_dataset(
            params['dataset_name'],
            config_name=params['config_name'],
            split=params['split'],
        )

        self.train_dataset = dataset_dict.get("train", None)
        self.eval_dataset = dataset_dict.get("validation", None)
        self.dataset_dict = dataset_dict

        if self.train_dataset is None:
            raise ValueError("No training dataset loaded")

        logger.info(f"Dataset loaded: {len(self.train_dataset)} train samples")
        if self.eval_dataset:
            logger.info(f"Evaluation dataset: {len(self.eval_dataset)} samples")

    def _get_task_type(self) -> str:
        """Get task type from config or class attribute."""
        if hasattr(self, 'TASK_TYPE'):
            return self.TASK_TYPE
        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'task_type'):
            task_type = self.config.dataset.task_type
            if isinstance(task_type, str):
                return task_type
            return task_type.value if hasattr(task_type, 'value') else str(task_type)
        return "sft"

    def _extract_dataset_config(self):
        """Extract dataset configuration from config."""
        if hasattr(self.config, 'dataset'):
            return self.config.dataset
        elif hasattr(self.config, 'datasets') and len(self.config.datasets) > 0:
            return self.config.datasets[0]
        else:
            raise ValueError("No dataset configuration found")

    def _extract_dataset_params(self, dataset_config) -> Dict[str, Any]:
        """Extract all dataset parameters from config."""
        return {
            'dataset_name': self._get_config_value(dataset_config, 'name', 'dataset_name', default='imdb'),
            'split': self._get_config_value(dataset_config, 'split', default=None),
            'config_name': self._get_config_value(dataset_config, 'config_name', 'subset', 'config', default=None),
            'system_prompt': self._get_config_value(dataset_config, 'system_prompt', default=None),
            'enable_thinking': self._get_config_value(self.config.train, 'enable_thinking', default=False),
            'column_mapping': self._get_config_value(dataset_config, 'column_mapping', default=None),
            'processing_fn': self._get_config_value(dataset_config, 'processing_fn', default=None),
            'processing_batched': self._get_config_value(dataset_config, 'processing_batched', default=False),
            'max_samples': self._get_config_value(dataset_config, 'max_samples', default=None),
            'max_eval_samples': self._get_config_value(dataset_config, 'max_eval_samples', default=None),
            'format_type': self._get_config_value(dataset_config, 'format_type', default=None),
            'keep_columns': self._get_config_value(dataset_config, 'keep_columns', default=False),
            'val_split_ratio': self._get_config_value(dataset_config, 'val_split_ratio', default=None),
            'test_split_ratio': self._get_config_value(dataset_config, 'test_split_ratio', default=None),
            'split_seed': self._get_config_value(dataset_config, 'split_seed', default=42),
            'curator_schema_gate': self._get_config_value(dataset_config, 'curator_schema_gate', default=True),
            'curator_clean': self._get_config_value(dataset_config, 'curator_clean', default=False),
            'curator_dedup': self._get_config_value(dataset_config, 'curator_dedup', default='none'),
            'curator_use_tiktoken': self._get_config_value(dataset_config, 'curator_use_tiktoken', default=False),
            'curator_max_tokens': self._get_config_value(dataset_config, 'curator_max_tokens', default=1_000_000),
        }

    def _get_config_value(self, config_obj, *keys, default=None):
        """Get config value, checking multiple possible key names."""
        for key in keys:
            if hasattr(config_obj, key):
                value = getattr(config_obj, key)
                if value is not None:
                    return value
        return default
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute single training step if manually looping."""
        pass
    
    def train(self) -> None:
        """Main training entrypoint. Usually delegated to TRL's trainer in subclasses."""
        logger.info("Starting SFT training...")
        self.setup_model()
        self.setup_data()
        
        if self.callback_handler is None:
            self.callback_handler = CallbackHandler(
                self.callbacks, self.model, self.tokenizer, 
                optimizer=getattr(self.trainer, "optimizer", None) if hasattr(self, 'trainer') else None,
                scheduler=getattr(self.trainer, "lr_scheduler", None) if hasattr(self, 'trainer') else None
            )
            self.callback_handler.add_callback(self)
        
        self.callback_handler.on_init_end(self.config, self.state, self.control)

    def evaluate(self, eval_dataset=None, *args, **kwargs) -> Dict[str, Any]:
        """Unified SFT evaluation with metrics and samples."""
        if not hasattr(self, 'trainer') or not self.trainer:
            logger.warning("No trainer available")
            return {}

        if eval_dataset:
            self.eval_dataset = eval_dataset

        logger.info(f"Running SFT evaluation for task: {self._get_task_type()}")

        # Run trainer evaluation
        eval_results = self.trainer.evaluate() if hasattr(self, 'eval_dataset') and self.eval_dataset else {}

        # Add task type
        eval_results['task_type'] = self._get_task_type()

        # Calculate perplexity from loss
        if "eval_loss" in eval_results:
            try:
                eval_results["eval_perplexity"] = float(
                    torch.exp(torch.tensor(float(eval_results["eval_loss"]))).item()
                )
            except Exception as e:
                logger.debug(f"Could not calculate perplexity: {e}")

        # Add quality metrics using SFTEvaluator for generation tasks
        task_type_str = self._get_task_type()
        if task_type_str not in ["text_classification", "token_classification"]:
            try:
                if self.model and self.tokenizer and hasattr(self, 'eval_dataset') and self.eval_dataset:
                    quality_metrics = self.evaluator.evaluate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        dataset=self.eval_dataset,
                        config=self.config
                    )

                    for key, value in quality_metrics.items():
                        if key not in eval_results:
                            eval_results[key] = value

                    logger.info(f"Added {len(quality_metrics)} quality metrics")
            except Exception as e:
                logger.warning(f"Could not compute quality metrics: {e}")

        # Generate samples for generation tasks
        if task_type_str not in ["text_classification", "token_classification"]:
            try:
                samples = self.generate_samples(num_samples=3)
                eval_results['qualitative_samples'] = samples
            except Exception as e:
                logger.warning(f"Could not generate samples: {e}")
                eval_results['qualitative_samples'] = []

        if self.callback_handler:
            self.callback_handler.on_evaluate(self.config, self.state, self.control, metrics=eval_results)

        logger.info(f"Evaluation completed: {len(eval_results)} metrics")
        return eval_results

    def generate_samples(self, num_samples: int = 3, custom_prompts: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Generate samples from eval/train dataset or custom prompts."""
        try:
            logger.info(f"Generating {num_samples} samples")

            # Use custom prompts if provided, otherwise sample from dataset
            if custom_prompts:
                prompts = custom_prompts[:num_samples]
            else:
                dataset = self.eval_dataset if hasattr(self, 'eval_dataset') and self.eval_dataset else self.train_dataset
                if not dataset:
                    logger.warning("No dataset available for sampling")
                    return []

                # Sample from dataset
                import random
                indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
                prompts = []
                for idx in indices:
                    sample = dataset[idx]
                    if isinstance(sample, dict) and 'text' in sample:
                        prompts.append(sample['text'][:200])  # Use first 200 chars
                    elif isinstance(sample, str):
                        prompts.append(sample[:200])
                    else:
                        logger.debug(f"Skipping sample with unexpected format")

            samples = []
            self.model.eval()

            for i, prompt in enumerate(prompts):
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
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
                        'response': response,
                        'task_type': self._get_task_type()
                    })

                except Exception as e:
                    logger.warning(f"Error generating sample {i}: {e}")
                    samples.append({
                        'prompt': prompt,
                        'response': f"Error: {str(e)}",
                        'task_type': self._get_task_type()
                    })

            logger.info(f"Generated {len(samples)} samples")
            return samples

        except Exception as e:
            logger.error(f"Sample generation failed: {e}")
            return []

    def cleanup(self) -> None:
        """Release model and GPU memory. Safe to call even before setup_model."""
        import torch
        if getattr(self, 'model', None) is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"{self.__class__.__name__} cleanup completed")

    def save_model(self, output_dir: Optional[str] = None) -> str:
        """Save trained model and tokenizer."""
        save_path = output_dir or self.config.logging.output_dir
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {save_path}")

        # Save model and tokenizer
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        # Save config with task type
        config_dict = {
            'task_type': self._get_task_type(),
            'model_name': self.config.model.name_or_path
        }
        config_path = save_path / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        logger.info(f"Model saved to {save_path}")
        return str(save_path)
