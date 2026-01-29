"""
Abstract base trainer for unified SFT training.

This module provides the abstract SFTTrainerBase class that defines the lifecycle
and interface for all SFT training algorithms.
"""

import logging
import time
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .config import SFTConfig
from .logging import SFTLogger
from .evaluator import SFTEvaluator
from ..callbacks import CallbackHandler, TrainerControl, TrainerCallback
from ...eval import BaseEvaluator
from ...eval.metrics import (
    RougeMetric, BleuMetric, PerplexityMetric
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Training state tracking."""
    step: int = 0
    epoch: int = 0
    best_metric: float = 0.0
    checkpoint_path: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    last_eval_time: float = field(default_factory=time.time)
    last_save_time: float = field(default_factory=time.time)
    
    def update_step(self, step: int):
        """Update current step."""
        self.step = step
    
    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.epoch = epoch
    
    def update_best_metric(self, metric: float):
        """Update best metric if improved."""
        if metric > self.best_metric:
            self.best_metric = metric
            return True
        return False
    
    def get_elapsed_time(self) -> float:
        """Get elapsed training time in seconds."""
        return time.time() - self.start_time


class SFTTrainerBase(ABC):
    """
    Abstract base trainer with lifecycle management.
    
    All SFT trainers should inherit from this class and implement
    the abstract methods for model, data, and training setup.
    """
    
    def __init__(self, config: SFTConfig, callbacks: Optional[List[TrainerCallback]] = None):
        """Initialize trainer with configuration."""
        self.config = config
        self.state = TrainingState()
        self.control = TrainerControl()
        
        # Initialize logging
        self.logger = SFTLogger(config.logging)
        
        # Initialize evaluator
        self.evaluator = SFTEvaluator(config)
        
        # Training components (to be set by subclasses)
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.data_loader = None
        self.trainer = None
        
        # Setup callbacks
        self.callbacks = callbacks or []
        self.callback_handler = None  # Will be initialized after model/tokenizer setup
        self.eval_dataset = None  # ADD THIS LINE
        self.custom_evaluator = None  # ADD THIS LINE
        
        logger.info(f"Initialized {self.__class__.__name__} for {config.dataset.task_type.value} task")
    
    @abstractmethod
    def setup_model(self) -> None:
        """Setup model, tokenizer, and optimization."""
        pass
    
    @abstractmethod
    def setup_data(self) -> None:
        """Setup datasets and data loaders."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute single training step."""
        pass
    
    def train(self) -> None:
        """Main training loop with hooks."""
        logger.info("Starting SFT training...")
        
        # Setup phase
        self.setup_model()
        self.setup_data()
        
        # Initialize callback handler
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, 
            optimizer=getattr(self.trainer, "optimizer", None),
            scheduler=getattr(self.trainer, "lr_scheduler", None)
        )
        self.callback_handler.add_callback(self)  # Add self as callback for simple hooks
        
        # Call on_init_end
        self.callback_handler.on_init_end(self.config, self.state, self.control)
        
        # Log initial configuration
        self.logger.log_config(self.config)
        
        # Training loop
        max_steps = self.config.train.max_steps
        if max_steps is None:
            # Calculate steps from epochs
            max_steps = self.config.train.epochs * len(self.data_loader)
        
        logger.info(f"Training for {max_steps} steps")
        
        self.callback_handler.on_train_begin(self.config, self.state, self.control)
        
        for step in range(max_steps):
            self.control.should_training_stop = False
            self.callback_handler.on_step_begin(self.config, self.state, self.control)
            
            # Get next batch
            batch = self.get_next_batch()
            
            # Execute training step
            metrics = self.train_step(batch)
            
            # Update state
            self.state.update_step(step)
            
            # Log metrics
            self.logger.log_metrics(metrics, step)
            
            self.callback_handler.on_log(self.config, self.state, self.control, logs=metrics)
            
            # Hook for step operations
            # self.on_step(step, metrics) # Handled by callback_handler now
            
            self.callback_handler.on_step_end(self.config, self.state, self.control)
            
            # Evaluation
            if step % self.config.train.eval_interval == 0 and step > 0:
                eval_metrics = self.evaluate()
                # self.on_eval(eval_metrics) # Handled by callback_handler
            
            # Checkpointing
            if step % self.config.train.save_interval == 0 and step > 0:
                self.save_checkpoint()
                
            if self.control.should_training_stop:
                logger.info("Training stopped by callback")
                break
        
        self.callback_handler.on_train_end(self.config, self.state, self.control)
        
        # Final evaluation and checkpoint
        final_eval_metrics = self.evaluate()
        # self.on_eval(final_eval_metrics)
        self.save_checkpoint()
        
        logger.info("SFT training completed")
    
    # def evaluate(
    #     self,
    #     eval_dataset=None,
    #     metric_key_prefix: str = "eval",
    #     **kwargs
    # ) -> Dict[str, float]:
    #     """Run evaluation and return metrics.
        
    #     Args:
    #         eval_dataset: Dataset to evaluate on (defaults to validation set)
    #         metric_key_prefix: Prefix for metric keys
    #         **kwargs: Additional evaluation arguments
        
    #     Returns:
    #         Dictionary of evaluation metrics
    #     """
    #     logger.info("Running evaluation...")
        
    #     # Use provided dataset or fall back to configured dataset
    #     dataset_to_use = eval_dataset if eval_dataset is not None else self.dataset
        
    #     if dataset_to_use is None:
    #         logger.warning("No evaluation dataset provided or configured")
    #         return {}
        
    #     eval_metrics = self.evaluator.evaluate(
    #         model=self.model,
    #         tokenizer=self.tokenizer,
    #         dataset=dataset_to_use,
    #         config=self.config,
    #         **kwargs
    #     )
        
    #     # Apply metric key prefix
    #     if metric_key_prefix:
    #         prefixed_metrics = {f"{metric_key_prefix}/{k}": v for k, v in eval_metrics.items()}
    #         eval_metrics = prefixed_metrics
        
    #     # Log evaluation metrics
    #     self.logger.log_metrics(eval_metrics, self.state.step, prefix="eval/")
        
    #     self.callback_handler.on_evaluate(self.config, self.state, self.control, metrics=eval_metrics)
        
    #     # Update best metric
    #     accuracy_key = f"{metric_key_prefix}/accuracy" if metric_key_prefix else "accuracy"
    #     if accuracy_key in eval_metrics:
    #         improved = self.state.update_best_metric(eval_metrics[accuracy_key])
    #         if improved:
    #             logger.info(f"New best metric: {eval_metrics[accuracy_key]:.4f}")
        
    #     return eval_metrics
    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        metric_key_prefix: str = "eval",
        use_custom_evaluator: bool = False,
        metrics: Optional[List] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Run evaluation and return metrics.
        
        Args:
            eval_dataset: Dataset to evaluate on (defaults to self.eval_dataset)
            metric_key_prefix: Prefix for metric keys
            use_custom_evaluator: If True, use BaseEvaluator. If False, use native SFTEvaluator.
            metrics: Metrics to compute (only for custom evaluator, overrides setup)
            **kwargs: Additional evaluation arguments
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation...")
        
        # Route to appropriate evaluator
        if use_custom_evaluator:
            eval_metrics = self._evaluate_with_custom(eval_dataset, metrics, **kwargs)
        else:
            eval_metrics = self._evaluate_native(eval_dataset, **kwargs)
        
        # Apply metric key prefix
        if metric_key_prefix and eval_metrics:
            prefixed_metrics = {f"{metric_key_prefix}/{k}": v for k, v in eval_metrics.items()}
            eval_metrics = prefixed_metrics
        
        # Log evaluation metrics
        if eval_metrics:
            self.logger.log_metrics(eval_metrics, self.state.step, prefix="eval/")
            self.callback_handler.on_evaluate(self.config, self.state, self.control, metrics=eval_metrics)
            
            # Update best metric
            accuracy_key = f"{metric_key_prefix}/accuracy" if metric_key_prefix else "accuracy"
            if accuracy_key in eval_metrics:
                improved = self.state.update_best_metric(eval_metrics[accuracy_key])
                if improved:
                    logger.info(f"New best metric: {eval_metrics[accuracy_key]:.4f}")
        
        return eval_metrics
    def _evaluate_with_custom(
        self,
        eval_dataset: Optional[Any],
        metrics: Optional[List],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate using BaseEvaluator (SFT only - no RL metrics)."""
        
        # Auto-setup if not configured
        if self.custom_evaluator is None:
            self.setup_custom_evaluator(evaluator_type="base", metrics=metrics)
        
        # Use provided dataset or fall back to configured dataset
        dataset = eval_dataset or self.eval_dataset or self.dataset
        
        if dataset is None:
            logger.warning("No evaluation dataset provided or configured")
            return {}
        
        # Base Evaluation (SFT)
        task_name = kwargs.pop('task_name', 'text_generation')
        return self.custom_evaluator.evaluate(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            task_name=task_name,
            **kwargs
    )

    def _evaluate_native(
        self,
        eval_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Use SFTEvaluator (native evaluation)."""
        dataset = eval_dataset or self.eval_dataset or self.dataset
        
        if dataset is None:
            logger.warning("No evaluation dataset provided or configured")
            return {}
        
        return self.evaluator.evaluate(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            config=self.config,
            **kwargs
        )
    def save_checkpoint(self) -> None:
        """Save checkpoint."""
        checkpoint_dir = Path(self.config.logging.output_dir) / f"checkpoint-{self.state.step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        
        # Save model and tokenizer
        if self.model is not None:
            self.model.save_pretrained(checkpoint_dir)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state_path = checkpoint_dir / "training_state.json"
        import json
        with open(state_path, 'w') as f:
            json.dump({
                "step": self.state.step,
                "epoch": self.state.epoch,
                "best_metric": self.state.best_metric,
                "start_time": self.state.start_time
            }, f, indent=2)
        
        self.state.checkpoint_path = str(checkpoint_dir)
        self.state.last_save_time = time.time()
        
        self.callback_handler.on_save(self.config, self.state, self.control)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load model and tokenizer
        if self.model is not None:
            self.model = self.model.from_pretrained(checkpoint_path)
        
        if self.tokenizer is not None:
            self.tokenizer = self.tokenizer.from_pretrained(checkpoint_path)
        
        # Load training state
        state_path = Path(checkpoint_path) / "training_state.json"
        if state_path.exists():
            import json
            with open(state_path, 'r') as f:
                state_data = json.load(f)
                self.state.step = state_data.get("step", 0)
                self.state.epoch = state_data.get("epoch", 0)
                self.state.best_metric = state_data.get("best_metric", 0.0)
                self.state.start_time = state_data.get("start_time", time.time())
    
    def get_next_batch(self) -> Dict[str, Any]:
        """Get next batch from data loader."""
        if self.data_loader is None:
            raise RuntimeError("Data loader not initialized. Call setup_data() first.")
        
        try:
            return next(self.data_loader)
        except (StopIteration, TypeError):
            # Restart data loader
            self.data_loader = iter(self.create_data_loader())
            return next(self.data_loader)
    
    def create_data_loader(self):
        """Create data loader - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_data_loader")
    
    def on_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Hook called after each training step."""
        pass
    
    def on_eval(self, metrics: Dict[str, float]) -> None:
        """Hook called after each evaluation."""
        pass
    
    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: str = "Upload fine-tuned model",
        **kwargs: Any,
    ) -> str:
        """Push model to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID on HuggingFace Hub (e.g., 'username/model-name')
            private: Whether the repository should be private
            token: HuggingFace token (if not provided, uses logged-in token)
            commit_message: Commit message for the upload
            **kwargs: Additional arguments for upload_folder
        
        Returns:
            URL of the uploaded repository
        
        Raises:
            RuntimeError: If model or tokenizer not loaded
            ImportError: If huggingface_hub not installed
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call train() first or load a model.")
        
        try:
            from huggingface_hub import HfApi, login
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for push_to_hub. "
                "Install with: pip install huggingface_hub"
            )
        
        # Login if token provided
        if token:
            login(token=token)
        
        # Save model first (if not already saved)
        if not hasattr(self, '_last_save_path') or self._last_save_path is None:
            save_path = self.save_model() if hasattr(self, 'save_model') else None
            if save_path is None:
                # Fallback: save to temp directory
                import tempfile
                save_path = tempfile.mkdtemp()
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
        else:
            save_path = self._last_save_path
        
        # Push to hub
        api = HfApi()
        api.upload_folder(
            folder_path=save_path,
            repo_id=repo_id,
            repo_type="model",
            private=private,
            commit_message=commit_message,
            **kwargs
        )
        
        repo_url = f"https://huggingface.co/{repo_id}"
        logger.info(f"✅ Model pushed to {repo_url}")
        return repo_url
    
    def predict(
        self,
        inputs: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Generate predictions from trained model.
        
        Args:
            inputs: Input text(s) to generate from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation arguments
        
        Returns:
            Generated text(s) - single string if input was string, list if input was list
        
        Raises:
            RuntimeError: If model or tokenizer not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call train() first or load a model.")
        
        if self.callback_handler is None:
            self.callback_handler = CallbackHandler(
                self.callbacks, self.model, self.tokenizer,
                optimizer=None,
                scheduler=None
            )
        
        
        self.model.eval()
        
        is_single = isinstance(inputs, str)
        if is_single:
            inputs = [inputs]
        
        # Tokenize
        tokenized = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self.config.model, 'max_seq_length', 512),
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove input prefix from predictions
        for i, (input_text, prediction) in enumerate(zip(inputs, predictions)):
            if prediction.startswith(input_text):
                predictions[i] = prediction[len(input_text):].strip()
            else:
                # If input wasn't at start, just return the full prediction
                predictions[i] = prediction.strip()
        
        self.callback_handler.on_prediction_step(self.config, self.state, self.control)
        
        return predictions[0] if is_single else predictions
    
    # TrainerCallback methods (can be overridden by subclasses if needed)
    def on_init_end(self, args, state, control, **kwargs):
        pass
        
    def on_train_begin(self, args, state, control, **kwargs):
        pass
        
    def on_train_end(self, args, state, control, **kwargs):
        pass
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        pass
        
    def on_epoch_end(self, args, state, control, **kwargs):
        pass
        
    def on_step_begin(self, args, state, control, **kwargs):
        pass
        
    def on_step_end(self, args, state, control, **kwargs):
        # Original hook call
        pass
        
    def on_evaluate(self, args, state, control, **kwargs):
        # Original hook call
        pass
        
    def on_save(self, args, state, control, **kwargs):
        pass
        
    def on_log(self, args, state, control, **kwargs):
        pass
        
    def on_prediction_step(self, args, state, control, **kwargs):
        pass

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if self.dataset is not None:
            del self.dataset
        if self.data_loader is not None:
            del self.data_loader
