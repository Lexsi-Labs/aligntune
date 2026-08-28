"""
Abstract base trainer for RLHF training, inheriting from UnifiedTrainerBase.
"""

import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Union

from .config import UnifiedConfig
from .distributed import BackendFactory
from .logging_utils import UnifiedLogger
from .evaluator import UnifiedEvaluator
from ..trainer_base import UnifiedTrainerBase, TrainingState
from ..callbacks import CallbackHandler, TrainerCallback

from ...eval import BaseEvaluator, RLEvaluator
from ...eval.metrics import RougeMetric, BleuMetric, PerplexityMetric

logger = logging.getLogger(__name__)


class RLTrainerBase(UnifiedTrainerBase):
    """
    RLHF-specific base trainer. Inherits lifecycle, saving, and callbacks from UnifiedTrainerBase.
    Adds support for Reward/Value models and complex RL train loops.
    """
    # Algorithm backends override this. Users can override it through the
    # public dataset `keep_columns` option.
    KEEP_COLUMNS = False
    
    def __init__(self, config: UnifiedConfig, callbacks: Optional[List[TrainerCallback]] = None):
        super().__init__(config, callbacks)
        
        self.backend = BackendFactory.create(config.distributed)
        self.logger = UnifiedLogger(config.logging)
        self.evaluator = UnifiedEvaluator(config)
        
        self.reward_functions = []
        self.ref_model = None
        self.reward_model = None
        self.value_model = None
        
        self.base_evaluator = None
        self.rl_evaluator = None
        
        logger.info(f"Initialized {self.__class__.__name__} with {config.algo.value} algorithm")

    @abstractmethod
    def setup_model(self) -> None:
        pass
        
    def setup_data(self) -> None:
        """
        Unified setup_data implementation for RL trainers.
        Uses DataManager to load and process datasets.
        Override this method only if you need custom dataset handling.
        """
        logger.info("Setting up RL datasets with unified DataManager...")

        task_type = self._get_task_type()
        dataset_config = self._extract_dataset_config()
        params = self._extract_dataset_params(dataset_config)
        keep_columns = (
            params['keep_columns']
            if params['keep_columns'] is not None
            else self.KEEP_COLUMNS
        )

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
            keep_columns=keep_columns,
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

        # Look up whatever split was actually requested/loaded first (DataManager
        # keeps the requested split's own name as the dict key rather than always
        # relabeling it "train" - see DataManager.load_dataset). Falling back to
        # "train" preserves the previous behavior when split is None/"train".
        requested_split = params['split']
        self.train_dataset = dataset_dict.get(requested_split, dataset_dict.get("train", None))
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
        if hasattr(self.config, 'algo'):
            return self.config.algo.value.lower()
        return "ppo"

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
            'config_name': self._get_config_value(dataset_config, 'config_name', default=None),
            'system_prompt': self._get_config_value(dataset_config, 'system_prompt', default=None),
            'enable_thinking': self._get_config_value(self.config.train, 'enable_thinking', default=False),
            'column_mapping': self._get_config_value(dataset_config, 'column_mapping', default=None),
            'processing_fn': self._get_config_value(dataset_config, 'processing_fn', default=None),
            'processing_batched': self._get_config_value(dataset_config, 'processing_batched', default=False),
            'max_samples': self._get_config_value(dataset_config, 'max_samples', default=None),
            'format_type': self._get_config_value(dataset_config, 'format_type', default=None),
            'keep_columns': self._get_config_value(dataset_config, 'keep_columns', default=None),
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

    def _setup_optimizer_scheduler(self, dataset_for_estimation=None):
        """Setup optimizer and scheduler configuration. Returns common params for all RL algorithms."""
        max_steps = self._get_config_value(self.config.train, 'max_steps', default=-1)
        num_epochs = self._get_config_value(self.config.train, 'epochs', 'num_epochs', default=1)
        warmup_steps = self._get_config_value(self.config.train, 'warmup_steps', default=0)
        warmup_ratio = self._get_config_value(self.config.train, 'warmup_ratio', default=0.1)

        return {
            'optimizer_type': self._get_config_value(self.config.train, 'optimizer_type', default='adamw_torch'),
            'learning_rate': self._get_config_value(self.config.train, 'learning_rate', default=5e-5),
            'lr_scheduler_type': self._get_config_value(self.config.train, 'lr_scheduler_type', default='linear'),
            'warmup_steps': warmup_steps,
            'warmup_ratio': warmup_ratio,
            'max_steps': max_steps,
            'num_epochs': num_epochs,
        }

    def setup_rewards(self) -> None:
        """Prepare configured registry rewards for TRL-native online trainers."""
        from aligntune.core.rl.reward_handler import prepare_trl_rewards

        reward_specs = self.config.rewards if isinstance(self.config.rewards, list) else []
        self.prepared_rewards = prepare_trl_rewards(reward_specs)
        self.reward_functions = self.prepared_rewards.functions
        
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute single RL training step."""
        pass
        
    def create_data_loader(self):
        raise NotImplementedError("Subclasses must implement create_data_loader")
        
    def get_next_batch(self) -> Dict[str, Any]:
        if self.data_loader is None:
            raise RuntimeError("Data loader not initialized.")
        try:
            return next(self.data_loader)
        except (StopIteration, TypeError):
            self.data_loader = iter(self.create_data_loader())
            return next(self.data_loader)

    def train(self) -> None:
        """Manual training loop for RL algorithms."""
        logger.info("Starting RL training...")
        self.setup_model()
        self.setup_data()
        self.setup_rewards()
        
        if self.data_loader is None:
            self.data_loader = self.create_data_loader()
            
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, 
            optimizer=None, scheduler=None
        )
        self.callback_handler.add_callback(self)
        self.callback_handler.on_init_end(self.config, self.state, self.control)
        
        self.logger.log_config(self.config)
        
        max_steps = self.config.train.max_steps
        if max_steps is None:
            if self.config.train.epochs is not None:
                max_steps = self.config.train.epochs * len(self.data_loader)
            else:
                raise ValueError("max_steps or epochs required")
                
        logger.info(f"Training for {max_steps} steps")
        self.callback_handler.on_train_begin(self.config, self.state, self.control)
        
        for step in range(max_steps):
            self.control.should_training_stop = False
            self.callback_handler.on_step_begin(self.config, self.state, self.control)
            
            batch = self.get_next_batch()
            metrics = self.train_step(batch)
            
            self.state.update_step(step)
            self.logger.log_metrics(metrics, step)
            self.callback_handler.on_log(self.config, self.state, self.control, logs=metrics)
            self.callback_handler.on_step_end(self.config, self.state, self.control)
            
            if step % self.config.train.eval_interval == 0 and step > 0:
                self.evaluate()
                
            if step % self.config.train.save_interval == 0 and step > 0:
                self.save_checkpoint()
                
            if self.control.should_training_stop:
                break
                
        self.callback_handler.on_train_end(self.config, self.state, self.control)
        self.evaluate()
        self.save_checkpoint()
        self._auto_export()
        logger.info("RL Training completed")

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """RL-specific evaluation (KL Divergence, Rewards, etc)."""
        if not self.backend.is_rank_0():
            return {}
            
        logger.info("Running RL Evaluation...")
        
        eval_results = {}
        if hasattr(self, 'rl_evaluator') and self.rl_evaluator:
            eval_results = self.rl_evaluator.evaluate_rl(
                policy_model=self.model,
                reference_model=self.ref_model or self.model,
                tokenizer=self.tokenizer,
                dataset=self.eval_dataset or self.dataset,
                reward_model=self.reward_model
            )
            
        if eval_results:
            self.logger.log_metrics(eval_results, self.state.step, prefix="eval/")
            if self.callback_handler:
                self.callback_handler.on_evaluate(self.config, self.state, self.control, metrics=eval_results)
                
        return eval_results

    def save_model(self, path: Optional[str] = None) -> str:
        """Save trained model."""
        from pathlib import Path

        save_path = Path(path or self.config.logging.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to: {save_path}")
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        return str(save_path)

    def load_model(self, path: str) -> None:
        """Load trained model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model from: {path}")
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info("Model loaded successfully")


# Backward compatibility alias for Unsloth trainers that import from this module
TrainerBase = RLTrainerBase
