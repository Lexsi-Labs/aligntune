"""
TRL Online DPO Trainer (Official TRL Implementation)

This module provides a TRL backend for Online DPO,
utilizing the official trl.experimental.online_dpo.OnlineDPOTrainer.
"""
import logging
import os
from typing import Dict, Any
import torch

from aligntune.core.rl import TrainerBase
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.model_loader import build_model, build_ref_model, build_reward_model
from aligntune.data.manager import DataManager
from aligntune.core.registry import TaskType
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


def _online_dpo_trainer_classes():
    """Import TRL OnlineDPO without requiring a working vLLM install.

    OnlineDPOTrainer always imports VLLMGeneration. If a CUDA-13 vLLM wheel
    is present, that import fails and create_rl_trainer reports
    ``No available backend for TrainingType.RL RLAlgorithm.ONLINE_DPO``.
    """
    if os.environ.get("ALIGNTUNE_ENABLE_VLLM", "0") != "1":
        try:
            import trl.import_utils as _iu
            _iu.is_vllm_available = lambda *a, **k: False
        except Exception:
            pass
    from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer
    return OnlineDPOConfig, OnlineDPOTrainer

class TRLOnlineDPOTrainer(TrainerBase):
    """Online DPO trainer using official TRL OnlineDPOTrainer."""

    TASK_TYPE = "grpo"  # Online DPO uses prompt-only (GRPO format) for generation
    KEEP_COLUMNS = True

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.model = None
        self.reference_model = None
        self.tokenizer = None
        self.reward_model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.dpo_config = None
        
    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL Online DPO trainer is available."""
        try:
            _online_dpo_trainer_classes()
            return True
        except ImportError:
            return False

    def setup_model(self) -> None:
        """Setup model and reference model using core loaders."""
        logger.info("Setting up Online DPO models using core loaders...")

        # Load policy model and tokenizer
        use_unsloth = getattr(self.config.model, 'use_unsloth', False)
        self.model, self.tokenizer = build_model(
            self.config,
            task_type=TaskType.SFT,
            use_unsloth=use_unsloth,
            apply_peft=getattr(self.config.model, 'use_peft', True)
        )

        # Load reference model
        self.reference_model = build_ref_model(self.config, base_model=self.model, use_unsloth=use_unsloth)

    def setup_data(self) -> None:
        """Setup prompts dataset using unified DataManager."""
        logger.info("Setting up prompts dataset with DataManager...")
        
        dataset_config = None
        if hasattr(self.config, 'dataset'):
            dataset_config = self.config.dataset
        elif hasattr(self.config, 'datasets') and len(self.config.datasets) > 0:
            dataset_config = self.config.datasets[0]
            
        dataset_name = self._get_config_value(dataset_config, 'name', 'dataset_name')
        split = self._get_config_value(dataset_config, 'split', default='train')
        config_name = self._get_config_value(dataset_config, 'config_name', default=None)
        val_split_ratio = self._get_config_value(
            dataset_config, 'val_split_ratio', default=None
        )
        test_split_ratio = self._get_config_value(
            dataset_config, 'test_split_ratio', default=None
        )
        split_seed = self._get_config_value(dataset_config, 'split_seed', default=42)
        
        manager = DataManager(
            task_type="grpo",  # GRPO mode returns prompt-only datasets suitable for OnlineDPO
            tokenizer=self.tokenizer,
            enable_thinking=getattr(self.config.train, 'enable_thinking', False),
            max_samples=self._get_config_value(dataset_config, 'max_samples', default=None),
            val_split_ratio=val_split_ratio,
            test_split_ratio=test_split_ratio,
            seed=split_seed,
        )
        
        dataset_dict = manager.load_dataset(dataset_name, config_name=config_name, split=split)
        self.train_dataset = dataset_dict["train"]
        self.eval_dataset = dataset_dict.get("validation", None)

    def setup_rewards(self) -> None:
        """Prepare rewards using the shared TRL reward bridge."""
        from aligntune.core.rl.reward_handler import (
            prepare_trl_rewards,
            resolve_trl_reward_weights,
        )

        reward_funcs = []
        reward_weights = []

        reward_model_name = getattr(self.config.model, "reward_model_name", None)
        if reward_model_name:
            self.reward_model = build_reward_model(self.config)
            # trl's (experimental) OnlineDPOTrainer distinguishes "model-based"
            # vs "custom function" rewards purely via `isinstance(reward_func,
            # nn.Module)` (see trl/experimental/online_dpo/online_dpo_trainer.py).
            # build_reward_model() returns the model wrapped in
            # UniversalRewardModelWrapper, which is a plain Python object (not
            # an nn.Module) so it fails that check, gets treated as a "custom
            # reward function", and trl tries to call it as
            # `reward_func(prompts=..., completions=..., ...)` /
            # access `.__name__` on it - both of which crash. Unwrap back to
            # the underlying nn.Module here so trl takes its normal
            # model-based reward path.
            reward_model_for_trl = getattr(self.reward_model, "_model", self.reward_model)
            reward_funcs.append(reward_model_for_trl)
            reward_weights.append(1.0)

        # 2. Load reward functions from config.rewards
        rewards_config = []
        if hasattr(self.config, 'rewards'):
            rewards_config = self.config.rewards if isinstance(self.config.rewards, list) else []

        if rewards_config:
            prepared = prepare_trl_rewards(rewards_config)
            reward_funcs.extend(prepared.functions)
            explicit_weights = getattr(self.config.train, "reward_weights", None)
            reward_weights.extend(resolve_trl_reward_weights(prepared, explicit_weights))

        if not reward_funcs:
            raise ValueError(
                "Online DPO requires a reward_model_name or at least one configured reward."
            )

        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights

    def setup_trainer(self) -> None:
        """Setup Online DPO trainer configuration."""
        logger.info("Setting up Online DPO trainer configuration...")

        OnlineDPOConfig, _ = _online_dpo_trainer_classes()
        from aligntune.core.precision_handler import PrecisionHandler

        train_config = self.config.train
        explicit_params = getattr(train_config, 'extra_params', {}) or {}
        output_dir = self._get_config_value(self.config.logging, 'output_dir', default="./outputs")

        precision = PrecisionHandler.get_precision_from_config(self.config, default='auto')
        precision_flags = PrecisionHandler.get_training_args_precision(precision)

        # Build config with all generation parameters
        self.dpo_config = OnlineDPOConfig(
            output_dir=output_dir,
            # Training params
            num_train_epochs=getattr(train_config, 'epochs', 3),
            max_steps=getattr(train_config, 'max_steps', -1),
            per_device_train_batch_size=getattr(train_config, 'per_device_batch_size', 4),
            per_device_eval_batch_size=getattr(train_config, 'per_device_eval_batch_size', 4),
            gradient_accumulation_steps=getattr(train_config, 'gradient_accumulation_steps', 2),
            learning_rate=getattr(train_config, 'learning_rate', 5e-7),
            weight_decay=getattr(train_config, 'weight_decay', 0.0),
            warmup_steps=getattr(train_config, 'warmup_steps', 0),
            fp16=precision_flags['fp16'],
            bf16=precision_flags['bf16'],

            # Generation parameters - KEY for multiple completions
            max_new_tokens=getattr(train_config, 'max_new_tokens', 64),
            max_length=getattr(train_config, 'max_length', 512),
            temperature=getattr(train_config, 'temperature', 0.9),
            top_p=getattr(train_config, 'top_p', 1.0),
            top_k=getattr(train_config, 'top_k', 0),
            repetition_penalty=getattr(train_config, 'repetition_penalty', 1.0),

            # Online DPO specific
            beta=getattr(train_config, 'beta', 0.1),
            loss_type=self._get_config_value(
                train_config, 'loss_type', default='sigmoid'
            ),
            missing_eos_penalty=getattr(train_config, 'missing_eos_penalty', 1.0),
            reward_weights=getattr(train_config, 'reward_weights', None),

            # Evaluation and saving
            eval_strategy=explicit_params.get('eval_strategy', getattr(train_config, 'eval_strategy', 'no')),
            eval_steps=self._get_config_value(
                train_config, 'eval_steps', 'eval_interval', default=500
            ),
            logging_steps=getattr(train_config, 'logging_steps', 10),
            logging_strategy=getattr(train_config, 'logging_strategy', 'steps'),
            save_strategy=explicit_params.get('save_strategy', getattr(train_config, 'save_strategy', 'steps')),
            save_steps=getattr(train_config, 'save_steps', 500),
            save_total_limit=getattr(train_config, 'save_total_limit', 3),
        )

        # Backfill anything else the caller set (e.g. max_grad_norm,
        # warmup_ratio, seed, ...) that the explicit list above doesn't
        # already cover.
        missing = extract_extra_and_missing_params(
            backend_config=self.dpo_config, config=self.config, algorithm='online_dpo'
        )
        for key, value in missing.items():
            setattr(self.dpo_config, key, value)

        if not self.dpo_config.loss_type:
            self.dpo_config.loss_type = 'sigmoid'

        if getattr(self, 'reward_weights', None) and not getattr(train_config, 'reward_weights', None):
            self.dpo_config.reward_weights = self.reward_weights

        logger.info("Online DPO trainer config created successfully")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return {}

    def train(self) -> Dict[str, Any]:
        """Execute Online DPO training loop."""
        logger.info("Starting Online DPO training using official TRL trainer...")

        self.setup_model()

        # Only setup data if not already manually set
        if self.train_dataset is None:
            self.setup_data()

        self.setup_rewards()
        self.setup_trainer()

        _, OnlineDPOTrainer = _online_dpo_trainer_classes()

        output_dir = self._get_config_value(self.config.logging, 'output_dir', default="./outputs")

        self.trainer = OnlineDPOTrainer(
            model=self.model,
            ref_model=self.reference_model,
            reward_funcs=self.reward_funcs,
            args=self.dpo_config,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

        # Train
        self.trainer.train()
        self.trainer.save_model(output_dir)

        return {"status": "success", "output_dir": output_dir}
