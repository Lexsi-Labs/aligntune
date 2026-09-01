"""
Unsloth Online DPO Trainer (Official TRL Implementation, Unsloth-accelerated)

This module provides an Unsloth-optimized backend for Online DPO,
utilizing the official trl.experimental.online_dpo.OnlineDPOTrainer with
Unsloth's performance optimizations applied to the policy and reference models.
"""
import logging
from typing import Dict, Any
import torch

from aligntune.core.rl import TrainerBase
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.model_loader import build_model, build_ref_model, build_reward_model
from aligntune.data.manager import DataManager
from aligntune.core.registry import TaskType
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


class _UnslothCausalLMRewardHead(torch.nn.Module):
    """Adapts an Unsloth-loaded causal LM into a sequence-classification-style
    reward model for TRL's OnlineDPOTrainer.

    build_reward_model()'s 'unsloth' loading path loads the reward model via
    FastLanguageModel.from_pretrained (a causal LM) and bolts on an unused
    `.score` Linear layer - the model's forward still runs through `lm_head`
    and returns vocab-sized logits, not a scalar reward. This wraps that
    causal LM so forward() instead pools the backbone's last-non-pad-token
    hidden state through the score head, mirroring what
    transformers' *ForSequenceClassification models do internally.
    """

    def __init__(self, causal_lm):
        super().__init__()
        self.causal_lm = causal_lm
        self.config = causal_lm.config

        existing_score = getattr(causal_lm, "score", None)
        if isinstance(existing_score, torch.nn.Linear):
            self.score = existing_score
        else:
            self.score = torch.nn.Linear(causal_lm.config.hidden_size, 1, bias=False)
        param = next(causal_lm.parameters())
        self.score = self.score.to(device=param.device, dtype=param.dtype)

        pad_token_id = getattr(causal_lm.config, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else getattr(causal_lm.config, "eos_token_id", None)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        backbone = getattr(self.causal_lm, self.causal_lm.base_model_prefix)
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]

        logits = self.score(hidden_states)  # [batch, seq_len, 1]

        batch_size = input_ids.shape[0]
        if self.pad_token_id is None:
            last_non_pad_token = -1
        else:
            non_pad_mask = (input_ids != self.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]  # [batch, 1]

        from transformers.modeling_outputs import SequenceClassifierOutputWithPast
        return SequenceClassifierOutputWithPast(logits=pooled_logits)


class UnslothOnlineDPOTrainer(TrainerBase):
    """Online DPO trainer using official TRL OnlineDPOTrainer, accelerated with Unsloth."""

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
        """Check if Unsloth Online DPO trainer is available."""
        try:
            import unsloth
            from trl.experimental.online_dpo import OnlineDPOTrainer, OnlineDPOConfig
            return True
        except ImportError:
            return False

    def setup_model(self) -> None:
        """Setup model and reference model using core loaders, with Unsloth acceleration."""
        logger.info("Setting up Unsloth Online DPO models using core loaders...")

        # CRITICAL: Patch attention classes (adds apply_qkv/apply_o where Unsloth
        # doesn't define them, e.g. Qwen2Attention) BEFORE any model loading, same
        # as the Unsloth PPO backend - Online DPO also generates completions during
        # training and hits the same missing-attribute path.
        from ..ppo.unsloth_patches import clear_all_unsloth_caches, patch_attention_classes_globally
        clear_all_unsloth_caches()
        patch_attention_classes_globally()

        # Load policy model and tokenizer (Unsloth-accelerated)
        self.model, self.tokenizer = build_model(
            self.config,
            task_type=TaskType.SFT,
            use_unsloth=True,
            apply_peft=getattr(self.config.model, 'use_peft', True)
        )

        # Load reference model (Unsloth-accelerated)
        self.reference_model = build_ref_model(self.config, base_model=self.model, use_unsloth=True)

        # Qwen (and many other) checkpoints ship a generation_config.json with
        # a large default max_length (e.g. 32768). TRL's OnlineDPOTrainer
        # builds its own GenerationConfig from only max_new_tokens/temperature/
        # etc, leaving max_length unset; transformers' generate() then falls
        # back to the model's own generation_config.max_length as the base
        # and warns that both are "set". Clearing it here is harmless -
        # max_new_tokens already fully determines generation length for
        # Online DPO - and removes the per-step warning spam.
        self.model.generation_config.max_length = None
        if self.reference_model is not None and hasattr(self.reference_model, 'generation_config'):
            self.reference_model.generation_config.max_length = None

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
        processing_classes = []
        reward_weights = []

        reward_model_name = getattr(self.config.model, "reward_model_name", None)
        if reward_model_name:
            self.reward_model = build_reward_model(self.config)
            reward_model_for_trl = getattr(self.reward_model, "_model", self.reward_model)
            reward_tokenizer = None
            reward_weights.append(1.0)
            loading_type = getattr(self.config.model, "reward_value_loading_type", None) or "standard"
            if loading_type == "unsloth" and not hasattr(reward_model_for_trl, "classifier"):
                reward_model_for_trl = _UnslothCausalLMRewardHead(reward_model_for_trl)
                reward_model_for_trl.eval()

                from transformers import AutoTokenizer

                reward_tokenizer = AutoTokenizer.from_pretrained(
                    reward_model_name, trust_remote_code=True
                )
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                reward_model_for_trl.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_model_for_trl.pad_token_id = reward_tokenizer.pad_token_id

            reward_funcs.append(reward_model_for_trl)
            processing_classes.append(reward_tokenizer)

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
        self.reward_processing_classes = processing_classes
        self.reward_weights = reward_weights

    def setup_trainer(self) -> None:
        """Setup Online DPO trainer configuration."""
        logger.info("Setting up Unsloth Online DPO trainer configuration...")

        from trl.experimental.online_dpo import OnlineDPOConfig
        from aligntune.core.precision_handler import PrecisionHandler

        train_config = self.config.train
        explicit_params = getattr(train_config, 'extra_params', {}) or {}
        output_dir = self._get_config_value(self.config.logging, 'output_dir', default="./outputs")

        # OnlineDPOConfig defaults bf16=True unless fp16 is set, regardless of
        # hardware support. Resolve the actual fp16/bf16 flags for the
        # detected GPU (or an explicit precision from config) instead of
        # relying on that default, so this doesn't crash on pre-Ampere GPUs.
        precision = PrecisionHandler.get_precision_from_config(self.config, default='auto')
        precision_flags = PrecisionHandler.get_training_args_precision(precision)

        # Build config with all generation parameters
        self.dpo_config = OnlineDPOConfig(
            output_dir=output_dir,
            run_name=self._get_config_value(self.config.logging, 'run_name', default='unsloth_online_dpo'),
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
        missing = extract_extra_and_missing_params(
            backend_config=self.dpo_config,
            config=self.config,
            algorithm='online_dpo',
        )
        for key, value in missing.items():
            setattr(self.dpo_config, key, value)

        if not self.dpo_config.loss_type:
            self.dpo_config.loss_type = 'sigmoid'
        if getattr(self, 'reward_weights', None) and not getattr(train_config, 'reward_weights', None):
            self.dpo_config.reward_weights = self.reward_weights
        logger.info("Unsloth Online DPO trainer config created successfully")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return {}

    def train(self) -> Dict[str, Any]:
        """Execute Online DPO training loop."""
        logger.info("Starting Online DPO training using Unsloth-accelerated TRL trainer...")

        self.setup_model()

        # Only setup data if not already manually set
        if self.train_dataset is None:
            self.setup_data()

        self.setup_rewards()
        self.setup_trainer()

        from trl.experimental.online_dpo import OnlineDPOTrainer

        output_dir = self._get_config_value(self.config.logging, 'output_dir', default="./outputs")

        self.trainer = OnlineDPOTrainer(
            model=self.model,
            ref_model=self.reference_model,
            reward_funcs=self.reward_funcs,
            reward_processing_classes=self.reward_processing_classes,
            args=self.dpo_config,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

        # Train
        self.trainer.train()
        self.trainer.save_model(output_dir)

        return {"status": "success", "output_dir": output_dir}
