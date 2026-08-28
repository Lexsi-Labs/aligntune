"""
Unsloth SPIN Trainer - Self-Play Fine-Tuning Implementation

This module implements SPIN (Self-Play Improvement through No-regret learning),
which generates synthetic preference pairs by playing the current model against
a frozen opponent checkpoint. The SFT response is chosen and the opponent's
response is rejected, creating fully synthetic preference data.

Unsloth-optimized: policy and reference models are loaded via Unsloth for
faster training and reduced memory usage.

Paper: https://arxiv.org/abs/2404.04291
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import tempfile
import shutil
from copy import deepcopy

from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.rl.registries import DatasetRegistry, RewardRegistry
from aligntune.core.rl.caching import DatasetCache
from aligntune.core.precision_handler import PrecisionHandler
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


def _ensure_boundary_space(prompt: str, completion: str) -> str:
    """Insert a separating space if prompt/completion would otherwise be glued together.

    TRL's DPOTrainer tokenizes `prompt` and `prompt + completion` independently
    and expects the former to be a token-prefix of the latter. Text with no
    whitespace at the join point (e.g. "...answer.The answer is 4") commonly
    breaks that expectation under BPE tokenization.
    """
    if prompt and completion and not prompt[-1].isspace() and not completion[:1].isspace():
        return " " + completion
    return completion


class UnslothSPINTrainer(TrainerBase):
    """
    SPIN Trainer using TRL's DPOTrainer internally, with Unsloth-accelerated models.

    Self-Play Improvement through No-regret learning (SPIN):
    - Generates synthetic preference pairs by comparing current model vs frozen opponent
    - Chosen: SFT reference response
    - Rejected: Opponent model's generated response
    - Updates opponent checkpoint after each round
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize SPIN trainer."""
        super().__init__(config)
        self.model = None
        self.opponent_model = None
        self.reference_model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.sft_dataset = None
        self.dataset_cache = None
        self.dataset_dict = None

        # SPIN-specific configuration
        self.num_rounds = self._get_config_value(config.train, 'num_rounds', default=2)
        self.generation_temperature = self._get_config_value(config.train, 'generation_temperature', default=0.7)
        self.generation_max_length = self._get_config_value(config.train, 'generation_max_length', default=512)
        self.generation_batch_size = self._get_train_value(
            'generation_batch_size', default=self._get_config_value(config.train, 'per_device_batch_size', default=8)
        )
        self.generation_prompt_length = self._get_train_value('generation_max_prompt_length', default=None)
        if self.generation_prompt_length is None:
            self.generation_prompt_length = self._get_train_value('max_prompt_length', default=512)
        self.generation_top_p = self._get_train_value('generation_top_p', default=None)
        if self.generation_top_p is None:
            self.generation_top_p = self._get_train_value('top_p', default=0.95)
        self.generation_top_k = self._get_train_value('generation_top_k', default=None)
        if self.generation_top_k is None:
            self.generation_top_k = self._get_train_value('top_k', default=0)
        self.generation_repetition_penalty = self._get_train_value(
            'generation_repetition_penalty', default=None
        )
        if self.generation_repetition_penalty is None:
            self.generation_repetition_penalty = self._get_train_value('repetition_penalty', default=1.0)
        self.generation_do_sample = self._get_train_value('generation_do_sample', default=None)
        self.generation_kwargs = self._get_train_value('generation_kwargs', default={}) or {}
        self.enable_thinking = self._get_train_value('enable_thinking', default=False)
        self.samples_per_round = self._get_config_value(config.train, 'samples_per_round', default=None)
        self.eval_samples = self._get_config_value(config.train, 'eval_samples', default=None)
        # NOTE: `max_steps` (the standard knob every other algorithm reads)
        # is a completely separate config field from `dpo_steps_per_round`
        # and was never consulted here, so passing max_steps=N had no effect
        # on SPIN and it silently trained dpo_steps_per_round (default 100)
        # steps per round instead. Prefer an explicit max_steps when given.
        _max_steps_override = self._get_config_value(config.train, 'max_steps', default=None)
        if _max_steps_override is not None and _max_steps_override > 0:
            self.dpo_steps_per_round = _max_steps_override
        else:
            self.dpo_steps_per_round = self._get_config_value(config.train, 'dpo_steps_per_round', default=100)

        self.current_round = 0
        self.opponent_checkpoint_dir = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth DPO trainer is available for SPIN."""
        try:
            import unsloth
            from trl import DPOTrainer, DPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return True
        except ImportError:
            return False

    def _get_config_value(self, config_obj, *attr_names, default=None):
        """Safely get config value from multiple possible attribute names."""
        if isinstance(config_obj, dict):
            for attr_name in attr_names:
                if attr_name in config_obj and config_obj[attr_name] is not None:
                    return config_obj[attr_name]
        else:
            for attr_name in attr_names:
                if hasattr(config_obj, attr_name):
                    value = getattr(config_obj, attr_name)
                    if value is not None:
                        return value
        return default

    def _get_train_value(self, *attr_names, default=None):
        """Read a train field, then an untyped factory kwarg if present."""
        value = self._get_config_value(self.config.train, *attr_names, default=None)
        if value is not None:
            return value
        extra_params = self._get_config_value(self.config.train, 'extra_params', default={}) or {}
        return self._get_config_value(extra_params, *attr_names, default=default)

    # Required abstract methods
    def setup_data(self) -> None:
        """Setup data - delegates to setup_dataset."""
        self.setup_dataset()

    def setup_rewards(self) -> None:
        """Setup rewards - not used in SPIN (uses synthetic preferences)."""
        logger.info("SPIN uses synthetic preference pairs instead of explicit rewards")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step - handled internally by TRL DPOTrainer."""
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call train() first.")
        return {}

    def setup_model(self) -> None:
        """Setup model and tokenizer for SPIN via unified model_loader (Unsloth-backed)."""
        try:
            from aligntune.core.model_loader import build_model
            from aligntune.core.registry import TaskType

            logger.info("=" * 80)
            logger.info("Setting up Unsloth SPIN models via model_loader")
            logger.info("=" * 80)

            # Check if PEFT should be applied
            peft_enabled = getattr(self.config.model, 'use_peft', False) or \
                           getattr(self.config.model, 'load_in_4bit', False) or \
                           getattr(self.config.model, 'load_in_8bit', False)

            # Load policy model with PEFT if needed
            self.model, self.tokenizer = build_model(
                self.config,
                task_type=TaskType.SFT,
                apply_peft=peft_enabled,
                use_unsloth=True
            )

            # SPIN generates opponent completions during training; decoder-only
            # models need left-padding for correct generation (see ppo.py).
            if self.tokenizer.padding_side != "left":
                self.tokenizer.padding_side = "left"

            # Load reference model for DPO
            if not peft_enabled:
                logger.info("Full fine-tuning: Loading frozen reference model")
                self.reference_model, _ = build_model(
                    self.config,
                    task_type=TaskType.SFT,
                    is_reference=True,
                    use_unsloth=True
                )
            else:
                logger.info("PEFT enabled: TRL DPO handles reference via adapter toggle")
                self.reference_model = None

            # Initialize opponent checkpoint as copy of model
            self._initialize_opponent_checkpoint()

            logger.info("Unsloth SPIN model setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup Unsloth SPIN model: {e}")
            raise

    def _initialize_opponent_checkpoint(self) -> None:
        """Initialize opponent checkpoint directory (not used, opponent is self.model)."""
        try:
            # Create temporary directory for opponent checkpoint
            # NOTE: We don't actually save here anymore - opponent is just self.model
            self.opponent_checkpoint_dir = tempfile.mkdtemp(prefix="spin_opponent_")
            logger.info(f"Initialized opponent checkpoint directory: {self.opponent_checkpoint_dir}")
            logger.info("NOTE: Opponent is self.model directly (no separate checkpoint needed)")

        except Exception as e:
            logger.error(f"Failed to initialize opponent checkpoint: {e}")
            raise

    def setup_dataset(self) -> None:
        """Setup SFT dataset for SPIN."""
        try:
            logger.info("Setting up SPIN datasets...")

            # Extract dataset configuration
            dataset_config = None
            if hasattr(self.config, 'dataset'):
                dataset_config = self.config.dataset
            elif hasattr(self.config, 'datasets') and len(self.config.datasets) > 0:
                dataset_config = self.config.datasets[0]
            else:
                raise ValueError("No dataset configuration found")

            # Extract parameters
            dataset_name = self._get_config_value(
                dataset_config, 'name', 'dataset_name', default='imdb'
            )
            split = self._get_config_value(dataset_config, 'split', default=None)
            config_name = self._get_config_value(dataset_config, 'config_name', default=None)
            system_prompt = self._get_config_value(dataset_config, 'system_prompt', default=None)
            enable_thinking = self.enable_thinking
            val_split_ratio = self._get_config_value(
                dataset_config, 'val_split_ratio', default=None
            )
            test_split_ratio = self._get_config_value(
                dataset_config, 'test_split_ratio', default=None
            )
            split_seed = self._get_config_value(
                dataset_config, 'split_seed', default=42
            )

            # Advanced DataManager features
            column_mapping = self._get_config_value(dataset_config, 'column_mapping', default=None)
            processing_fn = self._get_config_value(dataset_config, 'processing_fn', default=None)
            processing_batched = self._get_config_value(dataset_config, 'processing_batched', default=False)
            max_samples = self._get_config_value(dataset_config, 'max_samples', default=None)
            max_eval_samples = self._get_config_value(dataset_config, 'max_eval_samples', default=None)

            logger.info(f"Loading SFT dataset: {dataset_name} (split: {split}, config: {config_name})")

            # Initialize DataManager for SFT task
            from aligntune.data.manager import DataManager

            manager = DataManager(
                task_type="sft",
                system_prompt=system_prompt,
                tokenizer=self.tokenizer,
                enable_thinking=enable_thinking,
                column_mapping=column_mapping,
                processing_fn=processing_fn,
                max_samples=max_samples,
                processing_batched=processing_batched,
                val_split_ratio=val_split_ratio,
                test_split_ratio=test_split_ratio,
                seed=split_seed,
            )

            # Load dataset
            dataset_dict = manager.load_dataset(
                dataset_name,
                config_name=config_name,
                split=split,
            )

            # Store as SFT dataset (for generating chosen responses)
            self.sft_dataset = dataset_dict.get("train", None)
            self.train_dataset = self.sft_dataset  # For compatibility
            self.eval_dataset = dataset_dict.get("validation", None)
            self.dataset_dict = dataset_dict

            # Keep a deterministic validation set across all SPIN rounds and
            # shuffle the training pool once so each round can consume a new
            # slice without materializing generated responses for the full set.
            shuffle_seed = self._get_train_value('data_seed', 'seed', default=42)
            if self.sft_dataset is not None:
                self.train_pool = self.sft_dataset.shuffle(seed=shuffle_seed)
            else:
                self.train_pool = None
            if self.eval_dataset is not None:
                self.eval_dataset = self.eval_dataset.shuffle(seed=shuffle_seed)
                if self.eval_samples is not None:
                    self.eval_dataset = self.eval_dataset.select(
                        range(min(self.eval_samples, len(self.eval_dataset)))
                    )

            if self.sft_dataset:
                logger.info(f"SFT dataset loaded: {len(self.sft_dataset)} samples")
            if self.eval_dataset:
                logger.info(f"Evaluation dataset loaded: {len(self.eval_dataset)} samples")

        except Exception as e:
            logger.error(f"Failed to setup SPIN datasets: {e}")
            raise

    def setup_trainer(self) -> None:
        """Setup DPO trainer for SPIN rounds with Unsloth model."""
        try:
            from trl import DPOTrainer, DPOConfig

            logger.info("Setting up DPO trainer for SPIN rounds with Unsloth model")

            # Get optimizer, scheduler, and training params from base class
            optim_scheduler = self._setup_optimizer_scheduler(dataset_for_estimation=self.train_dataset)

            # Extract values from returned dict
            max_steps = optim_scheduler['max_steps']
            num_epochs = optim_scheduler['num_epochs']

            # Validate and set max_length and max_prompt_length
            max_length = self._get_config_value(self.config.train, 'max_length', 'max_seq_length', default=2048)
            max_prompt_length = self._get_config_value(self.config.train, 'max_prompt_length', default=None)

            # If max_prompt_length not set, default to half of max_length
            if max_prompt_length is None:
                max_prompt_length = max_length // 2
                logger.info(f"max_prompt_length not set, defaulting to max_length // 2 = {max_prompt_length}")

            # Validate: max_prompt_length must be < max_length
            if max_prompt_length >= max_length:
                old_prompt_length = max_prompt_length
                max_prompt_length = max_length // 2
                logger.warning(
                    f"max_prompt_length ({old_prompt_length}) >= max_length ({max_length}). "
                    f"Setting max_prompt_length to max_length // 2 = {max_prompt_length}"
                )

            # Create DPO config with proper parameters (following DPO pattern)
            loss_type = self._get_config_value(
                self.config.train, 'loss_type', default='sigmoid'
            )
            # Recent TRL versions expect a list of loss names. Accept a
            # scalar user value and normalize it at the backend boundary.
            if isinstance(loss_type, str):
                loss_type = [loss_type]

            eval_strategy = self._get_config_value(self.config.train, 'eval_strategy', default='no')
            save_strategy = self._get_config_value(self.config.train, 'save_strategy', default='steps')
            save_steps = self._get_config_value(
                self.config.train, 'save_steps', default=max(1, self.dpo_steps_per_round // 2)
            )
            precision = self._get_config_value(self.config.model, 'precision', default='auto')
            precision = getattr(precision, 'value', precision)
            bf16 = self._get_train_value('bf16', default=str(precision).lower() == 'bf16')
            fp16 = self._get_train_value('fp16', default=str(precision).lower() == 'fp16')

            dpo_config = DPOConfig(
                output_dir=self._get_config_value(self.config.logging, 'output_dir', default='./output'),
                run_name=self._get_config_value(self.config.logging, 'run_name', default='unsloth_spin'),
                per_device_train_batch_size=self._get_config_value(self.config.train, 'per_device_batch_size', default=4),
                per_device_eval_batch_size=self._get_config_value(self.config.train, 'per_device_eval_batch_size', default=4),
                num_train_epochs=num_epochs if max_steps == -1 else 1,
                max_steps=self.dpo_steps_per_round,  # SPIN uses fixed steps per round
                learning_rate=optim_scheduler['learning_rate'],
                lr_scheduler_type=optim_scheduler['lr_scheduler_type'],
                warmup_steps=optim_scheduler['warmup_steps'],
                warmup_ratio=optim_scheduler['warmup_ratio'],
                optim=optim_scheduler['optimizer_type'],
                logging_steps=self._get_config_value(self.config.train, 'logging_steps', default=10),
                logging_strategy=self._get_config_value(self.config.train, 'logging_strategy', default='steps'),
                eval_strategy=eval_strategy,
                eval_steps=self._get_config_value(self.config.train, 'eval_steps', 'eval_interval', default=None),
                save_steps=save_steps,
                save_strategy=save_strategy,
                save_total_limit=self._get_config_value(self.config.train, 'save_total_limit', default=None),
                bf16=bf16,
                fp16=fp16,
                seed=self._get_train_value('seed', 'data_seed', default=42),
                gradient_accumulation_steps=self._get_config_value(self.config.train, 'gradient_accumulation_steps', default=1),
                max_grad_norm=self._get_config_value(self.config.train, 'max_grad_norm', default=1.0),
                gradient_checkpointing=self._get_config_value(self.config.train, 'gradient_checkpointing', default=True),
                # DPO-specific parameters
                beta=self._get_config_value(self.config.train, 'beta', default=0.1),
                # trl's DPOConfig.loss_type is a list[str] (supports combining
                # multiple weighted loss terms) - wrap a plain string config value.
                loss_type=(lambda lt: lt if isinstance(lt, list) else [lt])(
                    self._get_config_value(self.config.train, 'loss_type', default='sigmoid')
                ),
                label_smoothing=self._get_config_value(self.config.train, 'label_smoothing', default=0.0),
                # NOTE: max_prompt_length was removed from trl's DPOConfig in
                # the installed trl version (1.7.1) - only max_length remains.
                # Passing it raises "TypeError: DPOConfig.__init__() got an
                # unexpected keyword argument 'max_prompt_length'". Still
                # computed above (kept for logging) but not forwarded here.
                max_length=max_length,
                remove_unused_columns=False,
                report_to=self._get_config_value(self.config.train, 'report_to', default='none'),
            )

            # Keep Unsloth in sync with the TRL SPIN backend: route supported
            # user-provided extras into the native DPOConfig after construction.
            missing = extract_extra_and_missing_params(
                backend_config=dpo_config, config=self.config, algorithm='spin'
            )
            for key, value in missing.items():
                setattr(dpo_config, key, value)

            # The backfill above may re-flatten loss_type back to a bare string
            # (DPOConfig's own default_factory also yields ['sigmoid'], so the
            # "already explicitly set" heuristic misses it) - re-wrap it.
            if dpo_config.loss_type is None:
                dpo_config.loss_type = ['sigmoid']
            elif isinstance(dpo_config.loss_type, list):
                dpo_config.loss_type = [item for item in dpo_config.loss_type if item is not None]
                if not dpo_config.loss_type:
                    dpo_config.loss_type = ['sigmoid']
            else:
                dpo_config.loss_type = [dpo_config.loss_type]

            logger.info(f"DPO Config: max_length={max_length}, max_prompt_length={max_prompt_length}")

            # DPO trainer will be created in train() after generating preference pairs
            self.dpo_config = dpo_config
            logger.info(
                "SPIN generation config: batch_size=%s, prompt_length=%s, max_new_tokens=%s, "
                "temperature=%s, top_p=%s, top_k=%s, repetition_penalty=%s, do_sample=%s",
                self.generation_batch_size,
                self.generation_prompt_length,
                self.generation_max_length,
                self.generation_temperature,
                self.generation_top_p,
                self.generation_top_k,
                self.generation_repetition_penalty,
                self.generation_do_sample,
            )
            logger.info("DPO trainer configuration prepared for SPIN")

        except Exception as e:
            logger.error(f"Failed to setup DPO trainer: {e}")
            raise

    def generate_responses(
        self,
        prompts: List[str],
        model: torch.nn.Module,
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Generate responses from a model.

        Args:
            prompts: List of prompt strings
            model: Model to generate from
            temperature: Generation temperature

        Returns:
            List of generated responses
        """
        try:
            logger.info(f"Generating {len(prompts)} responses (temperature={temperature})")

            # Tokenize prompts
            tokenization_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": self.generation_prompt_length is not None,
            }
            if self.generation_prompt_length is not None:
                tokenization_kwargs["max_length"] = self.generation_prompt_length
            inputs = self.tokenizer(prompts, **tokenization_kwargs)

            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(model.device)

            generation_kwargs = dict(self.generation_kwargs)
            generation_kwargs.setdefault("max_new_tokens", self.generation_max_length)
            generation_kwargs.setdefault("temperature", temperature)
            generation_kwargs.setdefault("top_p", self.generation_top_p)
            generation_kwargs.setdefault("top_k", self.generation_top_k)
            generation_kwargs.setdefault("repetition_penalty", self.generation_repetition_penalty)
            generation_kwargs.setdefault(
                "do_sample", temperature > 0 if self.generation_do_sample is None else self.generation_do_sample
            )
            generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
            if "max_new_tokens" in generation_kwargs:
                generation_kwargs.pop("max_length", None)
            if generation_kwargs.get("do_sample") is False:
                # These flags are only meaningful for sampling and newer
                # Transformers warn when they are passed to greedy decode.
                for key in ("temperature", "top_p", "top_k"):
                    generation_kwargs.pop(key, None)

            # Generate
            was_training = model.training
            unsloth_inference = False
            try:
                from unsloth import FastLanguageModel

                FastLanguageModel.for_inference(model)
                unsloth_inference = True
            except (ImportError, AttributeError) as exc:
                logger.debug("Unsloth inference patch unavailable; using model.generate: %s", exc)
            model.eval()
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
            finally:
                if unsloth_inference:
                    try:
                        FastLanguageModel.for_training(model)
                    except (AttributeError, RuntimeError) as exc:
                        logger.warning("Could not restore Unsloth training mode: %s", exc)
                if was_training:
                    model.train()

            # Decode only newly generated tokens. String-prefix removal is
            # unreliable when special tokens are normalized during decoding.
            input_width = inputs["input_ids"].shape[1]
            # Keep decoded text on CPU and release temporary generation
            # tensors before the next batch to limit CUDA allocator growth.
            generated_token_ids = outputs[:, input_width:].detach().cpu()
            generated_responses = self.tokenizer.batch_decode(
                generated_token_ids, skip_special_tokens=True
            )
            generated_responses = [response.rstrip() for response in generated_responses]

            del generated_token_ids, outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Generated {len(generated_responses)} responses successfully")
            return generated_responses

        except Exception as e:
            logger.error(f"Failed to generate responses: {e}")
            raise

    def create_preference_pairs(self, dataset=None) -> Dataset:
        """
        Create synthetic preference pairs from SFT dataset.

        Generates from current model and opponent, creates pairs:
        - chosen: SFT reference
        - rejected: opponent output

        Returns:
            Dataset with preference pairs
        """
        try:
            source_dataset = dataset if dataset is not None else self.sft_dataset
            logger.info(f"Creating preference pairs from {len(source_dataset)} SFT examples")

            from datasets import Dataset as HFDataset

            # Opponent is just self.model from previous round (no separate loading needed)
            logger.info("Using self.model as opponent (same model from previous round)")
            opponent = self.model

            pairs = []

            # Extract prompts from SFT dataset
            # Handle different column names
            prompt_col = None
            response_col = None

            sample = source_dataset[0]
            messages_mode = "messages" in sample and isinstance(sample.get("messages"), list)

            if messages_mode:
                prompts = []
                references = []
                for row in source_dataset:
                    turns = row.get("messages") or []
                    assistant_indices = [
                        i for i, turn in enumerate(turns)
                        if isinstance(turn, dict)
                        and turn.get("role") == "assistant"
                        and str(turn.get("content") or "").strip()
                    ]
                    if not assistant_indices:
                        raise ValueError("SFT messages must contain a non-empty assistant response")
                    answer_index = assistant_indices[-1]
                    context = turns[:answer_index]
                    if getattr(self.tokenizer, "chat_template", None):
                        template_kwargs = row.get("chat_template_kwargs") or {}
                        if not isinstance(template_kwargs, dict):
                            template_kwargs = {}
                        else:
                            template_kwargs = dict(template_kwargs)
                        # The trainer configuration is authoritative for SPIN.
                        template_kwargs["enable_thinking"] = bool(self.enable_thinking)
                        try:
                            prompt = self.tokenizer.apply_chat_template(
                                context,
                                tokenize=False,
                                add_generation_prompt=True,
                                **template_kwargs,
                            )
                        except TypeError as exc:
                            # Keep compatibility with templates that do not
                            # expose the optional enable_thinking argument.
                            if "enable_thinking" not in str(exc):
                                raise
                            template_kwargs.pop("enable_thinking", None)
                            prompt = self.tokenizer.apply_chat_template(
                                context,
                                tokenize=False,
                                add_generation_prompt=True,
                                **template_kwargs,
                            )
                    else:
                        prompt = "\n".join(
                            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                            for turn in context
                        )
                    prompts.append(prompt)
                    references.append(str(turns[answer_index].get("content") or "").strip())

            if 'prompt' in sample:
                prompt_col = 'prompt'
            elif 'question' in sample:
                prompt_col = 'question'
            elif 'text' in sample:
                prompt_col = 'text'

            # SFT-normalized rows may retain empty preference columns from
            # CuratorKIT. Prefer the canonical SFT completion and require a
            # non-empty response before selecting a source column.
            if not messages_mode:
                for candidate in ('completion', 'chosen', 'answer', 'response'):
                    if candidate in sample and str(sample.get(candidate) or '').strip():
                        response_col = candidate
                        break

            if not messages_mode and (not prompt_col or not response_col):
                logger.warning(
                    f"Could not find prompt/response columns. Available: {sample.keys()}"
                )
                raise ValueError("SFT dataset must have prompt and response columns")

            # Batch generation
            batch_size = self.generation_batch_size
            if not batch_size or batch_size <= 0:
                raise ValueError("generation_batch_size must be a positive integer")
            if not messages_mode:
                prompts = source_dataset[prompt_col]
                references = source_dataset[response_col]

            from tqdm.auto import tqdm

            for i in tqdm(
                range(0, len(prompts), batch_size),
                desc="SPIN generating responses",
                unit="batch",
            ):
                batch_prompts = prompts[i : i + batch_size]
                batch_refs = references[i : i + batch_size]

                # Generate from current model - UNUSED, wastes compute
                # current_responses = self.generate_responses(
                #     batch_prompts, self.model, temperature=self.generation_temperature
                # )

                # Generate from opponent
                opponent_responses = self.generate_responses(
                    batch_prompts, opponent, temperature=self.generation_temperature
                )

                # Create pairs: (prompt, chosen=SFT_ref, rejected=opponent)
                for prompt, chosen, rejected in zip(
                    batch_prompts, batch_refs, opponent_responses
                ):
                    # Guarantee a word-boundary separator between prompt and
                    # completion. TRL's DPOTrainer tokenizes `prompt` and
                    # `prompt + completion` separately and warns if they don't
                    # share a token prefix - gluing text together with no
                    # space (e.g. "chosen" pulled raw from a dataset column)
                    # is the most common cause of that mismatch.
                    chosen = _ensure_boundary_space(prompt, chosen)
                    rejected = _ensure_boundary_space(prompt, rejected)
                    pairs.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    })

            # No cleanup needed - opponent is self.model, not a separate loaded model

            logger.info(f"Created {len(pairs)} preference pairs")

            # Temporary validation output: inspect pairs in notebook runtimes
            # where logger configuration may hide INFO messages.
            print("=" * 80)
            print(f"SPIN preference-pair preview (first {min(3, len(pairs))})")
            for pair_index, pair in enumerate(pairs[:3], start=1):
                print(f"Pair {pair_index} | PROMPT:\n{pair['prompt'][:1000]}")
                print(f"Pair {pair_index} | CHOSEN / reference:\n{pair['chosen'][:1000]}")
                print(f"Pair {pair_index} | REJECTED / generated:\n{pair['rejected'][:1000]}")
            print("=" * 80)

            # Convert to HF Dataset
            preference_dataset = HFDataset.from_dict({
                "prompt": [p["prompt"] for p in pairs],
                "chosen": [p["chosen"] for p in pairs],
                "rejected": [p["rejected"] for p in pairs],
            })

            return preference_dataset

        except Exception as e:
            logger.error(f"Failed to create preference pairs: {e}")
            raise

    def update_opponent_checkpoint(self) -> None:
        """Opponent checkpoint update (no-op since opponent is self.model)."""
        try:
            # NOTE: Opponent is now self.model directly, so no actual checkpoint update needed
            # The trained model automatically becomes the opponent for the next round
            logger.info(f"Round {self.current_round + 1} complete - trained model becomes opponent for next round")

        except Exception as e:
            logger.error(f"Failed in opponent update: {e}")
            raise

    def train(self) -> Dict[str, Any]:
        """
        Main training loop for SPIN.

        Performs self-play iterations:
        - Generate synthetic preference pairs
        - Train DPO on pairs
        - Update opponent checkpoint
        - Repeat for num_rounds
        """
        try:
            from trl import DPOTrainer

            # NOTE: train() previously jumped straight into the round loop
            # without ever calling setup_model()/setup_data(), so self.model,
            # self.tokenizer and self.sft_dataset were all still None. The
            # first thing create_preference_pairs() does is
            # `len(self.sft_dataset)`, which raised "TypeError: object of
            # type 'NoneType' has no len()". Every other trainer in this repo
            # (GRPO, PPO, ...) explicitly calls its own setup_* methods at the
            # top of train() - SPIN needs to do the same.
            self.setup_model()
            self.setup_data()
            self.setup_trainer()  # builds self.dpo_config, used below each round

            eval_strategy = getattr(self.dpo_config, "eval_strategy", "no")
            eval_strategy = str(getattr(eval_strategy, "value", eval_strategy)).lower()
            eval_enabled = eval_strategy not in {"no", "none", "off", "disabled"}
            if not eval_enabled:
                logger.info("SPIN evaluation disabled; validation responses will not be generated")

            logger.info("=" * 80)
            logger.info("Starting SPIN Self-Play Fine-Tuning")
            logger.info(f"Number of rounds: {self.num_rounds}")
            logger.info("=" * 80)

            all_metrics = {}

            from tqdm.auto import tqdm

            for round_idx in tqdm(
                range(self.num_rounds),
                desc="SPIN rounds",
                unit="round",
            ):
                self.current_round = round_idx
                logger.info(f"\n{'='*80}")
                logger.info(f"SPIN Round {round_idx + 1}/{self.num_rounds}")
                logger.info(f"{'='*80}")

                # Step 1: Generate pairs from this round's training slice.
                logger.info("Step 1: Generating synthetic preference pairs...")
                source = self.train_pool if self.train_pool is not None else self.sft_dataset
                if self.samples_per_round is not None:
                    start = round_idx * self.samples_per_round
                    end = start + self.samples_per_round
                    if end > len(source):
                        raise ValueError(
                            f"SPIN needs {end} training rows by round {round_idx + 1}, "
                            f"but only {len(source)} are available"
                        )
                    source = source.select(range(start, end))
                preference_dataset = self.create_preference_pairs(source)

                eval_preference_dataset = None
                if eval_enabled and self.eval_dataset is not None and len(self.eval_dataset):
                    logger.info("Generating validation preference pairs from fixed validation rows...")
                    eval_preference_dataset = self.create_preference_pairs(self.eval_dataset)

                # Step 2: Train DPO
                logger.info(f"Step 2: Training DPO on {len(preference_dataset)} pairs...")

                # Create DPO trainer
                dpo_trainer = DPOTrainer(
                    model=self.model,
                    ref_model=self.reference_model,
                    args=self.dpo_config,
                    train_dataset=preference_dataset,
                    eval_dataset=eval_preference_dataset,
                    processing_class=self.tokenizer,  # DPO uses processing_class, not tokenizer
                    peft_config=None,  # PEFT already applied to model
                )
                self.trainer = dpo_trainer

                # Train for specified steps
                train_result = dpo_trainer.train()
                logger.info(f"DPO training completed for round {round_idx + 1}")

                # Store metrics
                round_key = f"round_{round_idx + 1}"
                all_metrics[round_key] = train_result.metrics if hasattr(train_result, 'metrics') else {}

                # Step 3: Update opponent checkpoint
                logger.info("Step 3: Updating opponent checkpoint...")
                self.update_opponent_checkpoint()

                # Save round checkpoint
                round_checkpoint = os.path.join(
                    self._get_config_value(self.config.logging, 'output_dir', default='./output'),
                    f"spin_round_{round_idx + 1}"
                )
                os.makedirs(round_checkpoint, exist_ok=True)
                self.model.save_pretrained(round_checkpoint)
                self.tokenizer.save_pretrained(round_checkpoint)
                logger.info(f"Round checkpoint saved to {round_checkpoint}")

            logger.info("\n" + "=" * 80)
            logger.info("SPIN training completed successfully!")
            logger.info("=" * 80)

            return {"metrics": all_metrics}

        except Exception as e:
            logger.error(f"SPIN training failed: {e}")
            raise
        finally:
            # Cleanup opponent checkpoint directory
            if self.opponent_checkpoint_dir and os.path.exists(self.opponent_checkpoint_dir):
                try:
                    shutil.rmtree(self.opponent_checkpoint_dir)
                    logger.info("Opponent checkpoint directory cleaned up")
                except Exception as e:
                    logger.warning(f"Could not cleanup opponent checkpoint: {e}")

    def save_model(self, output_dir: str) -> None:
        """Save trained model."""
        try:
            logger.info(f"Saving model to {output_dir}")
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
