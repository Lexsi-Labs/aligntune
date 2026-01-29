"""
Unsloth Counterfactual GRPO Backend Implementation

This module combines Unsloth's FastLanguageModel for efficient model loading
with TRL's CounterfactualGRPOTrainer for custom token importance weighting.

Key benefits:
- Unsloth's 2-5x speed improvement and memory efficiency
- Faithful counterfactual importance weighting (same logic as TRL backend)
- Compatible with all counterfactual parameters (boost_factor, min_weight, etc.)
"""

import logging
import time
import re
import math
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader

from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.rl.registries import DatasetRegistry, RewardRegistry
from aligntune.core.rl.caching import DatasetCache
from aligntune.core.sft.evaluator import EnhancedEvaluator
from aligntune.data.manager import DataManager
from aligntune.utils.math_grading import extract_math_gold, extract_math_answer, grade_math_answer

logger = logging.getLogger(__name__)


def apply_chat_template_safe(
        tokenizer,
        messages: list,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False) -> str:
    """Apply chat template with optional enable_thinking support."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking)


class UnslothCounterFactGRPOTrainer(TrainerBase):
    """
    Counterfactual GRPO trainer using Unsloth's FastLanguageModel + TRL's CounterfactualGRPOTrainer.

    This ensures:
    1. Fast model loading via Unsloth (4-bit quantization, optimized inference)
    2. Faithful counterfactual importance weighting (identical to TRL backend)
    3. All counterfactual parameters supported (boost_factor, min_weight, max_spans, etc.)
    """

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.training_history = []
        self.logging_manager = None
        self.evaluator = None
        self.unsloth_model = None
        self.reward_functions = []
        self.train_dataset = None
        self.eval_dataset = None
        self.reward_configs = []
        self.dataset_dict = None
        self.prompt_to_answer = {}

    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth and TRL are available."""
        try:
            import unsloth
            from unsloth import FastLanguageModel
            from trl import GRPOConfig
            # Check CounterfactualGRPOTrainer is available
            from aligntune.backends.trl.rl.counterfact_grpo.custom_trainer import CounterfactualGRPOTrainer
            return True
        except ImportError:
            return False

    def _get_config_value(self, config_obj, *attr_names, default=None):
        """Safely get config value from multiple possible attribute names."""
        if isinstance(config_obj, dict):
            for attr_name in attr_names:
                if attr_name in config_obj:
                    return config_obj[attr_name]
        else:
            for attr_name in attr_names:
                if hasattr(config_obj, attr_name):
                    return getattr(config_obj, attr_name)
        return default

    def setup_model(self) -> None:
        """Setup Unsloth-optimized model and tokenizer."""
        try:
            import unsloth
            from unsloth import FastLanguageModel

            # Handle both dict and object config
            if isinstance(self.config.model, dict):
                model_name = self.config.model.get('name_or_path')
                max_seq_length = self.config.model.get('max_seq_length', 2048)
                quantization = self.config.model.get('quantization', {})
                lora_r = self.config.model.get('lora_r', 32)
                lora_alpha = self.config.model.get('lora_alpha', 32)
                lora_dropout = self.config.model.get('lora_dropout', 0.1)
                lora_target_modules = self.config.model.get(
                    'lora_target_modules', [
                        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            else:
                model_name = self.config.model.name_or_path
                max_seq_length = self.config.model.max_seq_length
                quantization = getattr(self.config.model, 'quantization', {})
                lora_r = getattr(self.config.model, 'lora_r', 32)
                lora_alpha = getattr(self.config.model, 'lora_alpha', 32)
                lora_dropout = getattr(self.config.model, 'lora_dropout', 0.1)
                lora_target_modules = getattr(
                    self.config.model, 'lora_target_modules', [
                        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

            logger.info("=" * 80)
            logger.info(
                f"Setting up Unsloth Counterfactual GRPO model: {model_name}")
            logger.info("=" * 80)

            # Check if fast_inference (vLLM) is enabled
            fast_inference = self._get_config_value(
                self.config.train, 'fast_inference', default=False)
            vllm_gpu_memory_utilization = self._get_config_value(
                self.config.train, 'vllm_gpu_memory_utilization', default=0.7)

            precision = self._get_config_value(
                self.config.model, 'precision', default='fp32')

            # Configure Unsloth model parameters
            model_kwargs = {
                "max_seq_length": max_seq_length,
                "dtype": None,  # Auto-detect
                "load_in_4bit": quantization.get("load_in_4bit", True) if isinstance(quantization, dict) else True,
                "fast_inference": fast_inference,  # Unsloth's native vLLM integration
                # vLLM memory (0.95 for max speed)
                "gpu_memory_utilization": vllm_gpu_memory_utilization,
            }

            logger.info(f"Loading model with kwargs: {model_kwargs}")

            # Load model with Unsloth optimizations
            self.unsloth_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name, **model_kwargs)

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Configure model for training with LoRA
            logger.info(
                f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            logger.info(f"Target modules: {lora_target_modules}")

            self.unsloth_model = FastLanguageModel.get_peft_model(
                self.unsloth_model,
                r=lora_r,
                target_modules=lora_target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            logger.info(
                "Unsloth Counterfactual GRPO model setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup Unsloth model: {e}")
            raise

    def setup_rewards(self) -> None:
        """Setup reward functions for Counterfactual GRPO training."""
        try:
            logger.info("Setting up reward functions for Counterfactual GRPO")

            rewards_config = self.config.rewards if hasattr(
                self.config, 'rewards') else []

            if rewards_config:
                reward_configs = []
                for r in rewards_config:
                    if isinstance(r, dict):
                        reward_configs.append(r)
                    elif hasattr(r, 'to_dict'):
                        reward_configs.append(r.to_dict())
                    else:
                        reward_configs.append({
                            'type': getattr(r, 'type', 'length'),
                            'weight': getattr(r, 'weight', 1.0),
                            'params': getattr(r, 'params', {})
                        })
            else:
                reward_configs = [{"type": "length", "weight": 0.2, "params": {
                    "min_length": 10, "max_length": 200}}, ]

            # Load actual reward functions from registry
            from aligntune.rewards.registry import RewardRegistry as RewardsRegistry
            from aligntune.rewards.core import RewardConfig, RewardType

            for reward_config in reward_configs:
                try:
                    reward_type_name = reward_config['type']
                    params = reward_config.get('params', {})
                    weight = reward_config.get('weight', 1.0)

                    reward_type_mapping = {
                        'math': 'math_reasoning',
                        'code': 'code_quality',
                    }
                    reward_type = reward_type_mapping.get(
                        reward_type_name, reward_type_name)

                    reward_config_dict = {
                        "type": reward_type,
                        "params": params,
                        "weight": 1.0
                    }

                    try:
                        reward_func_obj = RewardsRegistry.get_reward_function(
                            reward_type, reward_config_dict)
                    except ValueError as ve:
                        logger.warning(
                            f"Reward type '{reward_type}' not found in registry: {ve}")
                        continue

                    if hasattr(reward_func_obj, 'compute'):
                        reward_func = reward_func_obj.compute
                    elif callable(reward_func_obj):
                        reward_func = reward_func_obj
                    else:
                        logger.warning(
                            f"Reward function '{reward_type}' is not callable, skipping")
                        continue

                    logger.info(
                        f"Loaded {reward_type} reward from registry (weight: {weight})")

                    self.reward_functions.append({
                        "function": reward_func,
                        "weight": weight,
                        "name": reward_type_name
                    })
                except Exception as e:
                    logger.warning(
                        f"Failed to load reward function '{
                            reward_config.get(
                                'type', 'unknown')}': {e}")
                    continue

            logger.info(
                f"Configured {len(self.reward_functions)} reward functions")
            self.reward_configs = reward_configs

        except Exception as e:
            logger.error(f"Failed to setup reward functions: {e}")
            raise

    def setup_data(self) -> None:
        """Setup datasets for Counterfactual GRPO training."""
        self.setup_dataset()
    
    def parse_gsm8k_gold(self,ans: str) -> str:
            """Parse gold answer from GSM8K format (#### answer)."""
            import re
            m = re.search(r"####\s*(.+)$", ans, flags=re.MULTILINE)
            if not m:
                nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
                return nums[-1] if nums else ""
            s = m.group(1).strip().replace(",", "")
            if re.fullmatch(r"-?\d+/\d+", s):
                num, den = s.split("/")
                try:
                    return str(float(num) / float(den))
                except:
                    return s
            return s

    def setup_dataset(self) -> None:
        """Setup datasets for GRPO training using unified DataManager."""
        logger.info("Setting up GRPO datasets with DataManager...")
        
        # Extract dataset configuration
        dataset_config = None
        if hasattr(self.config, 'dataset'):
            dataset_config = self.config.dataset
        elif hasattr(self.config, 'datasets') and len(self.config.datasets) > 0:
            dataset_config = self.config.datasets[0]
        else:
            raise ValueError("No dataset configuration found")
        
        # Extract parameters
        dataset_name = self._get_config_value(dataset_config, 'name', 'dataset_name', default='imdb')
        split = self._get_config_value(dataset_config, 'split', default=None)
        config_name = self._get_config_value(dataset_config, 'config_name', default=None)
        system_prompt = self._get_config_value(dataset_config, 'system_prompt', default=None)
        enable_thinking = self._get_config_value(self.config.train, 'enable_thinking', default=False)
        
        # Check if this is a math dataset
        is_math_dataset = any(math_name in dataset_name.lower() for math_name in ['gsm8k', 'math', 'metamath'])
        
        if is_math_dataset:
            logger.info(f"Detected math dataset: {dataset_name}. Using custom format_for_grpo logic.")
            
            # Load raw dataset
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, config_name, split=split)
            
            # Apply max_samples if specified
            max_samples = self._get_config_value(dataset_config, 'max_samples', default=None)
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                logger.info(f"Limited dataset to {max_samples} samples")
            
            # Apply the format_for_grpo function
            def format_for_grpo(example):
                """Format example for GRPO training."""
                # Try to extract prompt from various column formats
                prompt_text = None
                if 'chosen' in example:
                    text = example['chosen']
                    if '\n\nAssistant:' in text:
                        prompt_text = text.split('\n\nAssistant:')[0] + '\n\nAssistant:'
                    else:
                        prompt_text = text[:len(text)//2]
                elif 'prompt' in example:
                    prompt_text = example['prompt']
                elif 'query' in example:
                    prompt_text = example['query']
                elif 'question' in example:
                    prompt_text = example['question']
                elif 'problem' in example:
                    prompt_text = example['problem']
                elif 'instruction' in example:
                    prompt_text = example['instruction']
                elif 'text' in example:
                    prompt_text = example['text']
                else:
                    first_col = dataset.column_names[0]
                    prompt_text = str(example[first_col])
                
                if not prompt_text or len(prompt_text.strip()) == 0:
                    prompt_text = "Please provide a helpful response."
                
                # For math datasets, use simple prompt format without system prompt
                if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                    user_content = f"Solve this math problem step by step. Show your work and put your final numeric answer at the end.\n\nProblem:\n{prompt_text}"
                    messages = [{"role": "user", "content": user_content}]
                    prompt_text = apply_chat_template_safe(
                        self.tokenizer, 
                        messages, 
                        enable_thinking=enable_thinking
                    )
                else:
                    prompt_text = f"Solve this math problem step by step. Show your work and put your final numeric answer at the end.\n\nProblem:\n{prompt_text}\n\nAnswer:"
                
                # Extract gold answer - CRITICAL for reward computation!
                answer_clean = ""
                if 'answer' in example:
                    answer_clean = self.parse_gsm8k_gold(str(example['answer']))
                elif 'answer_clean' in example:
                    answer_clean = str(example['answer_clean'])
                elif 'solution' in example:
                    answer_clean = extract_math_gold(str(example['solution']))
                
                return {
                    "query": prompt_text,
                    "prompt": prompt_text,
                    "answer_clean": answer_clean,
                }
            
            # Apply formatting
            formatted_dataset = dataset.map(
                format_for_grpo,
                remove_columns=[col for col in dataset.column_names if col not in ['query', 'prompt', 'answer_clean']],
                desc="Formatting math dataset for GRPO"
            )
            
            # Filter empty prompts
            formatted_dataset = formatted_dataset.filter(
                lambda x: len(x['query'].strip()) > 0,
                desc="Filtering empty prompts"
            )
            
            # Shuffle dataset
            data_seed = self._get_config_value(self.config.train, 'data_seed', default=47)
            formatted_dataset = formatted_dataset.shuffle(seed=data_seed)
            logger.info(f"Dataset shuffled with data_seed={data_seed}")
            
            # Split into train/eval (e.g., 90/10 split)
            split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=data_seed)
            self.train_dataset = split_dataset['train']
            self.eval_dataset = split_dataset['test']
            
            # Build prompt→answer mapping
            self.prompt_to_answer = {}
            for item in formatted_dataset:
                self.prompt_to_answer[item["prompt"]] = item.get("answer_clean", "0")
            logger.info(f"Built prompt→answer mapping with {len(self.prompt_to_answer)} entries")
            
            logger.info(f"Math dataset loaded: {len(self.train_dataset)} train examples")
            logger.info(f"Math dataset eval: {len(self.eval_dataset)} eval examples")
            logger.info(f"Dataset columns: {self.train_dataset.column_names}")
            
            # Log sample
            if len(self.train_dataset) > 0:
                sample = self.train_dataset[0]
                logger.info(f"Sample prompt (first 100 chars): {sample['query'][:100]}...")
                logger.info(f"Sample answer_clean: {sample.get('answer_clean', 'N/A')}")
        
        else:
            # Normal dataset - use DataManager
            logger.info(f"Loading regular dataset: {dataset_name}")
            
            # Advanced DataManager features
            column_mapping = self._get_config_value(dataset_config, 'column_mapping', default=None)
            processing_fn = self._get_config_value(dataset_config, 'processing_fn', default=None)
            processing_batched = self._get_config_value(dataset_config, 'processing_batched', default=False)
            max_samples = self._get_config_value(dataset_config, 'max_samples', default=None)
            
            # Initialize DataManager for GRPO task
            from aligntune.data.manager import DataManager
            
            manager = DataManager(
                task_type="grpo",
                system_prompt=system_prompt,
                tokenizer=self.tokenizer,
                enable_thinking=enable_thinking,
                column_mapping=column_mapping,
                processing_fn=processing_fn,
                max_samples = max_samples, 
                processing_batched=processing_batched
            )
            
            # Load dataset
            dataset_dict = manager.load_dataset(
                dataset_name,
                config_name=config_name,
                split=split,
            )
            
            # Extract train and validation splits
            self.train_dataset = dataset_dict["train"]
            self.eval_dataset = dataset_dict.get("validation", None)
            self.dataset_dict = dataset_dict
            
            logger.info(f"Dataset loaded: {len(self.train_dataset)} train examples")
            if self.eval_dataset:
                logger.info(f"Evaluation dataset: {len(self.eval_dataset)} examples")
            
            # Log sample
            if len(self.train_dataset) > 0:
                sample = self.train_dataset[0]
                prompt_col = "prompt" if "prompt" in sample else "query"
                logger.info(f"Sample prompt (first 100 chars): {sample[prompt_col][:100]}...")
                logger.info(f"Dataset columns: {self.train_dataset.column_names}")          


    def _combined_reward_function(
            self,
            completions: List[str],
            prompts: List[str] = None,
            **kwargs) -> List[float]:
        """Outcome-based reward function matching training_script.py."""
        def parse_pred_number(text: str) -> str:
            nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            candidate = nums[-1] if nums else ""
            if re.fullmatch(r"-?\d+/\d+", candidate):
                num, den = candidate.split("/")
                try:
                    return str(float(num) / float(den))
                except BaseException:
                    return candidate
            return candidate

        def numeric_equal(a: str, b: str, rtol=1e-4, atol=1e-8) -> bool:
            try:
                fa = float(a)
                fb = float(b)
                return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
            except BaseException:
                return a.strip() == b.strip()

        if not completions:
            return []

        # Get gold answers from prompt→answer mapping
        if prompts:
            answers = [self.prompt_to_answer.get(p, "0") for p in prompts]
        else:
            answers = list(self.prompt_to_answer.values())

        # Expand answers for multiple completions per prompt
        factor = max(1, len(completions) // len(answers)) if answers else 1
        expanded = [a for a in answers for _ in range(
            factor)][:len(completions)]

        rewards = []
        correct_count = 0

        for comp, gold in zip(completions, expanded):
            if isinstance(comp, str):
                text = comp
            elif isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict):
                text = comp[0].get("content", str(comp))
            else:
                text = str(comp)

            pred = parse_pred_number(text)
            is_correct = numeric_equal(pred, gold)
            reward = 1.0 if is_correct else 0.0
            rewards.append(reward)

            if is_correct:
                correct_count += 1

        # Log accuracy
        if not hasattr(self, '_reward_step'):
            self._reward_step = 0
        self._reward_step += 1

        accuracy = correct_count / len(rewards) if rewards else 0.0
        print(
            f"\n[Unsloth CounterFact GRPO] Reward (step {
                self._reward_step}): {
                accuracy:.2%} ({correct_count}/{
                len(rewards)})")
        import sys
        sys.stdout.flush()

        return rewards

    def setup_trainer(self) -> None:
        """Setup CounterfactualGRPOTrainer with Unsloth model."""
        try:
            from trl import GRPOConfig
            # Import the SAME CounterfactualGRPOTrainer from TRL backend
            # (faithful implementation!)
            from aligntune.backends.trl.rl.counterfact_grpo.custom_trainer import CounterfactualGRPOTrainer

            logger.info(
                "Setting up CounterfactualGRPOTrainer with Unsloth model")

            # Get output directory
            if isinstance(self.config.logging, dict):
                output_dir = self.config.logging.get(
                    'output_dir', './output/unsloth_counterfact_grpo')
                run_name = self.config.logging.get(
                    'run_name', 'unsloth_counterfact_grpo')
            else:
                output_dir = getattr(
                    self.config.logging,
                    'output_dir',
                    './output/unsloth_counterfact_grpo')
                run_name = getattr(
                    self.config.logging,
                    'run_name',
                    'unsloth_counterfact_grpo')

            # Get training parameters
            num_epochs = self._get_config_value(
                self.config.train, 'epochs', 'num_epochs', default=1)
            # Ensure num_epochs is not None (TRL requires a number)
            if num_epochs is None:
                num_epochs = 1
            per_device_batch_size = self._get_config_value(
                self.config.train, 'per_device_batch_size', default=4)
            gradient_accumulation_steps = self._get_config_value(
                self.config.train, 'gradient_accumulation_steps', default=1)
            learning_rate = self._get_config_value(
                self.config.train, 'learning_rate', default=5e-5)
            max_steps = self._get_config_value(
                self.config.train, 'max_steps', default=-1)
            warmup_steps = self._get_config_value(
                self.config.train, 'warmup_steps', default=10)
            warmup_ratio = self._get_config_value(
                self.config.train, 'warmup_ratio', default=0.0)

            # Checkpointing
            save_steps = self._get_config_value(
                self.config.train, 'save_steps', default=500)
            save_total_limit = self._get_config_value(
                self.config.train, 'save_total_limit', default=None)
            logging_steps = self._get_config_value(
                self.config.train, 'logging_steps', default=10)

            # Sequence lengths
            max_prompt_length = self._get_config_value(
                self.config.train, 'max_prompt_length', default=256)
            max_completion_length = self._get_config_value(
                self.config.train, 'max_completion_length', default=768)

            # Generation parameters
            num_generations = self._get_config_value(
                self.config.train, 'num_generations', default=per_device_batch_size)
            temperature = self._get_config_value(
                self.config.train, 'temperature', default=0.6)
            top_p = self._get_config_value(
                self.config.train, 'top_p', default=0.95)

            # GRPO specific parameters
            beta = self._get_config_value(
                self.config.train, 'beta', 'kl_coef', default=0.005)
            epsilon = self._get_config_value(
                self.config.train, 'epsilon', 'cliprange', default=0.2)
            loss_type = self._get_config_value(
                self.config.train, 'loss_type', default='dapo')
            if loss_type == "sigmoid":
                loss_type = "dapo"
            scale_rewards = self._get_config_value(
                self.config.train, 'scale_rewards', default='group')
            mask_truncated_completions = self._get_config_value(
                self.config.train, 'mask_truncated_completions', default=True)
            max_grad_norm = self._get_config_value(
                self.config.train, 'max_grad_norm', default=0.0)

            # Counterfactual specific parameters (CRITICAL!)
            boost_factor = self._get_config_value(
                self.config.train, 'boost_factor', default=2.0)
            min_weight = self._get_config_value(
                self.config.train, 'min_weight', default=0.5)
            max_spans = self._get_config_value(
                self.config.train, 'max_spans', default=10)
            answer_weight = self._get_config_value(
                self.config.train, 'answer_weight', default=1.5)
            enable_gradient_conservation = self._get_config_value(
                self.config.train, 'enable_gradient_conservation', default=False)
            weight_debug = self._get_config_value(
                self.config.train, 'weight_debug', default=False)

            # Extra verbose mode for paper examples
            extra_verbose = self._get_config_value(
                self.config.train, 'extra_verbose', default=False)
            extra_verbose_log_path = self._get_config_value(
                self.config.train, 'extra_verbose_log_path', default=None)
            extra_verbose_sample_rate = self._get_config_value(
                self.config.train, 'extra_verbose_sample_rate', default=0.1)

            # Weighting mode
            weighting_mode = self._get_config_value(
                self.config.train, 'weighting_mode', default=None)
            method_name = self._get_config_value(
                self.config.train, 'method_name', default='counterfactual')

            # Convert legacy to weighting_mode if not explicitly set
            if weighting_mode is None:
                if method_name == "baseline":
                    weighting_mode = "vanilla"
                else:
                    weighting_mode = "counterfactual"

            # Gradient checkpointing
            use_gradient_checkpointing = self._get_config_value(
                self.config.train, 'use_gradient_checkpointing', default=True)

            # Seeds
            seed = self._get_config_value(
                self.config.train, 'seed', default=42)
            data_seed = self._get_config_value(
                self.config.train, 'data_seed', default=47)

            # vLLM for fast generation
            # NOTE: TRL's use_vllm doesn't work well with LoRA/PEFT models
            # Unsloth has native vLLM integration via fast_inference=True in
            # model loading
            use_vllm = self._get_config_value(
                self.config.train, 'use_vllm', default=False)
            vllm_gpu_memory_utilization = self._get_config_value(
                self.config.train, 'vllm_gpu_memory_utilization', default=0.7)

            # Get quantization config for display
            if isinstance(self.config.model, dict):
                quantization = self.config.model.get('quantization', {})
            else:
                quantization = getattr(self.config.model, 'quantization', {})
            load_in_4bit = quantization.get(
                "load_in_4bit", True) if isinstance(
                quantization, dict) else True
            precision = self._get_config_value(
                self.config.model, 'precision', default='fp32')
            # Handle enum vs string - get string value if it's an enum
            if hasattr(precision, 'value'):
                precision = precision.value
            # Ensure 'auto' defaults to 'bf16' for better memory efficiency
            if precision == 'auto':
                precision = 'bf16'
            # Create GRPO configuration
            grpo_config = GRPOConfig(
                output_dir=output_dir,
                run_name=run_name,
                num_train_epochs=num_epochs,
                max_steps=max_steps,
                per_device_train_batch_size=per_device_batch_size,
                num_generations=num_generations,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_completion_length=max_completion_length,
                max_prompt_length=max_prompt_length,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                warmup_ratio=warmup_ratio,
                max_grad_norm=max_grad_norm,
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_strategy="steps",
                save_total_limit=save_total_limit,
                remove_unused_columns=False,
                fp16=precision == "fp16",
                bf16=precision == "bf16",
                gradient_checkpointing=use_gradient_checkpointing,
                # GRPO specific
                loss_type=loss_type,
                beta=beta,
                epsilon=epsilon,
                scale_rewards=scale_rewards,
                temperature=temperature,
                top_p=top_p,
                mask_truncated_completions=mask_truncated_completions,
                # Seeds
                seed=seed,
                data_seed=data_seed,
                # vLLM for fast generation
                use_vllm=use_vllm,
                vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                # Reporting
                report_to=self.config.logging.loggers if hasattr(
                    self.config.logging, 'loggers') and self.config.logging.loggers else [],
            )

            # Reward wrapper
            def reward_wrapper(completions, prompts=None, **kwargs):
                """Wrapper to pass prompts to reward function for gold answer lookup."""
                if completions is None:
                    return [0.0] * (len(prompts) if prompts else 1)
                return self._combined_reward_function(
                    completions, prompts=prompts, **kwargs)

            # Patch GRPOTrainer.compute_loss to use our counterfactual weighting
            # Unsloth returns hidden states, but we compute log probs from them
            from trl import GRPOTrainer
            GRPOTrainer.compute_loss = CounterfactualGRPOTrainer.compute_loss
            GRPOTrainer._compute_loss = CounterfactualGRPOTrainer._compute_loss_impl
            print(
                "\n[INFO] Patched GRPOTrainer for counterfactual weighting with Unsloth\n")
            import sys
            sys.stdout.flush()

            # Create CounterfactualGRPOTrainer with Unsloth model
            # This is the KEY difference - we use the TRL CounterfactualGRPOTrainer
            # but pass in the Unsloth-optimized model!
            self.trainer = CounterfactualGRPOTrainer(
                model=self.unsloth_model,  # Unsloth's optimized model!
                args=grpo_config,
                train_dataset=self.train_dataset,
                processing_class=self.tokenizer,
                reward_funcs=[reward_wrapper],
                # Counterfactual params (passed to CounterfactualGRPOTrainer)
                boost_factor=boost_factor,
                min_weight=min_weight,
                max_spans=max_spans,
                weight_debug=weight_debug,
                answer_weight=answer_weight,
                weighting_mode=weighting_mode,
                enable_gradient_conservation=enable_gradient_conservation,
                # Extra verbose for paper examples
                extra_verbose=extra_verbose,
                extra_verbose_log_path=extra_verbose_log_path,
                extra_verbose_sample_rate=extra_verbose_sample_rate,
            )

            logger.info(
                "CounterfactualGRPOTrainer with Unsloth model setup completed")

        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step (not used - TRL handles internally)."""
        logger.debug(
            "train_step() called but Unsloth CounterFact GRPO uses TRL's internal training loop")
        return {"loss": 0.0}

    def create_data_loader(self) -> Optional[DataLoader]:
        """Create data loader (not used - TRL handles internally)."""
        logger.debug(
            "create_data_loader() called but Unsloth CounterFact GRPO uses TRL's internal data loading")
        return None

    def train(self) -> Dict[str, Any]:
        """Execute Counterfactual GRPO training with Unsloth optimizations."""
        try:
            logger.info("=" * 80)
            logger.info("Starting Unsloth Counterfactual GRPO training")
            logger.info("=" * 80)
            start_time = time.time()

            # Setup components
            self.setup_model()
            self.setup_rewards()
            self.setup_data()
            self.setup_trainer()

            # Start training
            training_result = self.trainer.train()

            # Get output directory
            if isinstance(self.config.logging, dict):
                output_dir = self.config.logging.get(
                    'output_dir', './output/unsloth_counterfact_grpo')
            else:
                output_dir = getattr(
                    self.config.logging,
                    'output_dir',
                    './output/unsloth_counterfact_grpo')

            # Save model
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            training_time = time.time() - start_time

            # Compile results
            results = {
                "training_time": training_time,
                "final_loss": training_result.training_loss if hasattr(
                    training_result,
                    'training_loss') else 0.0,
                "total_steps": training_result.global_step if hasattr(
                    training_result,
                    'global_step') else 0,
                "model_path": output_dir,
                "training_history": self.training_history,
                "num_reward_functions": len(
                    self.reward_functions),
                "num_datasets": len(
                    self.config.datasets) if hasattr(
                        self.config,
                        'datasets') else 0,
                "backend": "unsloth",
                "trainer": "counterfactual_grpo",
            }

            logger.info("=" * 80)
            logger.info(
                f"Unsloth Counterfactual GRPO training completed in {
                    training_time:.2f} seconds")
            if hasattr(training_result, 'training_loss'):
                logger.info(f"Final loss: {training_result.training_loss:.4f}")
            logger.info(f"Model saved to: {output_dir}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Counterfactual GRPO training failed: {e}")
            raise

    # def evaluate(self) -> Dict[str, Any]:
    #     """Evaluate the trained model."""
    #     try:
    #         if not self.eval_dataset:
    #             logger.warning("No evaluation dataset available")
    #             return {}

    #         logger.info("Evaluating Unsloth Counterfactual GRPO model")

    #         eval_results = self.trainer.evaluate()

    #         logger.info(f"Evaluation results: {eval_results}")

    #         return eval_results

    #     except Exception as e:
    #         logger.error(f"Evaluation failed: {e}")
    #         raise

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model."""
        try:
            if isinstance(self.config.logging, dict):
                default_path = self.config.logging.get(
                    'output_dir', './output/unsloth_counterfact_grpo')
            else:
                default_path = getattr(
                    self.config.logging,
                    'output_dir',
                    './output/unsloth_counterfact_grpo')

            save_path = path or default_path

            logger.info(
                f"Saving Unsloth Counterfactual GRPO model to: {save_path}")

            self.unsloth_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            # Save training configuration
            config_path = Path(save_path) / \
                "counterfact_grpo_training_config.yaml"
            with open(config_path, "w") as f:
                config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
                yaml.dump(config_dict, f, default_flow_style=False)

            logger.info(f"Model saved successfully to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        try:
            logger.info(
                f"Loading Unsloth Counterfactual GRPO model from: {path}")

            import unsloth
            from unsloth import FastLanguageModel

            if isinstance(self.config.model, dict):
                max_seq_length = self.config.model.get('max_seq_length', 2048)
                quantization = self.config.model.get('quantization', {})
            else:
                max_seq_length = getattr(
                    self.config.model, 'max_seq_length', 2048)
                quantization = getattr(self.config.model, 'quantization', {})

            load_in_4bit = quantization.get(
                "load_in_4bit", True) if isinstance(
                quantization, dict) else True

            self.unsloth_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
