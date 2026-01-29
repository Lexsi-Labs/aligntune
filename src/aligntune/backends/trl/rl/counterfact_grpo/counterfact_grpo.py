"""
TRL Counterfactual GRPO Trainer with Custom Importance Weighting

This module provides a TRL backend for Counterfactual Group Relative Policy Optimization,
integrating the custom CounterfactualGRPOTrainer with the same reward structure
as the standard GRPO implementation.
"""

import logging
import time
import yaml
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import torch
from torch.utils.data import DataLoader
import numpy as np

from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.rl.registries import DatasetRegistry, RewardRegistry
from aligntune.core.rl.caching import DatasetCache
from aligntune.core.rl.sample_logger import generate_and_log_samples
from aligntune.core.sft.evaluator import EnhancedEvaluator
from aligntune.utils.math_grading import extract_math_gold, extract_math_answer, grade_math_answer
from aligntune.core.precision_handler import PrecisionHandler

logger = logging.getLogger(__name__)


def apply_chat_template_safe(tokenizer, messages: list, tokenize: bool = False,
                              add_generation_prompt: bool = True, enable_thinking: bool = False) -> str:
    """Apply chat template with optional enable_thinking support.

    Note: enable_thinking is passed via **kwargs in most tokenizers, so we always
    pass it. Tokenizers that don't use it will simply ignore it.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking
    )


# ============================================================================
# COUNTERFACTUAL TRL TRAINER (mirrors TRLGRPOTrainer 1:1)
# ============================================================================
class TRLCounterFactGRPOTrainer(TrainerBase):
    """Counterfactual GRPO trainer using custom CounterfactualGRPOTrainer with enhanced rewards."""
    
    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.reward_functions = []
        self.training_history = []
        self.logging_manager = None
        self.evaluator = None
        self.dataset_dict = None
        
    @classmethod
    def is_available(cls) -> bool:
        """Check if required dependencies are available."""
        try:
            from trl import GRPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # Import the custom CounterfactualGRPOTrainer
            from .custom_trainer import CounterfactualGRPOTrainer
            return True
        except ImportError as e:
            logger.warning(f"Required dependencies not available: {e}")
            return False
    
    def setup_model(self) -> None:
        """Setup model using standard Transformers."""
        # Extract model config values safely
        model_name = self._get_config_value(self.config.model, 'name_or_path', 'model_name', default='gpt2')
        trust_remote_code = self._get_config_value(self.config.model, 'trust_remote_code', default=False)
        precision = self._get_config_value(self.config.model, 'precision', default='fp32')
        # Handle enum vs string - get string value if it's an enum
        if hasattr(precision, 'value'):
            precision = precision.value
        # Ensure 'auto' defaults to 'bf16' for better memory efficiency
        if precision == 'auto':
            precision = 'bf16'
        device_map = self._get_config_value(self.config.model, 'device_map', default='auto')
        load_in_4bit = self._get_config_value(self.config.model, 'load_in_4bit', default=False)
        load_in_8bit = self._get_config_value(self.config.model, 'load_in_8bit', default=False)
        use_peft = self._get_config_value(self.config.model, 'use_peft', default=False)
        
        # Auto-detect quantization from model name if not explicitly set
        if not load_in_4bit and not load_in_8bit:
            model_name_lower = model_name.lower()
            if 'bnb-4bit' in model_name_lower or '4bit' in model_name_lower or 'awq' in model_name_lower:
                logger.info(f"Auto-detected 4-bit quantization from model name: {model_name}")
                load_in_4bit = True
            elif 'bnb-8bit' in model_name_lower or '8bit' in model_name_lower:
                logger.info(f"Auto-detected 8-bit quantization from model name: {model_name}")
                load_in_8bit = True
        
        # Auto-enable PEFT if using quantization (required for training quantized models)
        if (load_in_4bit or load_in_8bit) and not use_peft:
            logger.info("Quantization detected - auto-enabling PEFT/LoRA adapters (required for training)")
            use_peft = True
        
        logger.info("=" * 80)
        logger.info(f"Setting up Counterfactual GRPO model: {model_name}")
        logger.info("=" * 80)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("Transformers not available. Install with: pip install transformers") from e
        
        # Load tokenizer first
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad token to eos token")
        
        # Memory profiling helper
        def log_gpu_memory(stage: str):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"[GPU MEM] {stage}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

        log_gpu_memory("Before model load")

        # Load model
        logger.info("Loading model...")
        model_kwargs = {
            "torch_dtype": torch.float16 if precision == "fp16" else torch.bfloat16 if precision == "bf16" else torch.float32,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        
        # Add quantization if specified
        if load_in_4bit:
            logger.info("Loading model with 4-bit quantization...")
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            model_kwargs["bnb_4bit_quant_type"] = "nf4"
            model_kwargs["bnb_4bit_use_double_quant"] = True
        elif load_in_8bit:
            logger.info("Loading model with 8-bit quantization...")
            model_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        log_gpu_memory("After model load")

        # Apply PEFT if specified or if quantization is used
        if use_peft:
            logger.info("Applying PEFT (LoRA) configuration...")
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Prepare model for k-bit training if using quantization
            if load_in_4bit or load_in_8bit:
                logger.info("Preparing model for k-bit training...")
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Extract PEFT config values safely
            lora_r = self._get_config_value(self.config.model, 'lora_r', 'r', default=16)
            lora_alpha = self._get_config_value(self.config.model, 'lora_alpha', 'alpha', default=32)
            lora_target_modules = self._get_config_value(
                self.config.model,
                'lora_target_modules',
                'target_modules',
                default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # All 7 modules like training_script.py
            )
            lora_dropout = self._get_config_value(self.config.model, 'lora_dropout', 'dropout', default=0.1)  # Match training_script.py
            
            logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            logger.info(f"Target modules: {lora_target_modules}")
            
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            logger.info("PEFT adapters applied successfully")
            log_gpu_memory("After PEFT setup")

        logger.info("=" * 80)
        logger.info("Counterfactual GRPO model setup completed successfully")
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        logger.info("=" * 80)
    
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

    def setup_data(self) -> None:
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

    
    def setup_rewards(self) -> None:
        """Setup reward functions using the centralized registry system."""
        logger.info("Setting up reward functions for Counterfactual GRPO...")
        
        # Get reward configurations
        rewards_config = []
        if hasattr(self.config, 'rewards'):
            rewards_config = self.config.rewards if isinstance(self.config.rewards, list) else []
        
        if not rewards_config:
            logger.warning("No reward configurations found, using default rewards")
            rewards_config = [
                {"type": "length", "weight": 0.2, "params": {"min_length": 20, "max_length": 200}},
                {"type": "sentiment", "weight": 0.2, "params": {"positive_weight": 1.0}},
                {"type": "safety", "weight": 0.2, "params": {"strict": True}},
                {"type": "diversity", "weight": 0.2, "params": {}},
                {"type": "fluency", "weight": 0.2, "params": {}},
            ]
        
        # Load reward functions from the registry
        for reward_config in rewards_config:
            reward_type = reward_config.get('type', 'length')
            weight = reward_config.get('weight', 1.0)
            params = reward_config.get('params', {})
            
            try:
                # Special case: custom reward function passed directly
                if reward_type == 'custom' and 'reward_function' in params:
                    reward_func = params['reward_function']
                    logger.info(f"Loaded custom reward function (weight: {weight})")
                else:
                    # Use the rewards registry to get reward functions
                    from aligntune.rewards.registry import RewardRegistry as RewardsRegistry
                    from aligntune.rewards.core import RewardConfig, RewardType
                    
                    # Map common variations to standard names
                    reward_type_mapping = {
                        'math': 'math_reasoning',
                        'code': 'code_quality',
                    }
                    reward_type = reward_type_mapping.get(reward_type, reward_type)
                    
                    try:
                        # Convert reward type string to enum
                        reward_type_enum = RewardType[reward_type.upper()]
                        
                        # Create RewardConfig with weight and params
                        reward_cfg = RewardConfig(
                            reward_type=reward_type_enum,
                            weight=1.0,  # Weight will be applied separately
                            params=params
                        )
                        
                        # Get reward function from registry
                        reward_func_obj = RewardsRegistry.get_reward_function(reward_type, reward_cfg)
                        # Extract the callable compute method
                        if hasattr(reward_func_obj, 'compute'):
                            reward_func = reward_func_obj.compute
                        elif callable(reward_func_obj):
                            reward_func = reward_func_obj
                        else:
                            logger.warning(f"Reward function '{reward_type}' is not callable, skipping")
                            continue
                        
                        logger.info(f"Loaded {reward_type} reward from registry (weight: {weight})")
                        
                    except KeyError:
                        # Reward type not in enum, try registry by name
                        logger.warning(f"Reward type '{reward_type}' not in RewardType enum, trying registry by name")
                        reward_func_obj = RewardsRegistry.get_reward_function(reward_type)
                        
                        if hasattr(reward_func_obj, 'compute'):
                            reward_func = reward_func_obj.compute
                        elif callable(reward_func_obj):
                            reward_func = reward_func_obj
                        else:
                            logger.warning(f"Reward function '{reward_type}' is not callable, skipping")
                            continue
                        
                        logger.info(f"Loaded {reward_type} reward by name (weight: {weight})")
                
                # Store reward function with metadata
                self.reward_functions.append({
                    "function": reward_func,
                    "weight": weight,
                    "name": reward_type
                })
                
            except Exception as e:
                logger.warning(f"Failed to load reward function '{reward_type}': {e}")
                logger.debug(f"Error details:", exc_info=True)
                
                # Fallback: try to continue with other rewards rather than failing
                continue
        
        if not self.reward_functions:
            logger.error("No reward functions were loaded! Adding a simple default length reward.")
            # Add a simple fallback reward so training doesn't fail
            def default_length_reward(text, reference=None, **kwargs):
                length = len(text.split())
                if length < 20:
                    return length / 20.0 * 0.5
                elif length > 200:
                    return max(0.0, 1.0 - (length - 200) / 200.0)
                else:
                    return 1.0
            
            self.reward_functions.append({
                "function": default_length_reward,
                "weight": 1.0,
                "name": "default_length"
            })
            logger.info("Added default length reward as fallback")
        
        logger.info(f"✓ Configured {len(self.reward_functions)} reward functions successfully")
        
        # Log summary of loaded rewards
        reward_summary = ", ".join([f"{rf['name']} ({rf['weight']:.2f})" for rf in self.reward_functions])
        logger.info(f"Reward functions: {reward_summary}")
    
    def _combined_reward_function(self, completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        """Outcome-based reward function matching training_script.py.

        Args:
            completions: List of generated completions
            prompts: List of prompts (used to look up gold answers)
            **kwargs: Additional arguments
        """
        import re
        import math

        # DEBUG: Verify function is being called
        print(f"\n[DEBUG] _combined_reward_function called with {len(completions) if completions else 0} completions")
        print(f"[DEBUG] prompts received: {len(prompts) if prompts else 'None'}")
        print(f"[DEBUG] prompt_to_answer has {len(self.prompt_to_answer)} entries")
        if prompts and len(prompts) > 0:
            sample_prompt = prompts[0][:100] + "..." if len(prompts[0]) > 100 else prompts[0]
            print(f"[DEBUG] Sample prompt from TRL: '{sample_prompt}'")
            # Check if prompt exists in mapping
            found = prompts[0] in self.prompt_to_answer
            print(f"[DEBUG] Prompt found in mapping: {found}")
            if not found and self.prompt_to_answer:
                # Show first key from mapping for comparison
                first_key = list(self.prompt_to_answer.keys())[0]
                sample_key = first_key[:100] + "..." if len(first_key) > 100 else first_key
                print(f"[DEBUG] Sample key from mapping: '{sample_key}'")

        def parse_pred_number(text: str) -> str:
            """Parse predicted number from model output."""
            nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            candidate = nums[-1] if nums else ""
            if re.fullmatch(r"-?\d+/\d+", candidate):
                num, den = candidate.split("/")
                try:
                    return str(float(num) / float(den))
                except:
                    return candidate
            return candidate

        def numeric_equal(a: str, b: str, rtol=1e-4, atol=1e-8) -> bool:
            """Check if two numeric strings are equal."""
            try:
                fa = float(a)
                fb = float(b)
                return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
            except:
                return a.strip() == b.strip()

        # Detect dataset type for appropriate grading
        dataset_name = ""
        if hasattr(self.config, 'datasets') and self.config.datasets:
            ds = self.config.datasets[0]
            dataset_name = ds.get('name', '') if isinstance(ds, dict) else getattr(ds, 'name', '')
        is_gsm8k = 'gsm8k' in dataset_name.lower()
        is_math = 'math' in dataset_name.lower() and 'gsm' not in dataset_name.lower()

        # Log grading mode once
        if not hasattr(self, '_logged_grading_mode'):
            grading_mode = "GSM8K (numeric)" if is_gsm8k else "MATH (symbolic)"
            print(f"\n[REWARD] Using {grading_mode} grading for dataset: {dataset_name}")
            self._logged_grading_mode = True

        if not completions:
            return []

        # Get gold answers from prompt→answer mapping (CRITICAL!)
        if prompts:
            answers = [self.prompt_to_answer.get(p, "0") for p in prompts]
        else:
            answers = list(self.prompt_to_answer.values())

        # Expand answers for multiple completions per prompt
        factor = max(1, len(completions) // len(answers)) if answers else 1
        expanded = [a for a in answers for _ in range(factor)][:len(completions)]

        rewards = []
        correct_count = 0
        debug_examples = []  # Collect examples for debug output

        for i, (comp, gold) in enumerate(zip(completions, expanded)):
            # Handle different completion formats
            if isinstance(comp, str):
                text = comp
            elif isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict):
                text = comp[0].get("content", str(comp))
            else:
                text = str(comp)

            # Parse predicted answer and compare - use dataset-appropriate grading
            if is_gsm8k:
                # GSM8K: Simple numeric comparison (fast)
                pred = parse_pred_number(text)
                is_correct = numeric_equal(pred, gold)
            else:
                # MATH: Robust symbolic grading (handles LaTeX, tuples, fractions)
                pred = extract_math_answer(text)
                is_correct = grade_math_answer(pred, gold)
            reward = 1.0 if is_correct else 0.0
            rewards.append(reward)

            if is_correct:
                correct_count += 1

            # Collect first 3 examples for debug output
            if i < 3:
                debug_examples.append({
                    'gold': gold,
                    'pred': pred,
                    'reward': reward,
                    'text_end': text[-80:] if len(text) > 80 else text
                })

        # Log accuracy
        if not hasattr(self, '_reward_step'):
            self._reward_step = 0
        self._reward_step += 1

        accuracy = correct_count / len(rewards) if rewards else 0.0
        print(f"\n🎯 Outcome Reward (step {self._reward_step}): {accuracy:.2%} ({correct_count}/{len(rewards)})")
        print("   Sample gold/pred/reward:")
        for j, ex in enumerate(debug_examples):
            status = "✓" if ex['reward'] > 0 else "✗"
            print(f"   [{j}] gold='{ex['gold']}' pred='{ex['pred']}' reward={ex['reward']:.1f} {status}")
            print(f"       ...{ex['text_end']}")
        print()
        import sys; sys.stdout.flush()

        return rewards
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step."""
        logger.debug("train_step() called but Counterfactual GRPO uses TRL's internal training loop")
        return {"loss": 0.0}
    
    def create_data_loader(self) -> Optional[DataLoader]:
        """Create data loader for training."""
        logger.debug("create_data_loader() called but Counterfactual GRPO uses TRL's internal data loading")
        return None
    
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
    

    def train(self) -> Dict[str, Any]:
        """Execute Counterfactual GRPO training."""
        from aligntune.core.optimization import get_optimizer_for_config, get_scheduler_for_config
        # Setup components
        self.setup_model()
        self.setup_rewards()
        self.setup_data()

        # Get training parameters
        num_epochs = self._get_config_value(self.config.train, 'epochs', 'num_epochs', 'num_train_epochs', default=1)
        learning_rate = self._get_config_value(self.config.train, 'learning_rate', 'lr', default=1e-7)
        per_device_batch_size = self._get_config_value(self.config.train, 'per_device_batch_size', 'batch_size', default=1)
        gradient_accumulation_steps = self._get_config_value(self.config.train, 'gradient_accumulation_steps', default=32)
        max_grad_norm = self._get_config_value(self.config.train, 'max_grad_norm', default=0.0)  # 0.0 disables clipping (like original script)
        weight_decay = self._get_config_value(self.config.train, 'weight_decay', default=0.0)
        warmup_steps = self._get_config_value(self.config.train, 'warmup_steps', default=10)
        warmup_ratio = self._get_config_value(self.config.train, 'warmup_ratio', default=0.0)
        precision = self._get_config_value(self.config.model, 'precision', default='fp32')
        # Handle enum vs string - get string value if it's an enum
        if hasattr(precision, 'value'):
            precision = precision.value
        # Ensure 'auto' defaults to 'bf16' for better memory efficiency
        if precision == 'auto':
            precision = 'bf16'
        print(f"[PRECISION DEBUG] value: '{precision}', type: {type(precision)}, fp16={precision == 'fp16'}, bf16={precision == 'bf16'}")
        output_dir = self._get_config_value(self.config.logging, 'output_dir', default='./output/counterfactual_grpo')

        # GRPO specific parameters - defaults matched to working training script
        beta = self._get_config_value(self.config.train, 'beta', 'kl_coef', default=0.005)  # Critical: was 0.1, now 0.005
        epsilon = self._get_config_value(self.config.train, 'epsilon', 'cliprange', default=0.2)
        loss_type = self._get_config_value(self.config.train, 'loss_type', default='dapo')  # TRL default is 'dapo'
        if loss_type == "sigmoid":
            loss_type = "dapo"
        scale_rewards = self._get_config_value(self.config.train, 'scale_rewards', default='group')

        # Generation parameters
        num_generations = self._get_config_value(self.config.train, 'num_generations', default=per_device_batch_size)
        mask_truncated_completions = self._get_config_value(self.config.train, 'mask_truncated_completions', default=True)
        temperature = self._get_config_value(self.config.train, 'temperature', default=0.6)
        top_p = self._get_config_value(self.config.train, 'top_p', default=0.95)
        reward_weights = self._get_config_value(self.config.train, 'reward_weights', default=[1.0])

        # Seed parameters
        seed = self._get_config_value(self.config.train, 'seed', default=42)
        data_seed = self._get_config_value(self.config.train, 'data_seed', default=47)  # Match training_script.py default

        # Logging parameters
        logging_steps = self._get_config_value(self.config.train, 'logging_steps', default=10)
        save_strategy = self._get_config_value(self.config.train, 'save_strategy', default='steps')
        save_total_limit = self._get_config_value(self.config.train, 'save_total_limit', default=None)

        # Run name for wandb/logging
        run_name = self._get_config_value(self.config.logging, 'run_name', default='counterfactual_grpo')


        # Use config-specified optimizer or default to adamw_torch
        optimizer_name = getattr(self.config.train, 'optimizer', 'adamw_torch')
        scheduler_name = getattr(self.config.train, 'lr_scheduler', 'cosine')

        # Create optimizer configuration
        optimizer_config = get_optimizer_for_config(
                optimizer_name,
                self.config.train.learning_rate,
                self.config.train.weight_decay or 0.01
            )
        # # Create scheduler configuration
        # scheduler_config = get_scheduler_for_config(
        #         scheduler_name,
        #         # max_steps,
        #         # warmup_steps
        #     )

        # Convert optimizer kwargs to string format for optim_args
        def kwargs_to_str(kwargs_dict):
            """Convert optimizer/scheduler kwargs dict to string format."""
            return ",".join(
                f"{k}={f'({','.join(map(str, v))})' if isinstance(v, tuple) else v}" 
                for k, v in kwargs_dict.items()
            ) if kwargs_dict else None

        optim_args_str = kwargs_to_str(optimizer_config['optimizer_kwargs'])

        # Gradient checkpointing (critical for VRAM savings!)
        use_gradient_checkpointing = self._get_config_value(self.config.train, 'use_gradient_checkpointing', default=True)

        # Counterfactual GRPO specific parameters
        boost_factor = self._get_config_value(self.config.train, 'boost_factor', default=2.0)
        min_weight = self._get_config_value(self.config.train, 'min_weight', default=0.5)
        max_spans = self._get_config_value(self.config.train, 'max_spans', default=10)
        answer_weight = self._get_config_value(self.config.train, 'answer_weight', default=1.5)
        enable_gradient_conservation = self._get_config_value(self.config.train, 'enable_gradient_conservation', default=False)
        weight_debug = self._get_config_value(self.config.train, 'weight_debug', default=False)

        # Extra verbose mode for paper examples
        extra_verbose = self._get_config_value(self.config.train, 'extra_verbose', default=False)
        extra_verbose_log_path = self._get_config_value(self.config.train, 'extra_verbose_log_path', default=None)
        extra_verbose_sample_rate = self._get_config_value(self.config.train, 'extra_verbose_sample_rate', default=0.1)

        # New unified weighting_mode: counterfactual, random, inverted, vanilla
        weighting_mode = self._get_config_value(self.config.train, 'weighting_mode', default=None)
        # Legacy params (backward compat)
        random_importance = self._get_config_value(self.config.train, 'random_importance', default=False)
        invert_importance = self._get_config_value(self.config.train, 'invert_importance', default=False)
        method_name = self._get_config_value(self.config.train, 'method_name', default='counterfactual')

        # Convert legacy to weighting_mode if not explicitly set
        if weighting_mode is None:
            if random_importance:
                weighting_mode = "random"
            elif invert_importance:
                weighting_mode = "inverted"
            elif method_name == "baseline":
                weighting_mode = "vanilla"
            else:
                weighting_mode = "counterfactual"
        
        
        
        # Checkpointing
        save_steps = self._get_config_value(self.config.train, 'save_steps', default=500)

        # Evaluation (eval_steps hardcoded as 100)
        eval_steps = self._get_config_value(self.config.train, 'eval_steps', default=100)
        eval_strategy = self._get_config_value(self.config.train, 'eval_strategy', default='steps')

        # Best model loading (not in config at all)
        load_best_model_at_end = self._get_config_value(self.config.train, 'load_best_model_at_end', default=False)
        metric_for_best_model = self._get_config_value(self.config.train, 'metric_for_best_model', default=None)
        greater_is_better = self._get_config_value(self.config.train, 'greater_is_better', default=False)

        # Logging strategy (not in config)
        logging_strategy = self._get_config_value(self.config.train, 'logging_strategy', default='steps')

        # Max steps (overrides num_train_epochs if set)
        max_steps = self._get_config_value(self.config.train, 'max_steps', default=-1)
        max_completion_length = self._get_config_value(self.config.train, 'max_completion_length', default=256)
        max_prompt_length = self._get_config_value(self.config.train, 'max_prompt_length', default=512)
        enable_thinking = self._get_config_value(self.config.train, 'enable_thinking', default=False)
        gradient_accumulation_steps = self._get_config_value(self.config.train, 'gradient_accumulation_steps', default=32)

        # Print all config parameters for visibility
        print("\n" + "=" * 80)
        print("COUNTERFACTUAL GRPO TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"  Model:                    {self._get_config_value(self.config.model, 'name_or_path', 'model_name')}")
        print(f"  Output dir:               {output_dir}")
        print(f"  Run name:                 {run_name}")
        print("-" * 80)
        print("TRAINING PARAMS:")
        print(f"  epochs:                   {num_epochs}")
        print(f"  max_steps:                {max_steps}")
        print(f"  per_device_batch_size:    {per_device_batch_size}")
        print(f"  num_generations:          {num_generations}")
        print(f"  gradient_accumulation:    {gradient_accumulation_steps}")
        print(f"  learning_rate:            {learning_rate}")
        print(f"  weight_decay:             {weight_decay}")
        print(f"  warmup_steps:             {warmup_steps}")
        print(f"  warmup_ratio:             {warmup_ratio}")
        print(f"  max_grad_norm:            {max_grad_norm}")
        print(f"  gradient_checkpointing:   {use_gradient_checkpointing}")
        print("-" * 80)
        print("GENERATION PARAMS:")
        print(f"  max_prompt_length:        {max_prompt_length}")
        print(f"  max_completion_length:    {max_completion_length}")
        print(f"  mask_truncated:           {mask_truncated_completions}")
        print(f"  enable_thinking:          {enable_thinking}")
        print("-" * 80)
        print("GRPO PARAMS:")
        print(f"  loss_type:                {loss_type}")
        print(f"  beta (KL coef):           {beta}")
        print(f"  epsilon (clip range):     {epsilon}")
        print(f"  scale_rewards:            {scale_rewards}")
        print(f"  temperature:              {temperature}")
        print(f"  top_p:                    {top_p}")
        print(f"  max_grad_norm:            {max_grad_norm} (0.0=disabled)")
        print("-" * 80)
        print("COUNTERFACTUAL PARAMS:")
        print(f"  weighting_mode:           {weighting_mode}")
        if weighting_mode != "vanilla":
            print(f"  boost_factor:             {boost_factor}")
            print(f"  min_weight:               {min_weight}")
            print(f"  max_spans:                {max_spans}")
            print(f"  answer_weight:            {answer_weight}")
            print(f"  gradient_conservation:    {enable_gradient_conservation}")
        print(f"  weight_debug:             {weight_debug}")
        print("-" * 80)
        print("CHECKPOINTING:")
        print(f"  save_steps:               {save_steps}")
        print(f"  save_total_limit:         {save_total_limit}")
        print(f"  save_strategy:            {save_strategy}")
        print(f"  logging_steps:            {logging_steps}")
        print(f"  eval_steps:               {eval_steps}")
        print("-" * 80)
        print("SEEDS:")
        print(f"  seed:                     {seed}")
        print(f"  data_seed:                {data_seed}")
        print("=" * 80 + "\n")
        import sys; sys.stdout.flush()
        
        # Generation parameters
        # num_generations = self._get_config_value(self.config.train, 'num_generations', default=None)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Setup Counterfactual GRPO trainer
        from trl import GRPOConfig
        from .custom_trainer import CounterfactualGRPOTrainer
        print(num_generations)

        # TRL requires num_train_epochs to be a number (not None)
        effective_epochs = num_epochs if num_epochs is not None else 1
        effective_max_steps = max_steps if max_steps is not None else -1

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            run_name=run_name,  # For wandb/logging
            num_train_epochs=effective_epochs,
            max_steps=effective_max_steps,
            per_device_train_batch_size=per_device_batch_size,
            num_generations=num_generations,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,

            # Learning rate & optimizer
            learning_rate=learning_rate,
            optim=optimizer_name,
            optim_args=optim_args_str,
            weight_decay=optimizer_config['optimizer_kwargs'].get('weight_decay', weight_decay),

            # Scheduler
            lr_scheduler_type=scheduler_name,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,

            # Gradient clipping
            max_grad_norm=max_grad_norm,

            # Logging & checkpointing
            logging_steps=logging_steps,
            logging_strategy=logging_strategy,  # ADD THIS
            save_steps=save_steps,  # CHANGE from hardcoded 100
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,

            # Evaluation
            eval_steps=eval_steps,  # CHANGE from hardcoded 100
            eval_strategy=eval_strategy,  # ADD THIS

            # Best model loading
            load_best_model_at_end=load_best_model_at_end,  # ADD THIS
            metric_for_best_model=metric_for_best_model,  # ADD THIS
            greater_is_better=greater_is_better,  # ADD THIS

            # Precision
            remove_unused_columns=False,
            fp16=precision == "fp16",
            bf16=precision == "bf16",

            # Gradient checkpointing (critical for VRAM!)
            gradient_checkpointing=use_gradient_checkpointing,

            # GRPO specific
            loss_type=loss_type,
            beta=beta,
            epsilon=epsilon,
            scale_rewards=scale_rewards,
            temperature=temperature,
            top_p=top_p,
            mask_truncated_completions=mask_truncated_completions,  # CRITICAL: was missing, TRL default=False
            reward_weights=reward_weights,  # Was missing

            # Seeds
            seed=seed,
            data_seed=data_seed,

            # Reporting
            report_to=self.config.logging.loggers if self.config.logging.loggers else [],
        )

#         grpo_config = GRPOConfig(
#                     output_dir=output_dir,
#                     num_train_epochs=num_epochs,
#                     per_device_train_batch_size=per_device_batch_size,
#                     num_generations = per_device_batch_size,
#                     gradient_accumulation_steps=gradient_accumulation_steps,
#                     learning_rate=learning_rate,
#                     warmup_steps=warmup_steps,
#                     logging_steps=10,
#                     save_steps=100,
#                     eval_steps=100,
#                     weight_decay=weight_decay,
#                     max_grad_norm=max_grad_norm,
#                     remove_unused_columns=False,
#                     fp16=precision == "fp16",
#                     bf16=precision == "bf16",
#                     beta=beta,
#                     epsilon=epsilon,
#                 )

#       # Create Counterfactual GRPO trainer
        import os
        should_use_pure_trl = os.environ.get('PURE_TRL_MODE', '0') == '1'

        # Memory profiling before trainer creation
        def log_gpu_memory(stage: str):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"[GPU MEM] {stage}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

        log_gpu_memory("Before trainer creation")

        # Reward wrapper that matches TRL's expected signature
        def reward_wrapper(completions, prompts=None, **kwargs):
            """Wrapper to pass prompts to reward function for gold answer lookup."""
            print(f"\n[REWARD_WRAPPER] Called! completions={len(completions) if completions else 'None'}, prompts={len(prompts) if prompts else 'None'}")
            print(f"[REWARD_WRAPPER] kwargs keys: {list(kwargs.keys())}")
            if completions is None:
                return [0.0] * (len(prompts) if prompts else 1)
            return self._combined_reward_function(completions, prompts=prompts, **kwargs)

        if should_use_pure_trl:
            logger.info("PURE_TRL_MODE enabled - using pure TRL API with CounterfactualGRPOTrainer")
            self.trainer = CounterfactualGRPOTrainer(
                model=self.model,
                args=grpo_config,
                train_dataset=self.train_dataset,
                processing_class=self.tokenizer,
                reward_funcs=[reward_wrapper],  # Must be a list, passes prompts!
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
        else:
            try:
                import unsloth
                logger.info("Detected Unsloth environment, using reward_funcs parameter")

                self.trainer = CounterfactualGRPOTrainer(
                    model=self.model,
                    args=grpo_config,
                    train_dataset=self.train_dataset,
                    processing_class=self.tokenizer,
                    reward_funcs=[reward_wrapper],  # Must be a list, passes prompts!
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
            except ImportError:
                logger.info("Using pure TRL, using reward_function parameter")
                self.trainer = CounterfactualGRPOTrainer(
                    model=self.model,
                    args=grpo_config,
                    train_dataset=self.train_dataset,
                    processing_class=self.tokenizer,
                    reward_funcs=[reward_wrapper],  # Must be a list, passes prompts!
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

        log_gpu_memory("After trainer creation")


        print(self.trainer.args)
        # Record training start
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting Counterfactual GRPO Training")
        logger.info(f"Dataset size: {len(self.train_dataset)}")
        logger.info(f"Num reward functions: {len(self.reward_functions)}")
        logger.info("=" * 80)
        
        # Start training
        train_result = self.trainer.train()
        
        # Log qualitative samples if available
        try:
            generate_and_log_samples(
                self.config.logging.sample_logging,
                self.model,
                self.tokenizer,
                getattr(self, 'reward_functions', None),
                stage="post-train",
                log=logger,
            )
        except Exception as sample_error:
            logger.warning(f"Unable to log qualitative samples: {sample_error}")
        
        # Record training end
        end_time = time.time()
        training_duration = end_time - start_time
        
        logger.info(f"Counterfactual GRPO training completed in {training_duration:.2f} seconds")
        
        # Extract metrics
        metrics = {}
        if hasattr(train_result, 'metrics'):
            metrics = train_result.metrics
        
        # Save model
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Compile results
        results = {
            "training_time": training_duration,
            "final_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else metrics.get('train_loss', 0.0),
            "total_steps": train_result.global_step if hasattr(train_result, 'global_step') else 0,
            "model_path": output_dir,
            "num_reward_functions": len(self.reward_functions),
            "num_datasets": 1,
            "metrics": metrics,
            "counterfactual_params": {
                "boost_factor": boost_factor,
                "min_weight": min_weight,
                "max_spans": max_spans,
                "answer_weight": answer_weight,
                "method_name": method_name,
                "random_importance": random_importance,
                "invert_importance": invert_importance,
                "enable_gradient_conservation": enable_gradient_conservation,
            }
        }
        
        logger.info("=" * 80)
        logger.info("Counterfactual GRPO Training Completed Successfully!")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Total steps: {results['total_steps']}")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info("=" * 80)
        
        return results
    
    # def evaluate(self) -> Dict[str, Any]:
    #     """Evaluate the trained model."""
    #     logger.info("Starting Counterfactual GRPO evaluation...")
        
    #     if not hasattr(self, 'trainer') or self.trainer is None:
    #         logger.warning("No trainer available for evaluation")
    #         return {}
        
    #     if hasattr(self.trainer, 'evaluate'):
    #         eval_results = self.trainer.evaluate()
    #         logger.info(f"Evaluation completed: {eval_results}")
    #         return eval_results
        
    #     return {}
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        num_epochs = self._get_config_value(self.config.train, 'epochs', 'num_epochs', 'num_train_epochs', default=1)
        
        # Counterfactual specific parameters
        boost_factor = self._get_config_value(self.config.train, 'boost_factor', default=2.0)
        min_weight = self._get_config_value(self.config.train, 'min_weight', default=0.5)
        max_spans = self._get_config_value(self.config.train, 'max_spans', default=10)
        method_name = self._get_config_value(self.config.train, 'method_name', default='counterfactual')
        
        stats = {
            'config': {
                'model_name': self.config.model.name_or_path,
                'task_type': 'counterfactual_group_relative_policy_optimization',
                'dataset_name': self.config.dataset.name,
                'epochs': num_epochs,
                'learning_rate': self._get_config_value(self.config.train, 'learning_rate', 'lr', default=2e-4),
                'batch_size': self._get_config_value(self.config.train, 'per_device_batch_size', 'batch_size', default=4),
                'use_peft': self._get_config_value(self.config.model, 'use_peft', default=False),
                'precision': self._get_config_value(self.config.model, 'precision', default='fp32'),
                'num_reward_functions': len(self.reward_functions),
                'counterfactual_method': method_name,
                'boost_factor': boost_factor,
                'min_weight': min_weight,
                'max_spans': max_spans,
            },
            'dataset_info': {
                'train_size': len(self.train_dataset) if hasattr(self, 'dataset') and self.train_dataset else 0,
                'val_size': 0,
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
        """Save configuration to YAML file."""
        num_epochs = self._get_config_value(self.config.train, 'epochs', 'num_epochs', 'num_train_epochs', default=1)
        
        # Counterfactual specific parameters
        boost_factor = self._get_config_value(self.config.train, 'boost_factor', default=2.0)
        min_weight = self._get_config_value(self.config.train, 'min_weight', default=0.5)
        max_spans = self._get_config_value(self.config.train, 'max_spans', default=10)
        method_name = self._get_config_value(self.config.train, 'method_name', default='counterfactual')
        
        config_dict = {
            'model_name': self.config.model.name_or_path,
            'task_type': 'counterfactual_group_relative_policy_optimization',
            'max_seq_length': self._get_config_value(self.config.model, 'max_seq_length', default=512),
            'learning_rate': self._get_config_value(self.config.train, 'learning_rate', 'lr', default=2e-4),
            'epochs': num_epochs,
            'batch_size': self._get_config_value(self.config.train, 'per_device_batch_size', 'batch_size', default=4),
            'dataset_name': self.config.dataset.name,
            'use_peft': self._get_config_value(self.config.model, 'use_peft', default=False),
            'precision': self._get_config_value(self.config.model, 'precision', default='fp32'),
            'num_reward_functions': len(self.reward_functions),
            'counterfactual_method': method_name,
            'boost_factor': boost_factor,
            'min_weight': min_weight,
            'max_spans': max_spans,
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Counterfactual GRPO configuration saved to {path}")
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model."""
        try:
            if isinstance(self.config.logging, dict):
                default_path = self.config.logging.get(
                    'output_dir', './output/counter_factgrpo')
            else:
                default_path = getattr(self.config.logging,
                                       'output_dir',
                                       './output/counter_factgrpo') if hasattr(self.config,
                                                                   'logging') else './output/counterfact_grpo'

            save_path = path or default_path

            logger.info(f"Saving  model to: {save_path}")

            # Save using Unsloth's optimized saving
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            # Save training configuration
            config_path = Path(save_path) / "counterfact_grpo_training_config.yaml"
            with open(config_path, "w") as f:
                config_dict = self.config.to_dict() if hasattr(
                    self.config, 'to_dict') else self.config
                yaml.dump(config_dict, f, default_flow_style=False)

            logger.info(f"GRPO model saved successfully to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save GRPO model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """Load a trained GRPO model."""
        try:
            logger.info(f"Loading Unsloth GRPO model from: {path}")

            import unsloth
            from unsloth import FastLanguageModel

            # Handle both dict and object config
            if isinstance(self.config.model, dict):
                max_seq_length = self.config.model.get('max_seq_length', 2048)
            else:
                max_seq_length = getattr(
                    self.config.model, 'max_seq_length', 2048)

            # Load model and tokenizer
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            logger.info("GRPO model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load GRPO model: {e}")
            raise
