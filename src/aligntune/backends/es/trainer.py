"""
Evolution Strategies (ES) Trainer - Full Implementation with Detailed Algorithm Comments.

This module implements the complete ES training loop for fine-tuning language models.
The algorithm works by:
1. Creating a population of perturbed LoRA weight variants
2. Generating solutions with each variant and scoring them
3. Computing gradients from fitness scores and noise
4. Updating the base weights with the ES gradient

This approach enables population-based optimization without backpropagation through the reward model.
"""

import logging
import torch
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import shutil
from tqdm import tqdm

from aligntune.core.es.trainer_base import ESTrainerBase
from aligntune.core.model_loader import build_model
from aligntune.data.manager import DataManager
from aligntune.data.schemas import TaskType
from aligntune.rewards.core import RewardFunctionFactory, RewardConfig, RewardType
from aligntune.core.rollout import HFRolloutBackend, VLLMRolloutBackend
from peft.peft_model import PeftModel
from ...core.logging.wandb_utils import WandBLogger

logger = logging.getLogger(__name__)


def _add_nvidia_cuda_libs_to_path() -> None:
    """Load pip nvidia-* libcudart into this process before importing vLLM."""
    import ctypes

    dirs = []
    try:
        import nvidia

        root = Path(next(iter(nvidia.__path__)))
        for so in root.rglob("libcudart.so*"):
            dirs.append(so.parent)
    except Exception:
        return
    loaded = False
    for d in dict.fromkeys(dirs):
        for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
            path = d / name
            if not path.exists():
                continue
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                loaded = True
            except OSError:
                continue
        extra = str(d)
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = extra if not current else extra + ":" + current
    if loaded:
        logger.info("Loaded NVIDIA CUDA runtime for vLLM")


@dataclass
class AdapterInfo:
    """Metadata for a perturbed LoRA adapter variant."""
    path: str  # Path to saved adapter weights
    noise_a: torch.Tensor  # Random noise added to LoRA matrix A
    noise_b: torch.Tensor  # Random noise added to LoRA matrix B
    seed: int  # Seed for deterministic noise recreation


class ESTask:
    """
    Wrapper around dataset that provides batch sampling for ES evaluation.

    In ES, we need to evaluate each adapter on the same set of prompts for fair comparison.
    This class handles batching and cycling through the dataset.
    """

    def __init__(self, dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_idx = 0
        self.epoch = 0

    def get_batch(self) -> Dict[str, Any]:
        """
        Returns the next batch of prompts from the dataset.
        Cycles to beginning when reaching end.
        """
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, len(self.dataset))

        # Cycle back to beginning when reaching end
        if end_idx - start_idx < self.batch_size:
            remaining = self.batch_size - (end_idx - start_idx)
            batch = self.dataset[start_idx:end_idx]
            batch_end = self.dataset[0:remaining]

            # Merge batch and batch_end
            for key in batch:
                if isinstance(batch[key], list):
                    batch[key].extend(batch_end[key])

            self.current_idx = remaining
            self.epoch += 1
        else:
            batch = self.dataset[start_idx:end_idx]
            self.current_idx = end_idx

        return batch


class ESTrainerBackend(ESTrainerBase):
    """
    Complete Evolution Strategies Training Implementation.

    Algorithm Overview:
    ==================
    ES is a population-based black-box optimization method that:

    1. INITIALIZATION:
       - Load base model with LoRA adapters
       - Initialize LoRA weights (A, B matrices)

    2. PER ITERATION:
       a) POPULATION GENERATION:
          For i in [1, population_size]:
            - Create random noise: noise_a ~ N(0, sigma), noise_b ~ N(0, sigma)
            - Perturb: A_i = A_base + noise_a, B_i = B_base + noise_b
            - Save adapter to /dev/shm (RAM disk) for fast loading

       b) EVALUATION:
          For each adapter i:
            - Load adapter into vLLM (continuous batching engine)
            - Generate solutions for all prompts in batch
            - Score each solution with reward function
            - Fitness_i = average reward across batch

       c) GRADIENT COMPUTATION:
          gradient = 0
          For i in [1, population_size]:
            - Recreate noise_i using stored seed
            - gradient += (Fitness_i / sigma) * noise_i
          gradient /= population_size

       d) WEIGHT UPDATE:
          A_base = A_base + learning_rate * gradient_a
          B_base = B_base + learning_rate * gradient_b

    Key Features:
    - No gradients through reward model (true black-box optimization)
    - Population-based: better exploration than single-point gradient descent
    - Derivative-free: works with non-differentiable reward functions
    - Parallel evaluation: all adapters evaluated in batches via vLLM
    """

    TASK_TYPE = "grpo"
    KEEP_COLUMNS = True

    def __init__(self, config: Any, callbacks: Optional[List] = None):
        super().__init__(config, callbacks)
        self.task = None
        self.vllm_engine = None
        self.base_model = None
        self.config = config

        # Setup adapter cache with intelligent fallback (RAM disk -> disk)
        self.adapter_cache_dir = self._setup_adapter_cache_dir()

        # Setup completion logging
        self.completion_log_path = Path(self.config.output_dir) / "completions.log"
        self.completion_log_path.parent.mkdir(parents=True, exist_ok=True)

    def setup_model(self) -> None:
        """Load base model with LoRA using build_model()."""
        logger.info("=" * 80)
        logger.info("ES: Setting up base model with LoRA adapters")
        logger.info("=" * 80)

        # Single call to build_model handles everything:
        # - Precision (fp32, fp16, bf16, auto)
        # - Quantization (4-bit, 8-bit)
        # - Device mapping (auto, DDP, FSDP)
        # - PEFT wrapping (LoRA, DoRA, Vera, Kron)
        self.base_model, self.tokenizer = build_model(
            config=self.config,
            task_type=TaskType.SFT,
            use_unsloth=getattr(self.config.es, "use_unsloth", False),
            apply_peft=True,  # ES currently supports LoRA adapters only.
            is_reference=False
        )

        self.model = self.base_model

        logger.info(f"Model loaded: {self.config.model.name_or_path}")
        logger.info("PEFT adapters applied and ready for training")

    def setup_data(self) -> None:
        """Load dataset via DataManager."""
        logger.info("=" * 80)
        logger.info("ES: Loading dataset via DataManager")
        logger.info("=" * 80)

        task_type = self.config.dataset.task_type or self.TASK_TYPE
        keep_columns = (
            self.config.dataset.keep_columns
            if self.config.dataset.keep_columns is not None
            else self.KEEP_COLUMNS
        )

        # DataManager handles:
        # - Auto-detection of dataset format
        # - Column mapping to standard schema
        # - System prompt injection with chat templates
        # - Train/validation splitting
        # ES consumes the GRPO prompt/reward schema.
        data_manager = DataManager(
            task_type=task_type,
            column_mapping=getattr(self.config.dataset, "column_mapping", None),
            system_prompt=getattr(self.config.dataset, "system_prompt", None),
            tokenizer=self.tokenizer,
            enable_thinking=getattr(self.config.dataset, "enable_thinking", False),
            processing_fn=getattr(self.config.dataset, "processing_fn", None),
            processing_batched=getattr(self.config.dataset, "processing_batched", False),
            max_samples=getattr(self.config.dataset, "max_samples", None),
            max_length=getattr(self.config.model, "max_seq_length", 1024),
            expected_format=getattr(self.config.dataset, "format_type", None),
            keep_columns=keep_columns,
            val_split_ratio=getattr(self.config.dataset, "val_split_ratio", None),
            test_split_ratio=getattr(self.config.dataset, "test_split_ratio", None),
            seed=getattr(self.config.dataset, "split_seed", 42),
            curator_schema_gate=getattr(self.config.dataset, "curator_schema_gate", True),
            curator_clean=getattr(self.config.dataset, "curator_clean", False),
            curator_dedup=getattr(self.config.dataset, "curator_dedup", "none"),
            curator_use_tiktoken=getattr(self.config.dataset, "curator_use_tiktoken", False),
            curator_max_tokens=getattr(self.config.dataset, "curator_max_tokens", 1_000_000),
        )

        # Load dataset with config_name if provided (same as SFT)
        config_name = getattr(self.config.dataset, 'subset', None)
        loader_kwargs = getattr(self.config.dataset, "loader_kwargs", {})

        dataset_dict = data_manager.load_dataset(
            self.config.dataset.name,
            config_name=config_name,
            split=getattr(self.config.dataset, "split", None),
            **loader_kwargs
        )

        train_dataset = dataset_dict.get("train", dataset_dict[list(dataset_dict.keys())[0]])
        self.dataset_dict = dataset_dict
        self.train_dataset = train_dataset
        self.eval_dataset = dataset_dict.get("validation")
        logger.info(f"Dataset loaded: {len(train_dataset)} examples")

        # Wrap in ESTask for batch sampling
        self.task = ESTask(
            dataset=train_dataset,
            batch_size=self.config.es.prompt_batch_size
        )

    def setup_rollout_backend(self) -> None:
        """Initialize rollout backend. vLLM if it loads, otherwise HF generate."""
        logger.info("=" * 80)
        logger.info("ES: Initializing rollout backend")
        logger.info("=" * 80)

        dtype = getattr(self.config.model, "dtype", "auto")
        max_model_len = getattr(self.config.model, "max_seq_length", 2048)
        _add_nvidia_cuda_libs_to_path()
        try:
            if VLLMRolloutBackend is None:
                raise ImportError("vLLM backend not imported")
            self.rollout_backend = VLLMRolloutBackend(
                model_name_or_path=self.config.model.name_or_path,
                tokenizer=self.tokenizer,
                dtype=dtype,
                max_model_len=max_model_len,
                tensor_parallel_size=getattr(self.config.es, "tensor_parallel_size", 1),
                enable_lora=True,
                max_lora_rank=getattr(self.config.model.peft, "rank", 8),
            )
            self.rollout_backend.initialize()
            logger.info("vLLM backend initialized with LoRA support")
        except Exception as e:
            logger.warning("vLLM unavailable (%s); falling back to Hugging Face generate", e)
            dt = (dtype or "auto").lower()
            if dt in ("fp16", "half"):
                dtype = "float16"
            elif dt in ("bf16",):
                dtype = "bfloat16"
            self.rollout_backend = HFRolloutBackend(
                model_name_or_path=self.config.model.name_or_path,
                dtype=dtype,
            )
            self.rollout_backend.initialize()
            self.rollout_backend.device = str(
                next(self.rollout_backend.model.parameters()).device
            )

    def setup_reward_function(self) -> None:
        """Initialize reward function(s) via RewardFunctionFactory."""
        logger.info("=" * 80)
        logger.info("ES: Setting up reward function(s)")
        logger.info("=" * 80)

        # Support both single reward and multiple rewards
        # Priority: config.rewards (list) > config.es.reward_type (single string)
        reward_functions = []

        if hasattr(self.config, 'rewards') and self.config.rewards:
            # Multiple rewards configuration (like GRPO/RLOO)
            for reward_config in self.config.rewards:
                # Handle both dict and object configs
                if isinstance(reward_config, dict):
                    reward_type = reward_config.get('type')
                    params = reward_config.get('params', {})
                else:
                    reward_type = getattr(reward_config, 'type', None)
                    params = getattr(reward_config, 'params', {})

                if not reward_type:
                    continue

                # Case 1: Custom function directly in params
                if 'function' in params and callable(params['function']):
                    reward_functions.append(params['function'])
                    logger.info(f"✓ Loaded custom reward function: {reward_type}")
                # Case 2: Callable wrapped in type field
                elif callable(reward_type):
                    reward_functions.append(reward_type)
                    logger.info(f"✓ Loaded callable reward from type field")
                # Case 3: String type - use RewardFunctionFactory
                else:
                    try:
                        reward_type_enum = RewardType[reward_type.upper()]
                        reward_cfg = RewardConfig(
                            reward_type=reward_type_enum,
                            weight=1.0,
                            params=params
                        )
                        reward_fn = RewardFunctionFactory.create_reward(reward_cfg)
                        reward_functions.append(reward_fn)
                        logger.info(f"✓ Loaded {reward_type} reward from factory")
                    except Exception as e:
                        logger.warning(f"Could not load reward '{reward_type}': {e}")

        elif hasattr(self.config.es, 'reward_type') and self.config.es.reward_type:
            # Single reward configuration (backward compatible)
            reward_type_str = self.config.es.reward_type
            reward_type_enum = RewardType[reward_type_str.upper()]
            reward_config = RewardConfig(
                reward_type=reward_type_enum,
                weight=1.0,
                params=getattr(self.config, "reward_config", {})
            )
            reward_fn = RewardFunctionFactory.create_reward(reward_config)
            reward_functions.append(reward_fn)
            logger.info(f"✓ Loaded single reward: {reward_type_str}")
        else:
            raise ValueError("No reward configuration found. Set config.rewards or config.es.reward_type")

        # Wrap reward function(s) to handle both RewardFunction objects and callables
        if not reward_functions:
            raise ValueError("No valid reward functions loaded")

        if len(reward_functions) > 1:
            logger.info(f"Using combined reward function with {len(reward_functions)} rewards (averaged)")

        self.reward_fn = self._create_combined_reward_fn(reward_functions)

    def _create_combined_reward_fn(self, reward_functions):
        """Create a combined reward function that averages multiple rewards."""
        def combined_reward(prompt=None, response=None, reference=None, **kwargs):
            scores = []
            for reward_fn in reward_functions:
                try:
                    # Check if it's a RewardFunction object (has compute method)
                    if hasattr(reward_fn, 'compute'):
                        # RewardFunction.compute(text, reference, **kwargs)
                        score = reward_fn.compute(text=response, reference=reference, **kwargs)
                    else:
                        # Custom callable function
                        score = reward_fn(prompt=prompt, response=response, reference=reference, **kwargs)
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Reward function error: {e}")
            return sum(scores) / len(scores) if scores else 0.0
        return combined_reward

    def generate_local_adapters(self, iteration: int) -> List[AdapterInfo]:
        """
        Create population_size variants by perturbing base LoRA weights.

        Algorithm:
        ==========
        For each member i in population (i = 0 to population_size-1):
          1. Generate seed deterministically from (iteration, i)
             This ensures noise can be recreated later

          2. Generate random noise from standard normal distribution
             noise_a ~ N(0, sigma), noise_b ~ N(0, sigma)
             Shape matches LoRA matrices: [hidden_dim, rank]

          3. Perturb LoRA weights
             A_i = A_base + noise_a
             B_i = B_base + noise_b

          4. Save adapter to /dev/shm (RAM disk)
             This allows vLLM to load different adapters per request
             Much faster than disk I/O

          5. Store metadata (path, noise, seed) for later gradient computation
             Seeds enable deterministic noise recreation without storing tensors

        Returns: List of AdapterInfo containing (path, noise_a, noise_b, seed)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ES Iteration {iteration}: Generating {self.config.es.population_size} adapter variants")
        logger.info(f"{'='*80}")

        adapters = []
        sigma = self.config.es.sigma
        population_size = self.config.es.population_size

        # Extract base LoRA weights from model
        if isinstance(self.base_model, PeftModel):
            base_adapter_name = self.base_model.active_adapters[0] if hasattr(self.base_model, 'active_adapters') else 'default'
        else:
            base_adapter_name = 'default'

        # Collect all LoRA matrices (A and B from each module)
        base_a_matrices = {}
        base_b_matrices = {}
        for name, module in self.base_model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                base_a_matrices[name] = module.lora_A[base_adapter_name].weight.data.clone()
                base_b_matrices[name] = module.lora_B[base_adapter_name].weight.data.clone()

        # Generate population
        for i in range(population_size):
            # Deterministic seed from iteration and member index
            seed = iteration * 1000 + i
            rng = np.random.RandomState(seed)

            logger.info(f"  Creating adapter {i+1}/{population_size}...")

            # Generate noise for each LoRA module and perturb the base model's
            # LoRA weights IN-PLACE, then persist via PeftModel.save_pretrained().
            #
            # NOTE: This used to build a `perturbed_a`/`perturbed_b` dict and
            # torch.save() it as a raw "weights.pt" blob. That format is NOT
            # a valid PEFT adapter (no adapter_config.json, wrong state-dict
            # key names), so vLLM's LoRARequest could never actually load it -
            # every population member silently evaluated the frozen base
            # model instead of its perturbed adapter, which made the entire
            # ES optimization loop a no-op (fitness differences were pure
            # sampling noise, and apply_es_update()'s weight update never
            # affected generation). Writing a real PEFT adapter directory via
            # save_pretrained() fixes this: vLLM can load it directly.
            noise_tensors_for_storage = {}

            for module_name in base_a_matrices.keys():
                base_a = base_a_matrices[module_name]
                base_b = base_b_matrices[module_name]

                # Generate noise with same shape as base weights
                noise_a = torch.tensor(
                    rng.normal(0, sigma, size=base_a.shape),
                    dtype=base_a.dtype,
                    device=base_a.device
                )
                noise_b = torch.tensor(
                    rng.normal(0, sigma, size=base_b.shape),
                    dtype=base_b.dtype,
                    device=base_b.device
                )

                # Store first module's noise for tracking
                if module_name not in noise_tensors_for_storage:
                    noise_tensors_for_storage[module_name] = (noise_a.clone(), noise_b.clone())

            # Apply perturbation: A_i = A_base + noise_a, B_i = B_base + noise_b
            for name, module in self.base_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and name in base_a_matrices:
                    noise_a, noise_b = noise_tensors_for_storage[name]
                    module.lora_A[base_adapter_name].weight.data.copy_(base_a_matrices[name] + noise_a)
                    module.lora_B[base_adapter_name].weight.data.copy_(base_b_matrices[name] + noise_b)

            # Save adapter in standard PEFT format (adapter_config.json +
            # adapter weights) so vLLM's LoRARequest can load it.
            # save_embedding_layers=False: PEFT's default 'auto' bundles the
            # full lm_head/embed_tokens tensors for models with tied word
            # embeddings (e.g. Qwen2.5-0.5B-Instruct), which vLLM's LoRA
            # loader rejects as "unsupported LoRA weight". ES never targets
            # the embedding layers, so they aren't needed in the adapter.
            adapter_path = self.adapter_cache_dir / f"adapter_iter{iteration}_member{i}"
            adapter_path.mkdir(parents=True, exist_ok=True)
            self.base_model.save_pretrained(str(adapter_path), save_embedding_layers=False)

            # Restore base weights before perturbing for the next population member
            for name, module in self.base_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and name in base_a_matrices:
                    module.lora_A[base_adapter_name].weight.data.copy_(base_a_matrices[name])
                    module.lora_B[base_adapter_name].weight.data.copy_(base_b_matrices[name])

            # Store metadata for this adapter
            adapter_info = AdapterInfo(
                path=str(adapter_path),
                noise_a=noise_tensors_for_storage[list(noise_tensors_for_storage.keys())[0]][0],
                noise_b=noise_tensors_for_storage[list(noise_tensors_for_storage.keys())[0]][1],
                seed=seed
            )
            adapters.append(adapter_info)

        logger.info(f"Generated {len(adapters)} adapter variants")
        return adapters

    def generate_and_score(self, adapters: List[AdapterInfo]) -> np.ndarray:
        """
        Evaluate all adapters on the same batch of prompts.

        Algorithm:
        ==========
        For each adapter i in population:
          1. Load adapter weights into vLLM engine
             - Point the LoRA weights in vLLM to adapter_i
             - vLLM will use these weights for generation

          2. Generate solutions for all prompts in current batch
             - Generate num_return_sequences variants per prompt
             - Use temperature for diversity

          3. Score all generated solutions with reward_fn
             - Reward function takes (prompt, solution) -> score
             - Can be any non-differentiable function

          4. Compute fitness = average reward across batch
             - This is the fitness score for this adapter

        Returns: Array of shape [population_size] with fitness scores

        Why batch evaluation?
        - All adapters evaluated on SAME prompts ensures fair comparison
        - Reduces variance in fitness estimation
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ES: Generating and scoring {len(adapters)} adapters")
        logger.info(f"{'='*80}")

        # Get current batch of prompts
        batch = self.task.get_batch()
        prompts = batch.get('prompt', batch.get('input', []))

        fitness_scores = []
        num_return_sequences = getattr(self.config.es, "num_return_sequences", 1)

        for idx, adapter_info in enumerate(adapters):
            logger.info(f"  Evaluating adapter {idx+1}/{len(adapters)}...")

            # vLLM takes lora_adapter_path. HF generate does not — load the
            # adapter onto the rollout model for this call only.
            # vLLM top_k=-1 means "off". HF generate requires top_k >= 1.
            top_k = self.config.es.top_k
            gen_kwargs = dict(
                prompts=prompts,
                max_new_tokens=self.config.es.max_new_tokens,
                temperature=self.config.es.temperature,
                top_p=self.config.es.top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
            )
            if isinstance(self.rollout_backend, HFRolloutBackend):
                if top_k is None or top_k <= 0:
                    gen_kwargs["top_k"] = 50
                wrapped = PeftModel.from_pretrained(
                    self.rollout_backend.model, adapter_info.path
                )
                base = self.rollout_backend.model
                self.rollout_backend.model = wrapped
                try:
                    generations = self.rollout_backend.generate(**gen_kwargs)
                finally:
                    self.rollout_backend.model = base
                    del wrapped
            else:
                generations = self.rollout_backend.generate(
                    **gen_kwargs, lora_adapter_path=adapter_info.path
                )

            # Score all generations
            scores = []
            for prompt_idx, prompt in enumerate(prompts):
                # Get reference for this prompt
                reference = batch.get('reference', [None] * len(prompts))[prompt_idx]
                logger.info(f"[DEBUG] Reference for prompt {prompt_idx}: {reference}")

                # generations[prompt_idx] is list of num_return_sequences outputs
                for generated_text in generations[prompt_idx]:
                    score = self.reward_fn(
                        prompt=prompt,
                        response=generated_text,
                        reference=reference
                    )
                    scores.append(score)

                    # Log completion and reward
                    with open(self.completion_log_path, "a") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"PROMPT: {prompt}\n")
                        f.write(f"COMPLETION: {generated_text}\n")
                        f.write(f"REWARD: {score:.4f}\n")

            # Fitness = average score for this adapter
            fitness = np.mean(scores) if scores else 0.0
            fitness_scores.append(fitness)

            logger.info(f"    Adapter {idx+1} fitness: {fitness:.4f}")

        fitness_array = np.array(fitness_scores)
        logger.info(f"Fitness scores computed: mean={fitness_array.mean():.4f}, std={fitness_array.std():.4f}")

        return fitness_array

    def apply_es_update(self, adapters: List[AdapterInfo], fitness_scores: np.ndarray) -> None:
        """
        Compute ES gradient from fitness scores and apply weight update.

        Algorithm:
        ==========
        The ES gradient (Natural ES) is computed as:

            gradient = (1 / (population_size * sigma)) * Σ(fitness_i * noise_i)

        Derivation (brief):
        - We want to maximize J(θ) = E[fitness(θ + σ*N)] where N ~ N(0, I)
        - Taking gradient: ∇J(θ) ∝ E[fitness(θ + σ*N) * N]
        - Empirically: approximate with finite samples from population

        Steps:
        1. NORMALIZE FITNESS SCORES
           - Subtract mean and divide by (std + epsilon)
           - Rescales fitness values to have mean 0, std 1
           - Prevents premature convergence to local optima
           - Makes learning rate more stable

        2. ACCUMULATE GRADIENT
           For each adapter i:
             - Recreate noise_i using stored seed (deterministic)
             - weighted_noise = (normalized_fitness_i / sigma) * noise_i
             - gradient_accumulator += weighted_noise
           - gradient = gradient_accumulator / population_size

        3. UPDATE BASE WEIGHTS
           For each LoRA module:
             - A_new = A_old + learning_rate * gradient_a
             - B_new = B_old + learning_rate * gradient_b
           - Update the base model's LoRA weights

        Why this works:
        - High fitness -> noise added in that direction -> move weights that way
        - Low fitness -> noise added in that direction -> move weights opposite way
        - Population explores different directions -> robust gradient estimate
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ES: Computing gradient from fitness scores")
        logger.info(f"{'='*80}")

        # Step 1: Normalize fitness scores
        fitness_mean = fitness_scores.mean()
        fitness_std = fitness_scores.std()

        if fitness_std < 1e-8:
            fitness_std = 1.0
            logger.warning("Fitness std is near zero, using default normalization")

        normalized_fitness = (fitness_scores - fitness_mean) / (fitness_std + 1e-8)

        logger.info(f"Fitness stats: mean={fitness_mean:.4f}, std={fitness_std:.4f}")
        logger.info(f"Fitness range after normalization: [{normalized_fitness.min():.4f}, {normalized_fitness.max():.4f}]")

        # Step 2: Accumulate ES gradient
        sigma = self.config.es.sigma
        lr = self.config.es.learning_rate

        gradient_accumulators = {}

        for idx, adapter_info in enumerate(adapters):
            # Recreate noise deterministically using stored seed
            seed = adapter_info.seed
            rng = np.random.RandomState(seed)

            if isinstance(self.base_model, PeftModel):
                base_adapter_name = self.base_model.active_adapters[0] if hasattr(self.base_model, 'active_adapters') else 'default'
            else:
                base_adapter_name = 'default'

            # Recreate noise for this adapter
            for name, module in self.base_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    base_a = module.lora_A[base_adapter_name].weight.data
                    base_b = module.lora_B[base_adapter_name].weight.data

                    # Recreate exact same noise using seed
                    noise_a = torch.tensor(
                        rng.normal(0, sigma, size=base_a.shape),
                        dtype=base_a.dtype,
                        device=base_a.device
                    )
                    noise_b = torch.tensor(
                        rng.normal(0, sigma, size=base_b.shape),
                        dtype=base_b.dtype,
                        device=base_b.device
                    )

                    # Accumulate weighted noise: (fitness_i / sigma) * noise_i
                    weight = normalized_fitness[idx] / sigma

                    if name not in gradient_accumulators:
                        gradient_accumulators[name] = {'a': None, 'b': None}

                    if gradient_accumulators[name]['a'] is None:
                        gradient_accumulators[name]['a'] = weight * noise_a
                        gradient_accumulators[name]['b'] = weight * noise_b
                    else:
                        gradient_accumulators[name]['a'] += weight * noise_a
                        gradient_accumulators[name]['b'] += weight * noise_b

        # Average gradient across population
        for name in gradient_accumulators:
            gradient_accumulators[name]['a'] /= len(adapters)
            gradient_accumulators[name]['b'] /= len(adapters)

        logger.info(f"Gradient accumulated over {len(adapters)} adapters")

        # Step 3: Update base model weights
        logger.info(f"Updating base LoRA weights with learning_rate={lr}")

        for name, module in self.base_model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if name in gradient_accumulators:
                    gradient_a = gradient_accumulators[name]['a']
                    gradient_b = gradient_accumulators[name]['b']

                    # Update: W_new = W_old + lr * gradient
                    module.lora_A[base_adapter_name].weight.data.add_(gradient_a, alpha=lr)
                    module.lora_B[base_adapter_name].weight.data.add_(gradient_b, alpha=lr)

        logger.info("Base LoRA weights updated")

        # Clean up adapter cache to save space
        for adapter_info in adapters:
            try:
                shutil.rmtree(adapter_info.path)
            except Exception as e:
                logger.debug(f"Could not clean adapter cache: {e}")

    def train(self) -> Dict[str, Any]:
        """
        Main ES training loop orchestrating all components.

        Algorithm Flow:
        ===============
        for iteration in range(num_iterations):
          1. Generate population of perturbed adapters
          2. Evaluate all adapters on same batch of prompts
          3. Compute ES gradient from fitness scores
          4. Update base LoRA weights
          5. Log metrics and save checkpoints

        This implements the complete ES algorithm end-to-end.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING EVOLUTION STRATEGIES TRAINING")
        logger.info("=" * 80)

        # Setup components (mirrors the pattern used by other trainers, e.g.
        # TRLGRPOTrainer.train()): callers should be able to invoke
        # trainer.train() directly without manually calling setup_* first.
        self.setup_model()
        self.setup_data()
        self.setup_reward_function()
        self.setup_rollout_backend()

        num_iterations = self.config.es.num_iterations
        save_freq = self.config.es.save_freq
        log_freq = self.config.es.log_freq

        training_state = {
            'best_fitness': float('-inf'),
            'best_fitness_history': [],
            'mean_fitness_history': [],
            'iteration': 0
        }

        # Store on self for external access
        self.fitness_history = []

        for iteration in tqdm(range(num_iterations), desc="ES Training", unit="iter"):
            # 1. Generate population of perturbed adapters
            adapters = self.generate_local_adapters(iteration)

            # 2. Evaluate all adapters
            fitness_scores = self.generate_and_score(adapters)

            # Track best fitness
            best_fitness = fitness_scores.max()
            mean_fitness = fitness_scores.mean()

            training_state['best_fitness'] = max(training_state['best_fitness'], best_fitness)
            training_state['best_fitness_history'].append(best_fitness)
            training_state['mean_fitness_history'].append(mean_fitness)
            training_state['iteration'] = iteration

            # Store fitness history on self
            self.fitness_history.append(mean_fitness)

            # Update progress bar with metrics
            tqdm.write(f"Iteration {iteration+1}: best={best_fitness:.4f}, mean={mean_fitness:.4f}, overall_best={training_state['best_fitness']:.4f}")

            # 3. Compute gradient and update weights
            self.apply_es_update(adapters, fitness_scores)

            # Save checkpoint every save_freq iterations
            if (iteration + 1) % save_freq == 0:
                self.save_checkpoint(f"iteration_{iteration+1}")
                logger.info(f"Checkpoint saved at iteration {iteration+1}")

            # Call callbacks if provided
            if self.callbacks:
                for callback in self.callbacks:
                    if hasattr(callback, 'on_step_end'):
                        callback.on_step_end(training_state)

        logger.info("\n" + "=" * 80)
        logger.info("EVOLUTION STRATEGIES TRAINING COMPLETED")
        logger.info(f"Best fitness achieved: {training_state['best_fitness']:.4f}")
        logger.info("=" * 80)

        return training_state

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        Optional evaluation on a separate test set.

        This could evaluate the final model on a holdout test set
        using the same reward function used during training.
        """
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating final model...")
        logger.info("=" * 80)

        return {
            'eval_fitness': 0.0,
        }
