"""
vLLM rollout backend for high-throughput generation during RL training.

vLLM provides 5-10x faster generation compared to standard transformers through:
- Continuous batching
- PagedAttention for KV cache management
- Kernel optimizations
- GPU memory efficiency

vLLM Rollout Backend
"""

import logging
import os
from typing import List, Optional, Dict, Any
import torch

from .base import BaseRolloutBackend

logger = logging.getLogger(__name__)


class VLLMRolloutBackend(BaseRolloutBackend):
    """
    vLLM-based rollout backend for high-performance generation.

    Features:
    - Continuous batching for variable-length inputs
    - PagedAttention for efficient KV cache management
    - 5-10x faster generation vs standard transformers
    - Support for tensor parallelism across GPUs
    - LoRA adapter support
    """

    def __init__(
        self,
        model_name_or_path: str,
        gpu_memory_utilization: float = 0.7,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        lora_path: Optional[str] = None,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        trust_remote_code: bool = True,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize vLLM rollout backend.

        Args:
            model_name_or_path: Model name or path
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Model data type ("auto", "float32", "float16", "bfloat16")
            max_model_len: Maximum model sequence length
            lora_path: Optional path to a LoRA adapter loaded once at engine
                startup and used for every generate() call.
            enable_lora: Enable the vLLM LoRA engine even when `lora_path` is
                not set at construction time. Needed for callers (e.g. the ES
                trainer) that swap in a different adapter path per generate()
                call via the `lora_adapter_path` argument. NOTE: previously
                this flag (and `max_lora_rank` below) was accepted only via
                **kwargs and silently discarded - the vLLM engine was never
                actually started with LoRA support, so any adapter passed to
                generate() was ignored and every call ran the frozen base
                model. Declaring them as explicit constructor params fixes
                that.
            max_lora_rank: Max LoRA rank the vLLM engine should allocate for.
            trust_remote_code: Trust remote code for model loading
            seed: Random seed for sampling
            **kwargs: Additional configuration
        """
        super().__init__(model_name_or_path, **kwargs)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.lora_path = lora_path
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None

    def initialize(self) -> None:
        """Initialize vLLM engine and load model."""
        logger.info(f"Initializing vLLM backend for {self.model_name_or_path}")

        # Set env vars BEFORE importing vLLM
        os.environ["VLLM_LOG_LEVEL"] = "WARNING"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )

        # Load tokenizer for chat template formatting
        logger.info("Loading tokenizer for chat template support")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code
        )

        # Map dtype string to actual type. Accepts both vLLM's own spelling
        # ("float16"/"bfloat16") and aligntune's PrecisionHandler spelling
        # ("fp16"/"bf16") since the ES trainer forwards a single `dtype`
        # value (config.model.dtype) to both PrecisionHandler (for the HF
        # model) and this backend - without this normalization, a caller
        # using the PrecisionHandler convention (e.g. dtype="fp16") would
        # silently fall through to "auto" here.
        dtype_normalized = (self.dtype or "auto").lower()
        if dtype_normalized in ("auto",):
            dtype = "auto"
        elif dtype_normalized in ("float16", "fp16", "half"):
            dtype = "float16"
        elif dtype_normalized in ("bfloat16", "bf16"):
            dtype = "bfloat16"
        else:
            dtype = "auto"

        # Initialize vLLM LLM engine
        engine_kwargs = {
            "model": self.model_name_or_path,
            "dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "seed": self.seed,
            "disable_log_stats": True,  # Fix Colab OutStream compatibility
        }

        if self.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.max_model_len

        # Handle LoRA adapter support. Enabled either because a fixed
        # `lora_path` was given at construction, or because the caller
        # explicitly requested `enable_lora=True` (e.g. the ES trainer, which
        # swaps in a different adapter directory per generate() call via
        # `lora_adapter_path` rather than a single fixed path).
        if self.lora_path or self.enable_lora:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_loras"] = 1
            engine_kwargs["max_lora_rank"] = self.max_lora_rank
            logger.info(f"LoRA support enabled (max_lora_rank={self.max_lora_rank}).")

        self.llm = LLM(**engine_kwargs)

        # Default sampling params (will be overridden per generation call)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            top_k=-1,
            max_tokens=128,
            seed=self.seed,
        )

        self.initialized = True
        logger.info("vLLM backend initialized successfully")

    def format_prompt_with_chat_template(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format prompt using model's chat template.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Returns:
            Formatted prompt string
        """
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            # No chat template, use simple format
            if system_prompt:
                return f"{system_prompt}\n\n{prompt}"
            return prompt

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        num_return_sequences: int = 1,
        lora_adapter_path: Optional[str] = None,
        **kwargs
    ) -> List[List[str]]:
        """
        Generate completions using vLLM.

        vLLM's continuous batching automatically schedules all prompts
        for efficient parallel processing, providing significant speedup.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Sequences to generate per prompt
            lora_adapter_path: Optional path to a LoRA adapter directory to
                load for THIS call only (e.g. a per-population-member
                adapter from the ES trainer). Takes precedence over the
                fixed `lora_path` set at construction. Requires the engine
                to have been initialized with `enable_lora=True`. NOTE: this
                param used to be swallowed by **kwargs and silently ignored,
                so callers that swap adapters per-call (like the ES trainer)
                always got generations from the frozen base model.
            **kwargs: Additional generation parameters

        Returns:
            List of lists of generated text. Shape: [len(prompts), num_return_sequences]
        """
        if not self.initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        if not prompts:
            return []

        try:
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError:
            raise ImportError("vLLM imports failed")

        # Create sampling parameters for this generation call
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
            seed=self.seed,
        )

        # Create LoRA request if an adapter is available for this call.
        # A per-call `lora_adapter_path` (e.g. from the ES trainer) takes
        # precedence over the fixed `self.lora_path`. We reuse a single
        # `lora_int_id` and pass `load_inplace=True` so each call can swap in
        # a different adapter's weights under the same engine LoRA slot.
        adapter_path = lora_adapter_path or self.lora_path
        lora_request = None
        if adapter_path:
            lora_request = LoRARequest(
                lora_name="lora_adapter",
                lora_int_id=1,
                lora_path=adapter_path,
                load_inplace=True,
            )

        # Generate using vLLM's batch generation (continuous batching)
        logger.debug(
            f"vLLM batch generating {len(prompts)} prompts x "
            f"{num_return_sequences} samples = "
            f"{len(prompts) * num_return_sequences} total generations"
        )

        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request
        )

        # Extract generated text from outputs
        results = []
        for output in outputs:
            prompt_completions = []
            for completion in output.outputs:
                prompt_completions.append(completion.text)
            results.append(prompt_completions)

        return results

    def cleanup(self) -> None:
        """Release vLLM engine and resources."""
        logger.info("Cleaning up vLLM rollout backend")

        try:
            import gc

            if self.llm is not None:
                # Shutdown vLLM engine gracefully
                try:
                    if hasattr(self.llm, "engine"):
                        if hasattr(self.llm.engine, "workers"):
                            logger.debug("Shutting down vLLM engine workers...")
                            for worker in self.llm.engine.workers:
                                if hasattr(worker, "shutdown"):
                                    try:
                                        worker.shutdown()
                                    except Exception as e:
                                        logger.debug(f"Worker shutdown warning: {e}")
                except Exception as e:
                    logger.debug(f"Engine shutdown warning: {e}")

                del self.llm

            # Aggressive garbage collection
            for _ in range(3):
                gc.collect()

            # Clear GPU memory
            if torch.cuda.is_available():
                logger.debug("Clearing CUDA memory...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Reset memory stats for clean tracking
                torch.cuda.reset_peak_memory_stats()

                # Clear cache on all available devices
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()

            logger.info("vLLM backend cleanup completed")

        except Exception as e:
            logger.warning(f"Warning during vLLM cleanup: {e}")

        self.initialized = False
