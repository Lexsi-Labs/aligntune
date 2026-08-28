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
            lora_path: Optional path to LoRA adapter
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
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.llm = None
        self.sampling_params = None

    def initialize(self) -> None:
        """Initialize vLLM engine and load model."""
        logger.info(f"Initializing vLLM backend for {self.model_name_or_path}")

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )

        # Suppress vLLM verbose output
        os.environ["VLLM_LOG_LEVEL"] = "WARNING"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Map dtype string to actual type
        if self.dtype == "auto":
            dtype = "auto"
        elif self.dtype == "float16":
            dtype = "float16"
        elif self.dtype == "bfloat16":
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
        }

        if self.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.max_model_len

        # Handle LoRA adapter if provided
        if self.lora_path:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_loras"] = 1
            engine_kwargs["max_lora_rank"] = 64
            logger.info(f"LoRA support enabled. Adapter: {self.lora_path}")

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

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        num_return_sequences: int = 1,
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

        # Create LoRA request if adapter is available
        lora_request = None
        if self.lora_path:
            lora_request = LoRARequest("lora_adapter", 1, self.lora_path)

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
