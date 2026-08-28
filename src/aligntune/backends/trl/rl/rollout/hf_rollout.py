"""
HuggingFace Transformers rollout backend for generation during RL training.

Uses standard transformers.AutoModelForCausalLM.generate() for compatibility.

vLLM Rollout Backend
"""

import logging
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseRolloutBackend

logger = logging.getLogger(__name__)


class HFRolloutBackend(BaseRolloutBackend):
    """
    HuggingFace Transformers-based rollout backend.

    Uses standard transformers.generate() for inference.
    This is the baseline backend for compatibility with all models.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        **kwargs
    ):
        """
        Initialize HF rollout backend.

        Args:
            model_name_or_path: Model name or path
            tokenizer_name: Tokenizer name (defaults to model_name_or_path)
            device: Device placement ("auto", "cuda", "cpu")
            dtype: Data type ("auto", "float32", "float16", "bfloat16")
            **kwargs: Additional configuration
        """
        super().__init__(model_name_or_path, **kwargs)
        self.tokenizer_name = tokenizer_name or model_name_or_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def initialize(self) -> None:
        """Load HF model and tokenizer."""
        logger.info(f"Loading HF model from {self.model_name_or_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=self.config.get("trust_remote_code", True)
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.get("trust_remote_code", True),
        }

        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.device

        if self.dtype == "auto":
            model_kwargs["torch_dtype"] = torch.float32
        elif self.dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs
        )

        # Enable gradient checkpointing if requested
        if self.config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        self.model.eval()
        self.initialized = True
        logger.info("HF backend initialized successfully")

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
        Generate completions using HF transformers.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Sequences to generate per prompt
            **kwargs: Additional generation parameters

        Returns:
            List of lists of generated text
        """
        if not self.initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        if not prompts:
            return []

        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_prompt_length", 512)
        )

        # Move to device
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with specified parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                top_k=top_k if top_k > 0 else -1,
                num_return_sequences=num_return_sequences,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        # Extract only the generated parts (remove prompts)
        prompt_lengths = [len(self.tokenizer.encode(p)) for p in prompts]
        results = []

        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
            # Find where generated text starts by looking for prompt content
            if generated.startswith(prompt):
                completion = generated[len(prompt):]
            else:
                # Fallback: use the entire generated text
                completion = generated

            results.append(completion)

        # Reshape results into [num_prompts, num_return_sequences]
        final_results = []
        for i in range(len(prompts)):
            start_idx = i * num_return_sequences
            end_idx = start_idx + num_return_sequences
            final_results.append(results[start_idx:end_idx])

        return final_results

    def cleanup(self) -> None:
        """Release HF model resources."""
        logger.info("Cleaning up HF rollout backend")
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.initialized = False
        logger.info("HF backend cleanup completed")
