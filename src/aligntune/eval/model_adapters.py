"""
Model adapter abstraction for unified text generation across backends.

Provides a minimal interface for generating text with any backend (HF, vLLM, GGUF, Ollama).
This abstraction allows AlignmentAuditor and other eval tools to work with different
model formats without tight coupling.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    Abstract base class for text generation over any backend.

    Provides a minimal, backend-agnostic interface for prompting models.
    Implementations can wrap HF models, vLLM servers, GGUF files, Ollama, etc.
    """

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate text completions for a batch of prompts.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens to generate per prompt.
            temperature: Sampling temperature (0.0 = deterministic, > 0.0 = stochastic).
            **kwargs: Backend-specific generation parameters.

        Returns:
            List of generated text completions (same length as prompts).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up resources (GPU memory, connections, etc.).

        Safe to call multiple times or not at all.
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """
        Return the name of the backend (e.g., "huggingface", "vllm", "gguf").

        Used for logging and debugging.
        """
        pass


class HFModelAdapter(ModelAdapter):
    """
    Wraps a HuggingFace model + tokenizer.

    Used by existing code (e.g., AlignmentAuditor) that passes raw HF models.
    Handles generation with standard HF model.generate() API.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
    ):
        """
        Initialize HF model adapter.

        Args:
            model: HuggingFace model instance.
            tokenizer: HuggingFace tokenizer instance.
            device: Device to run model on ("cuda", "cpu", etc.).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Ensure model is on the right device
        if hasattr(model, "to"):
            self.model = model.to(device)

        logger.debug(
            f"HFModelAdapter initialized: model={type(model).__name__}, "
            f"device={device}"
        )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate text using HF model.generate().

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Passed to model.generate() (e.g., top_p, top_k).

        Returns:
            List of generated completions.
        """
        if not prompts:
            return []

        self.model.eval()
        completions = []

        with torch.no_grad():
            for prompt in prompts:
                try:
                    # Tokenize prompt
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                    ).to(self.device)

                    # Generate
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0.0 else 1.0,
                        do_sample=temperature > 0.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )

                    # Decode
                    full_text = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    # Remove prompt from output
                    completion = full_text[len(prompt):].strip()
                    completions.append(completion)

                except Exception as e:
                    logger.warning(
                        f"HFModelAdapter.generate() failed for prompt "
                        f"(len={len(prompt)}): {e}"
                    )
                    completions.append("")

        return completions

    def close(self) -> None:
        """
        Clean up HF model resources.

        Moves model to CPU and clears GPU cache if applicable.
        """
        if hasattr(self.model, "to"):
            self.model.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("HFModelAdapter closed")

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "huggingface"


class VLLMModelAdapter(ModelAdapter):
    """
    Wraps vLLM for fast batch inference.

    Supports any model path vLLM understands:
    - HF model directories
    - GGUF files (via repo:quant syntax, e.g., "path/to/repo:Q4_K_M")
    - GPTQ/AWQ artifacts

    Lazy-imports vllm to keep it optional.
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """
        Initialize vLLM adapter.

        Args:
            model_path: Path to model (HF dir, GGUF file with :quant suffix, etc.)
            dtype: Precision ("auto", "float16", "bfloat16", etc.)
            gpu_memory_utilization: GPU memory fraction to use.
            **kwargs: Additional arguments passed to vllm.LLM().

        Raises:
            ImportError: If vllm is not installed.
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vllm is not installed. Install with:\n"
                "  pip install vllm\n"
                "For GGUF support, also install:\n"
                "  pip install vllm[gguf]"
            )

        self.model_path = model_path
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization

        try:
            self.llm = LLM(
                model=model_path,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                **kwargs
            )
            logger.debug(
                f"VLLMModelAdapter initialized: model={model_path}, "
                f"dtype={dtype}, gpu_mem={gpu_memory_utilization}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM with model {model_path}: {e}")
            raise

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate text using vLLM batch inference.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Passed to llm.generate() (e.g., top_p, top_k).

        Returns:
            List of generated completions (same length as prompts).
        """
        if not prompts:
            return []

        try:
            from vllm import SamplingParams

            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature if temperature > 0.0 else 1.0,
                **kwargs
            )

            # Batch generate
            outputs = self.llm.generate(prompts, sampling_params)

            # Extract completions
            completions = [output.outputs[0].text for output in outputs]
            return completions

        except Exception as e:
            logger.warning(f"VLLMModelAdapter.generate() failed: {e}")
            return [""] * len(prompts)

    def close(self) -> None:
        """
        Clean up vLLM resources.

        Safe to call multiple times.
        """
        if hasattr(self, "llm") and self.llm is not None:
            try:
                if hasattr(self.llm, "shutdown"):
                    self.llm.shutdown()
                logger.debug("VLLMModelAdapter closed")
            except Exception as e:
                logger.warning(f"Error closing vLLM: {e}")
            finally:
                self.llm = None

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "vllm"


class GGUFModelAdapter(ModelAdapter):
    """
    Wraps llama-cpp-python for local GGUF inference.

    Used when vLLM is unavailable or for local CPU inference.
    Lazy-imports llama_cpp to keep the dependency optional.

    Raises ImportError with helpful instructions if llama-cpp-python is not installed.
    """

    def __init__(
        self,
        gguf_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        **kwargs
    ):
        """
        Initialize GGUF adapter via llama-cpp-python.

        Args:
            gguf_path: Path to .gguf file.
            n_ctx: Context window size.
            n_gpu_layers: Number of layers to offload to GPU (-1 = all).
            **kwargs: Additional arguments passed to llama_cpp.Llama().

        Raises:
            ImportError: If llama-cpp-python is not installed.
            FileNotFoundError: If gguf_path does not exist.
        """
        try:
            import llama_cpp
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. Install with:\n"
                "  pip install llama-cpp-python\n"
                "For GPU acceleration:\n"
                "  CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python"
            )

        from pathlib import Path

        gguf_path_obj = Path(gguf_path)
        if not gguf_path_obj.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        self.gguf_path = gguf_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

        try:
            self.llm = llama_cpp.Llama(
                model_path=gguf_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                **kwargs
            )
            logger.debug(
                f"GGUFModelAdapter initialized: path={gguf_path}, "
                f"n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize llama-cpp with {gguf_path}: {e}")
            raise

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate text using llama-cpp-python.

        Note: llama-cpp-python does not support batch generation,
        so we generate sequentially for each prompt.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Passed to llm.create_completion() (e.g., top_p).

        Returns:
            List of generated completions (same length as prompts).
        """
        if not prompts:
            return []

        completions = []

        for prompt in prompts:
            try:
                output = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0.0 else 1.0,
                    **kwargs
                )
                # Extract generated text from response
                completion = output["choices"][0]["text"]
                completions.append(completion)

            except Exception as e:
                logger.warning(f"GGUFModelAdapter.generate() failed for prompt: {e}")
                completions.append("")

        return completions

    def close(self) -> None:
        """
        Clean up GGUF resources.

        Safe to call multiple times.
        """
        if hasattr(self, "llm") and self.llm is not None:
            try:
                # llama_cpp.Llama doesn't have explicit shutdown,
                # but we can release the resource
                del self.llm
                logger.debug("GGUFModelAdapter closed")
            except Exception as e:
                logger.warning(f"Error closing llama-cpp: {e}")
            finally:
                self.llm = None

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "gguf"


class OllamaModelAdapter(ModelAdapter):
    """
    Wraps Ollama for local or remote inference.

    Connects to an Ollama server endpoint for text generation.
    Lazy-imports ollama client to keep the dependency optional.
    """

    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize Ollama adapter.

        Args:
            model_name: Model name (e.g., "llama2", "mistral", "neural-chat").
            base_url: Ollama API base URL.
            **kwargs: Additional arguments (reserved for future use).

        Raises:
            ImportError: If ollama client is not installed.
        """
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama is not installed. Install with:\n"
                "  pip install ollama\n"
                "Also ensure Ollama server is running:\n"
                "  ollama serve"
            )

        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

        logger.debug(
            f"OllamaModelAdapter initialized: model={model_name}, "
            f"base_url={base_url}"
        )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate text using Ollama.

        Note: Ollama client does not natively support batch generation,
        so we generate sequentially for each prompt.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens to generate (passed as num_predict).
            temperature: Sampling temperature.
            **kwargs: Additional arguments passed to ollama.generate().

        Returns:
            List of generated completions (same length as prompts).
        """
        if not prompts:
            return []

        completions = []

        for prompt in prompts:
            try:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    num_predict=max_new_tokens,
                    options={
                        "temperature": temperature if temperature > 0.0 else 1.0,
                    },
                    stream=False,
                    **kwargs
                )
                # Extract generated text from response
                completion = response.get("response", "")
                completions.append(completion)

            except Exception as e:
                logger.warning(
                    f"OllamaModelAdapter.generate() failed for prompt "
                    f"(len={len(prompt)}): {e}"
                )
                completions.append("")

        return completions

    def close(self) -> None:
        """
        Close Ollama connection.

        Ollama client is stateless (HTTP-based), so this is mostly a no-op.
        Safe to call multiple times.
        """
        logger.debug("OllamaModelAdapter closed (no-op)")

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "ollama"


def build_adapter(
    artifact: "Any",
    **kwargs
) -> ModelAdapter:
    """
    Route an ExportedArtifact to the correct ModelAdapter implementation.

    Provides a fallback chain for GGUF artifacts:
    1. Try vLLM (fast, batch-capable)
    2. Fall back to llama-cpp-python if vLLM unavailable

    Args:
        artifact: ExportedArtifact with .format and .path attributes.
                 Formats: "hf" (fp16), "hf_4bit", "hf_8bit", "gguf", "ollama"
        **kwargs: Format-specific arguments (e.g., dtype, n_ctx, base_url)

    Returns:
        ModelAdapter instance for the artifact type.

    Raises:
        ValueError: If artifact format is unknown.
        ImportError: If required dependencies are missing and no fallback works.
        FileNotFoundError: If artifact path doesn't exist.
    """
    format_type = getattr(artifact, "format", None)
    path = getattr(artifact, "path", None)

    if not format_type or not path:
        raise ValueError(
            "artifact must have .format and .path attributes. "
            f"Got artifact={artifact}"
        )

    logger.debug(f"build_adapter: format={format_type}, path={path}")

    # HuggingFace models (fp16, int4, int8)
    if format_type == "hf":
        # Load HF checkpoint
        from transformers import AutoTokenizer, AutoModelForCausalLM

        try:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            return HFModelAdapter(model, tokenizer, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load HF model from {path}: {e}")
            raise

    elif format_type == "hf_4bit":
        # Load HF 4-bit quantized (bitsandbytes)
        from transformers import AutoTokenizer, AutoModelForCausalLM

        try:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                load_in_4bit=True,
                device_map="auto",
                bnb_4bit_compute_dtype="float16",
            )
            return HFModelAdapter(model, tokenizer, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load 4-bit HF model from {path}: {e}")
            raise

    elif format_type == "hf_8bit":
        # Load HF 8-bit quantized (bitsandbytes)
        from transformers import AutoTokenizer, AutoModelForCausalLM

        try:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                load_in_8bit=True,
                device_map="auto",
            )
            return HFModelAdapter(model, tokenizer, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load 8-bit HF model from {path}: {e}")
            raise

    # GGUF format: try vLLM first, then llama-cpp-python
    elif format_type == "gguf":
        try:
            # Fast path: vLLM
            logger.debug(f"Attempting vLLM for GGUF: {path}")
            return VLLMModelAdapter(path, **kwargs)
        except ImportError:
            logger.debug("vLLM not available, falling back to llama-cpp-python")
            try:
                # Fallback: llama-cpp-python
                return GGUFModelAdapter(path, **kwargs)
            except ImportError as e:
                raise ImportError(
                    f"Neither vLLM nor llama-cpp-python available for GGUF. "
                    f"Install one: pip install vllm  OR  pip install llama-cpp-python. "
                    f"Original error: {e}"
                )

    # Ollama
    elif format_type == "ollama":
        model_name = kwargs.pop("model_name", path)
        base_url = kwargs.pop("base_url", "http://localhost:11434")
        return OllamaModelAdapter(
            model_name=model_name,
            base_url=base_url,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown artifact format: {format_type}. "
            f"Supported: hf, hf_4bit, hf_8bit, gguf, ollama"
        )
