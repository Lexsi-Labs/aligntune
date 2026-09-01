"""
GGUF exporter for model export pipeline.

Supports GGUF export from both TRL and Unsloth checkpoints with optional quantization.
"""

import logging
import subprocess
import tempfile
import urllib.request
import tarfile
from pathlib import Path
from typing import Optional, Union, Literal, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseExporter

logger = logging.getLogger(__name__)

# GGUF quantization presets (keys = AlignTune names, values = llama-quantize types)
GGUF_QUANT_PRESETS = {
    "Q2_K": "q2_k",
    "Q3_K_M": "q3_k_m",
    "Q4_K_S": "q4_k_s",
    "Q4_K_M": "q4_k_m",
    "Q5_K_S": "q5_k_s",
    "Q5_K_M": "q5_k_m",
    "Q6_K": "q6_k",
    "Q8_0": "q8_0",
}


class GGUFExporter(BaseExporter):
    """
    Exporter for GGUF format with support for both TRL and Unsloth backends.

    Supports quantization via llama.cpp's quantize tool.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        converter: Optional[Literal["llama-cpp", "unsloth"]] = None,
        quantization: Optional[str] = None,
    ):
        """
        Initialize GGUF exporter.

        Args:
            output_dir: Base output directory for exported artifacts
            converter: Force specific converter ("llama-cpp" or "unsloth")
            quantization: Quantization preset (Q2_K, Q3_K_M, Q4_K_S, Q4_K_M,
                Q5_K_S, Q5_K_M, Q6_K, Q8_0)
        """
        super().__init__(output_dir)
        self.converter = converter
        self.quantization = quantization

        if quantization and quantization not in GGUF_QUANT_PRESETS:
            logger.warning(
                f"Unknown quantization preset: {quantization}. "
                f"Valid options: {list(GGUF_QUANT_PRESETS.keys())}"
            )

    def _download_llama_cpp_converter(self, temp_dir: Path) -> Path:
        """
        Download llama.cpp converter script from pinned release.

        Args:
            temp_dir: Temporary directory for download

        Returns:
            Path to converter script
        """
        converter_path = temp_dir / "convert_hf_to_gguf.py"

        if converter_path.exists():
            logger.info(f"Using cached converter at {converter_path}")
            return converter_path

        # Pinned llama.cpp release
        release_url = (
            "https://github.com/ggerganov/llama.cpp/releases/download/b3691/"
            "llama-3691-bin-win-x64.zip"
        )
        converter_script_url = (
            "https://raw.githubusercontent.com/ggerganov/llama.cpp/refs/tags/b3691/"
            "convert_hf_to_gguf.py"
        )

        logger.info("Downloading llama.cpp converter script...")
        try:
            urllib.request.urlretrieve(converter_script_url, converter_path)
            logger.info(f"Downloaded converter to {converter_path}")
            return converter_path
        except Exception as e:
            logger.error(f"Failed to download converter: {e}")
            raise

    def _get_llama_quantize(self) -> Optional[Path]:
        """
        Find llama-quantize executable in PATH or download it.

        Returns:
            Path to quantize executable or None if not available
        """
        import shutil

        # Try to find in PATH
        quantize_exe = shutil.which("llama-quantize")
        if quantize_exe:
            logger.info(f"Found llama-quantize at {quantize_exe}")
            return Path(quantize_exe)

        # Try to download with llama.cpp release
        logger.warning("llama-quantize not found in PATH. Quantization may not be available.")
        return None

    def _check_unsloth_available(self) -> bool:
        """Check if Unsloth is installed."""
        try:
            import unsloth
            logger.info(f"Unsloth version {unsloth.__version__} is available")
            return True
        except ImportError:
            logger.warning("Unsloth is not installed. Cannot use Unsloth converter.")
            return False

    def _export_via_unsloth(
        self, model: Any, output_path: Union[str, Path], **kwargs
    ) -> str:
        """
        Export via Unsloth's native GGUF export.

        Args:
            model: Unsloth model object
            output_path: Output directory
            **kwargs: Additional export arguments

        Returns:
            Path to exported GGUF file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting via Unsloth's save_pretrained_gguf...")

        # Unsloth's save_pretrained_gguf handles LoRA merge automatically
        try:
            # Try the Unsloth 2.0+ API
            gguf_path = model.save_pretrained_gguf(
                save_directory=output_path,
                quantization_method=self.quantization or "q4_k_m",
            )
            return str(gguf_path)
        except TypeError:
            # Fallback for older Unsloth versions
            logger.info("Using fallback Unsloth export (older API)")
            gguf_path = output_path / "model.gguf"
            model.save_pretrained_gguf(
                save_directory=output_path,
                quantization_method=self.quantization or "q4_k_m",
            )
            return str(gguf_path)

    def _export_via_llama_cpp(
        self, checkpoint_path: Union[str, Path], output_path: Union[str, Path], **kwargs
    ) -> str:
        """
        Export via llama.cpp's converter script.

        Args:
            checkpoint_path: Path to checkpoint directory
            output_path: Output directory
            **kwargs: Additional export arguments

        Returns:
            Path to exported GGUF file
        """
        checkpoint_path = Path(checkpoint_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting via llama.cpp converter...")

        # Create temp directory for converter script
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            converter_script = self._download_llama_cpp_converter(temp_dir)

            # Get model directory from checkpoint
            model_dir = checkpoint_path / "model"
            if not model_dir.exists():
                model_dir = checkpoint_path

            gguf_path = output_path / "model.gguf"

            # Run converter
            cmd = [
                "python",
                str(converter_script),
                str(model_dir),
                "--outfile",
                str(gguf_path),
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"Converter output: {result.stdout}")
                if result.stderr:
                    logger.warning(f"Converter stderr: {result.stderr}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Converter failed: {e.stderr}")
                raise RuntimeError(f"GGUF conversion failed: {e}")

        # Apply quantization if specified
        if self.quantization:
            gguf_path = self._quantize_gguf(gguf_path)

        return str(gguf_path)

    def _quantize_gguf(self, gguf_path: Union[str, Path]) -> Path:
        """
        Quantize an existing GGUF file.

        Args:
            gguf_path: Path to original GGUF file

        Returns:
            Path to quantized GGUF file
        """
        gguf_path = Path(gguf_path)
        quantize_exe = self._get_llama_quantize()

        if not quantize_exe:
            logger.warning("Quantization requested but llama-quantize not available")
            return gguf_path

        quant_method = GGUF_QUANT_PRESETS.get(self.quantization, self.quantization)
        output_path = gguf_path.parent / f"{gguf_path.stem}_{self.quantization}.gguf"

        cmd = [str(quantize_exe), str(gguf_path), str(output_path), quant_method]

        logger.info(f"Quantizing with {self.quantization}...")
        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Quantization output: {result.stdout}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Quantization failed: {e.stderr}")
            raise RuntimeError(f"GGUF quantization failed: {e}")

    def prepare_model(self, checkpoint_path: Union[str, Path], **kwargs) -> Any:
        """
        Prepare model for GGUF export (load HF model).

        Args:
            checkpoint_path: Path to checkpoint directory
            **kwargs: Additional arguments

        Returns:
            Loaded model
        """
        checkpoint_path = Path(checkpoint_path)
        model_dir = checkpoint_path / "model"
        if not model_dir.exists():
            model_dir = checkpoint_path

        logger.info(f"Loading model from {model_dir}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="cpu",  # GGUF export typically happens on CPU
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def export_model(
        self, model: Any, output_path: Union[str, Path], checkpoint_path: Optional[Union[str, Path]] = None, **kwargs
    ) -> str:
        """
        Export model to GGUF format.

        Args:
            model: Prepared model object
            output_path: Output directory
            checkpoint_path: Original checkpoint path (for llama-cpp fallback)
            **kwargs: Additional export arguments

        Returns:
            Path to exported GGUF file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine which converter to use
        if self.converter == "unsloth":
            if not self._check_unsloth_available():
                raise RuntimeError("Unsloth converter requested but not installed")
            return self._export_via_unsloth(model, output_path, **kwargs)

        elif self.converter == "llama-cpp":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required for llama-cpp converter")
            return self._export_via_llama_cpp(checkpoint_path, output_path, **kwargs)

        # Auto-detect converter based on available tools
        if self._check_unsloth_available():
            logger.info("Auto-detected Unsloth, using Unsloth converter")
            return self._export_via_unsloth(model, output_path, **kwargs)

        if checkpoint_path is None:
            logger.warning("No checkpoint_path for llama-cpp converter, trying Unsloth")
            if self._check_unsloth_available():
                return self._export_via_unsloth(model, output_path, **kwargs)
            raise RuntimeError("No available converters (Unsloth or llama-cpp)")

        logger.info("Auto-detected TRL, using llama.cpp converter")
        return self._export_via_llama_cpp(checkpoint_path, output_path, **kwargs)

    def export(
        self,
        checkpoint_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Execute full GGUF export pipeline.

        Args:
            checkpoint_path: Path to checkpoint directory
            output_path: Output path (optional)
            **kwargs: Format-specific export arguments

        Returns:
            Path to exported GGUF file
        """
        checkpoint_path = Path(checkpoint_path)

        if not self.validate_checkpoint(checkpoint_path):
            raise ValueError(f"Invalid checkpoint: {checkpoint_path}")

        if output_path is None:
            output_path = self.output_dir
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata = self.get_checkpoint_metadata(checkpoint_path)
        logger.info(f"Detected backend: {metadata.get('backend', 'unknown')}")

        # For GGUF, we may use llama-cpp converter which doesn't need loaded model
        # Try direct converter approach first
        if self.converter == "llama-cpp" or (
            not self.converter and metadata.get("backend") == "trl"
        ):
            try:
                return self._export_via_llama_cpp(checkpoint_path, output_path, **kwargs)
            except Exception as e:
                logger.info(f"llama-cpp converter failed: {e}, trying model-based approach")

        # Fall back to model-based export
        model = self.prepare_model(checkpoint_path, **kwargs)
        return self.export_model(model, output_path, checkpoint_path=checkpoint_path, **kwargs)
