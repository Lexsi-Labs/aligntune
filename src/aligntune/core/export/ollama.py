"""
Ollama exporter for model export pipeline.

Creates Ollama Modelfile from GGUF artifacts and optionally loads them into Ollama.
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Union
from .base import BaseExporter

logger = logging.getLogger(__name__)


class OllamaExporter(BaseExporter):
    """
    Exporter for Ollama model format.

    Generates a Modelfile from a GGUF artifact and optionally creates the Ollama model.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        gguf_path: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        create_model: bool = False,
    ):
        """
        Initialize Ollama exporter.

        Args:
            output_dir: Base output directory for Modelfile
            gguf_path: Path to GGUF artifact to wrap
            model_name: Name for Ollama model (e.g., "my-model:latest")
            create_model: Whether to run 'ollama create' to load into Ollama
        """
        super().__init__(output_dir)
        self.gguf_path = gguf_path
        self.model_name = model_name or "custom-model:latest"
        self.create_model = create_model

    def _check_ollama_available(self) -> bool:
        """Check if Ollama CLI is available."""
        ollama_exe = shutil.which("ollama")
        if ollama_exe:
            logger.info(f"Found Ollama at {ollama_exe}")
            return True
        logger.warning("Ollama CLI not found. Model must be imported manually.")
        return False

    def _generate_modelfile(
        self, gguf_path: Union[str, Path], output_path: Union[str, Path]
    ) -> str:
        """
        Generate a Modelfile from GGUF artifact.

        Args:
            gguf_path: Path to GGUF file
            output_path: Output directory for Modelfile

        Returns:
            Path to generated Modelfile
        """
        gguf_path = Path(gguf_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        modelfile_path = output_path / "Modelfile"

        # Generate Modelfile content
        # Use relative path if GGUF is in same directory as Modelfile
        if gguf_path.parent == output_path:
            gguf_ref = f"./{gguf_path.name}"
        else:
            gguf_ref = str(gguf_path)

        modelfile_content = f"""FROM {gguf_ref}

# Model parameters
PARAMETER num_ctx 2048
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt
SYSTEM \"\"\"You are a helpful AI assistant.\"\"\"
"""

        logger.info(f"Writing Modelfile to {modelfile_path}")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        logger.info(f"Generated Modelfile at {modelfile_path}")
        return str(modelfile_path)

    def _create_ollama_model(self, modelfile_path: Union[str, Path]) -> bool:
        """
        Create Ollama model from Modelfile using 'ollama create' command.

        Args:
            modelfile_path: Path to Modelfile

        Returns:
            bool: True if successful
        """
        modelfile_path = Path(modelfile_path)

        if not self._check_ollama_available():
            logger.warning("Ollama not available, skipping model creation")
            return False

        cmd = ["ollama", "create", self.model_name, "-f", str(modelfile_path)]

        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Ollama output: {result.stdout}")
            logger.info(f"Successfully created Ollama model: {self.model_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Ollama model: {e.stderr}")
            logger.info("You can manually import with: ollama create <name> -f Modelfile")
            return False

    def prepare_model(self, checkpoint_path: Union[str, Path], **kwargs) -> Union[str, Path]:
        """
        For Ollama, "prepare_model" means validate GGUF input.

        Args:
            checkpoint_path: Not used for Ollama
            **kwargs: Additional arguments

        Returns:
            Path to GGUF file (from self.gguf_path)
        """
        if self.gguf_path is None:
            raise ValueError("gguf_path must be provided to OllamaExporter")

        gguf_path = Path(self.gguf_path)
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        if not gguf_path.suffix.lower() == ".gguf":
            logger.warning(f"File does not have .gguf extension: {gguf_path}")

        return gguf_path

    def export_model(
        self, gguf_path: Union[str, Path], output_path: Union[str, Path], **kwargs
    ) -> str:
        """
        Export GGUF as Ollama Modelfile.

        Args:
            gguf_path: Path to GGUF file
            output_path: Output directory for Modelfile
            **kwargs: Additional arguments

        Returns:
            Path to Modelfile
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate Modelfile
        modelfile_path = self._generate_modelfile(gguf_path, output_path)

        # Optionally create model in Ollama
        if self.create_model:
            self._create_ollama_model(modelfile_path)

        return modelfile_path

    def export(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        gguf_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Execute full Ollama export pipeline.

        Args:
            checkpoint_path: Not used (GGUF path is primary input)
            output_path: Output path for Modelfile
            gguf_path: Path to GGUF file (overrides self.gguf_path)
            **kwargs: Additional arguments

        Returns:
            Path to Modelfile
        """
        # Use provided gguf_path or fall back to instance variable
        if gguf_path:
            self.gguf_path = gguf_path

        if self.gguf_path is None:
            raise ValueError("gguf_path must be provided")

        if output_path is None:
            output_path = self.output_dir
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating Ollama model from {self.gguf_path}")

        # Prepare (validate GGUF)
        gguf_path = self.prepare_model(None)

        # Export (generate Modelfile and optionally create)
        modelfile_path = self.export_model(gguf_path, output_path, **kwargs)

        logger.info(f"Ollama export completed: {modelfile_path}")
        return modelfile_path
