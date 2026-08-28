"""
TextToLoRAHypernet: Generates LoRA weights from task description embeddings.

The hypernetwork learns to map task embeddings (from a sentence transformer)
to LoRA adapter weights. This enables efficient task-specific fine-tuning without
explicit LoRA training per task.

Architecture:
    embedding (hidden_dim) → Dense(1024) → ReLU → Dense(num_targets × rank × hidden_dim)
                            → Reshape → [{"A": tensor, "B": tensor}, ...]

The network generates both A and B matrices for LoRA:
    - A matrices initialized from N(0, small_std) for initial stochasticity
    - B matrices initialized to zeros (standard LoRA initialization)
    - Each module pair can target different layer groups in the base model

Initialization follows LoRA conventions to ensure stable training dynamics.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class TextToLoRAHypernet(nn.Module):
    """
    Hypernetwork for generating LoRA A/B matrices from text embeddings.

    This module transforms task description embeddings into LoRA adapter weights.
    It enables meta-learning of task-specific adaptations without explicit per-task training.

    Args:
        hidden_dim (int): Embedding dimension (e.g., 768 for sentence-transformers).
            Must match the dimensionality of embeddings from the embedding model.
        lora_r (int): Rank of generated LoRA adapters (e.g., 8, 16, 32).
            Controls the expressiveness of generated adapters. Typical values: 8-32.
        num_target_modules (int): Number of different target modules for which to generate
            LoRA weights. Each module gets an independent A/B pair. Default: 5.
        mlp_hidden (int): Hidden dimension for the internal MLP. Default: 512.
            Larger values increase model capacity but increase parameters.
        embedding_model_name (str): Name of the sentence-transformer model to use for
            embedding task descriptions. Default: "all-MiniLM-L6-v2".
        lora_init_std (float): Standard deviation for A matrix initialization.
            B matrices are initialized to zero. Default: 0.02 (small value for stable training).
        device (str): Device for model computation ("cpu" or "cuda"). Default: "cpu".

    Attributes:
        hypernet (nn.Module): The MLP that generates LoRA weights
        lora_r (int): Rank of LoRA adapters
        hidden_dim (int): Embedding dimension
        num_target_modules (int): Number of target modules
        lora_init_std (float): Initialization scale for A matrices
        device (str): Computation device

    Example:
        >>> hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16, num_target_modules=5)
        >>> embeddings = torch.randn(2, 768)  # batch_size=2, hidden_dim=768
        >>> lora_weights = hypernet(embeddings)
        >>> len(lora_weights)
        5
        >>> lora_weights[0]["A"].shape
        torch.Size([2, 16, 768])
        >>> lora_weights[0]["B"].shape
        torch.Size([2, 768, 16])
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        lora_r: int = 16,
        num_target_modules: int = 5,
        mlp_hidden: int = 512,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        lora_init_std: float = 0.02,
        device: str = "cpu",
    ):
        """
        Initialize the Text-to-LoRA hypernetwork.

        Args:
            hidden_dim: Embedding dimension from sentence-transformer
            lora_r: Rank of LoRA adapters to generate
            num_target_modules: Number of independent LoRA pairs to generate
            mlp_hidden: Hidden dimension for MLP layers
            embedding_model_name: Sentence-transformer model identifier
            lora_init_std: Std dev for A matrix initialization (B starts at 0)
            device: "cpu" or "cuda"

        Raises:
            ValueError: If hidden_dim, lora_r, or num_target_modules <= 0
            ValueError: If mlp_hidden <= 0 or lora_init_std <= 0
        """
        super().__init__()

        # Validate parameters
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if lora_r <= 0:
            raise ValueError(f"lora_r must be positive, got {lora_r}")
        if num_target_modules <= 0:
            raise ValueError(f"num_target_modules must be positive, got {num_target_modules}")
        if mlp_hidden <= 0:
            raise ValueError(f"mlp_hidden must be positive, got {mlp_hidden}")
        if lora_init_std <= 0:
            raise ValueError(f"lora_init_std must be positive, got {lora_init_std}")

        self.hidden_dim = hidden_dim
        self.lora_r = lora_r
        self.num_target_modules = num_target_modules
        self.mlp_hidden = mlp_hidden
        self.embedding_model_name = embedding_model_name
        self.lora_init_std = lora_init_std
        self.device = device

        # Output dimension: num_targets × (rank × hidden_dim + hidden_dim × rank)
        # = num_targets × (2 × rank × hidden_dim) for A and B matrices
        output_dim = num_target_modules * 2 * lora_r * hidden_dim

        # MLP: embedding → dense → ReLU → dense → output
        self.hypernet = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, output_dim),
        )

        # Initialize hypernetwork weights
        self._init_hypernet()

        self.to(device)

        logger.info(
            f"Initialized TextToLoRAHypernet: hidden_dim={hidden_dim}, "
            f"lora_r={lora_r}, num_targets={num_target_modules}, "
            f"mlp_hidden={mlp_hidden}, device={device}"
        )

    def _init_hypernet(self) -> None:
        """
        Initialize hypernetwork weights with Xavier uniform.

        This ensures stable training dynamics for the meta-learner.
        """
        for module in self.hypernet.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, task_embedding: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate LoRA weights from task embeddings.

        Forward pass:
        1. Take task embedding [batch_size, hidden_dim]
        2. Pass through MLP: [batch_size, output_dim]
        3. Reshape to per-module A/B matrices
        4. Return list of dicts with "A" and "B" keys

        Args:
            task_embedding (Tensor): Task description embeddings of shape
                [batch_size, hidden_dim]. Embeddings should come from the
                embedding model specified in get_embedding_model().

        Returns:
            List[Dict[str, Tensor]]: List of length num_target_modules, where
                each entry is {"A": tensor, "B": tensor}
                - "A": shape [batch_size, lora_r, hidden_dim]
                  - Drawn from N(0, lora_init_std)
                  - Used as left projection in LoRA computation
                - "B": shape [batch_size, hidden_dim, lora_r]
                  - Initialized to zero
                  - Used as right projection in LoRA computation

            The output can be used directly in LoRA forward:
                y = x + alpha/r * (x @ A.T @ B.T)

        Raises:
            ValueError: If input shape is incorrect
            RuntimeError: If tensors are on different devices

        Example:
            >>> hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16)
            >>> embeddings = torch.randn(4, 768)
            >>> lora_list = hypernet(embeddings)
            >>> len(lora_list)  # num_target_modules
            5
            >>> lora_list[0]["A"].shape
            torch.Size([4, 16, 768])
            >>> lora_list[0]["B"].shape
            torch.Size([4, 768, 16])
        """
        # Validate input
        if task_embedding.dim() != 2:
            raise ValueError(
                f"Expected 2D input [batch_size, hidden_dim], "
                f"got shape {task_embedding.shape}"
            )

        batch_size, embedding_dim = task_embedding.shape

        if embedding_dim != self.hidden_dim:
            raise ValueError(
                f"Input embedding dimension {embedding_dim} doesn't match "
                f"hypernet hidden_dim {self.hidden_dim}"
            )

        if task_embedding.device != torch.device(self.device):
            raise RuntimeError(
                f"Input tensor on device {task_embedding.device}, "
                f"but hypernet on {self.device}"
            )

        # Generate raw LoRA weights through MLP
        raw_weights = self.hypernet(task_embedding)  # [batch_size, output_dim]

        # Reshape to per-module LoRA pairs
        # raw_weights: [batch_size, num_targets * 2 * rank * hidden_dim]
        reshaped = raw_weights.reshape(
            batch_size, self.num_target_modules, 2 * self.lora_r * self.hidden_dim
        )  # [batch_size, num_targets, 2*rank*hidden_dim]

        lora_list = []

        for module_idx in range(self.num_target_modules):
            module_weights = reshaped[:, module_idx, :]  # [batch_size, 2*rank*hidden_dim]

            # Split into A and B
            a_raw = module_weights[:, : self.lora_r * self.hidden_dim]
            b_raw = module_weights[:, self.lora_r * self.hidden_dim :]

            # Reshape A: [batch_size, lora_r, hidden_dim]
            a_matrix = a_raw.reshape(batch_size, self.lora_r, self.hidden_dim)

            # Reshape B: [batch_size, hidden_dim, lora_r]
            b_matrix = b_raw.reshape(batch_size, self.hidden_dim, self.lora_r)

            # Scale A matrix by lora_init_std to control initialization magnitude
            a_matrix = a_matrix * self.lora_init_std

            # Zero out B matrix (standard LoRA initialization)
            b_matrix = b_matrix * 0.0

            lora_list.append({"A": a_matrix, "B": b_matrix})

        return lora_list

    def get_embedding_model(self):
        """
        Get or create a sentence-transformer for embedding task descriptions.

        Returns a model that can embed text strings into fixed-size vectors
        of dimension self.hidden_dim.

        This method provides the embedding model to be used with this hypernetwork.
        The returned model should:
        1. Accept a list of strings as input
        2. Return embeddings of shape [num_strings, hidden_dim]
        3. Work on CPU (for this implementation)

        Returns:
            model: A callable that takes List[str] and returns torch.Tensor
                of shape [len(strings), hidden_dim]

        Raises:
            ImportError: If sentence-transformers is not installed

        Note:
            The actual embedding model is NOT loaded in this method.
            In production, you would use:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.embedding_model_name)

            For testing/mocking, return a simple embedding function.

        Example:
            >>> hypernet = TextToLoRAHypernet()
            >>> # In production:
            >>> # embedding_fn = hypernet.get_embedding_model()
            >>> # embeddings = embedding_fn(["task description 1", "task description 2"])
            >>> # embeddings.shape -> [2, 768]
        """
        logger.info(
            f"Creating embedding model: {self.embedding_model_name} "
            f"(hidden_dim={self.hidden_dim})"
        )

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.embedding_model_name)
            return model

        except ImportError:
            raise ImportError(
                "sentence-transformers is required for text embeddings. "
                "Install with: pip install sentence-transformers"
            )

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters in the hypernetwork.

        Returns:
            int: Total parameter count

        Example:
            >>> hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16)
            >>> num_params = hypernet.get_num_parameters()
            >>> num_params > 0
            True
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_lora_config(self) -> Dict[str, any]:
        """
        Get configuration dictionary for the hypernetwork.

        Useful for serialization and reconstruction.

        Returns:
            Dict[str, any]: Configuration with keys:
                - hidden_dim: Embedding dimension
                - lora_r: LoRA rank
                - num_target_modules: Number of target modules
                - mlp_hidden: MLP hidden dimension
                - embedding_model_name: Model identifier
                - lora_init_std: A matrix initialization scale
                - num_parameters: Total trainable parameters

        Example:
            >>> hypernet = TextToLoRAHypernet()
            >>> config = hypernet.get_lora_config()
            >>> config["lora_r"]
            16
        """
        return {
            "hidden_dim": self.hidden_dim,
            "lora_r": self.lora_r,
            "num_target_modules": self.num_target_modules,
            "mlp_hidden": self.mlp_hidden,
            "embedding_model_name": self.embedding_model_name,
            "lora_init_std": self.lora_init_std,
            "num_parameters": self.get_num_parameters(),
        }
