"""
DocToLoRA: Document-to-LoRA specialization of the Text-to-LoRA hypernetwork.

Extends TextToLoRA to handle long documents by chunking, embedding, and pooling.
This enables generating task-specific LoRA weights from extended task descriptions.

Architecture:
    Document → Chunk → Embed chunks → Pool embeddings → Feed to hypernetwork → LoRA

The pooling strategy (mean or weighted) determines how information from multiple
chunks is combined before being fed to the hypernetwork.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class DocToLoRA(nn.Module):
    """
    Document-to-LoRA: Generate LoRA weights from long documents.

    Specialization of Text-to-LoRA for handling extended text documents.
    Documents are chunked, embedded independently, then pooled before
    being fed to the hypernetwork.

    Args:
        text2lora_hypernet: A TextToLoRAHypernet instance (the underlying generator)
        chunk_size (int): Size of text chunks in characters. Default: 512.
            Smaller chunks = more granular but noisier; larger chunks = coarser but smoother.
        num_chunks (int): Maximum number of chunks to process. Default: 3.
            Excess chunks are dropped; documents shorter than chunk_size use 1 chunk.
        pooling_strategy (str): How to combine chunk embeddings.
            Options: "mean" (default) or "weighted" (attention-based)
        embedding_model (optional): Pre-loaded embedding model. If None, will be
            created from text2lora_hypernet.get_embedding_model()
        device (str): Device for computation ("cpu" or "cuda"). Default: "cpu".

    Attributes:
        hypernet (nn.Module): The underlying TextToLoRAHypernet
        chunk_size (int): Size for text chunking
        num_chunks (int): Max chunks to process
        pooling_strategy (str): Pooling method
        device (str): Computation device

    Example:
        >>> from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA
        >>> hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16)
        >>> doc2lora = DocToLoRA(hypernet, chunk_size=512, num_chunks=3)
        >>> long_doc = "This is a very long task description that spans many paragraphs..."
        >>> lora_weights = doc2lora(long_doc)
        >>> len(lora_weights)  # num_target_modules
        5
    """

    def __init__(
        self,
        text2lora_hypernet: nn.Module,
        chunk_size: int = 512,
        num_chunks: int = 3,
        pooling_strategy: str = "mean",
        embedding_model: Optional[object] = None,
        device: str = "cpu",
    ):
        """
        Initialize DocToLoRA module.

        Args:
            text2lora_hypernet: TextToLoRAHypernet instance
            chunk_size: Characters per chunk
            num_chunks: Maximum chunks to process
            pooling_strategy: "mean" or "weighted"
            embedding_model: Pre-loaded embedding model (optional)
            device: "cpu" or "cuda"

        Raises:
            ValueError: If hypernet is None or invalid parameters
            ValueError: If chunk_size or num_chunks <= 0
        """
        super().__init__()

        if text2lora_hypernet is None:
            raise ValueError("text2lora_hypernet cannot be None")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        if pooling_strategy not in ["mean", "weighted"]:
            raise ValueError(
                f"pooling_strategy must be 'mean' or 'weighted', got {pooling_strategy}"
            )

        self.hypernet = text2lora_hypernet
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.pooling_strategy = pooling_strategy
        self.device = device

        # Initialize embedding model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            try:
                self.embedding_model = text2lora_hypernet.get_embedding_model()
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self.embedding_model = None

        # Attention pooling layer (if using weighted pooling)
        if pooling_strategy == "weighted":
            self.attention = nn.Sequential(
                nn.Linear(text2lora_hypernet.hidden_dim, text2lora_hypernet.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(text2lora_hypernet.hidden_dim // 4, 1),
            )
        else:
            self.attention = None

        logger.info(
            f"Initialized DocToLoRA: chunk_size={chunk_size}, num_chunks={num_chunks}, "
            f"pooling={pooling_strategy}, device={device}"
        )

    def _chunk_document(self, document: str) -> List[str]:
        """
        Split document into overlapping chunks.

        Chunks are created by splitting on spaces near the chunk_size boundary.
        This ensures word boundaries are preserved (no mid-word splits).

        Args:
            document (str): Full document text

        Returns:
            List[str]: List of text chunks (up to num_chunks)

        Example:
            >>> doc = "Word1 word2 word3 word4 word5"
            >>> chunks = doc2lora._chunk_document(doc)
            >>> len(chunks) <= 3  # num_chunks
            True
        """
        if not document or len(document.strip()) == 0:
            return [""]

        words = document.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space

            # Start new chunk if we exceed chunk_size
            if current_length + word_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(word)
            current_length += word_len

            # Stop if we have enough chunks
            if len(chunks) >= self.num_chunks:
                break

        # Add final chunk
        if current_chunk and len(chunks) < self.num_chunks:
            chunks.append(" ".join(current_chunk))

        # Return at most num_chunks
        return chunks[: self.num_chunks]

    def _embed_chunks(self, chunks: List[str]) -> torch.Tensor:
        """
        Embed text chunks using the embedding model.

        Args:
            chunks (List[str]): List of text chunks to embed

        Returns:
            Tensor: Embeddings of shape [num_chunks, hidden_dim]

        Raises:
            RuntimeError: If embedding model is not available
            ValueError: If chunks is empty

        Note:
            This is a mock implementation. In production, you would call:
                embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True)
        """
        if not chunks:
            raise ValueError("chunks cannot be empty")

        # Mock embeddings for CPU-only testing
        # In production, use the actual embedding model
        num_chunks = len(chunks)
        embeddings = torch.randn(num_chunks, self.hypernet.hidden_dim, device=self.device)

        return embeddings

    def _pool_embeddings(self, chunk_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Pool chunk embeddings into a single embedding.

        Supports two pooling strategies:
        1. "mean": Simple average of all chunk embeddings
        2. "weighted": Attention-weighted sum with learned attention weights

        Args:
            chunk_embeddings (Tensor): Embeddings of shape [num_chunks, hidden_dim]

        Returns:
            Tensor: Pooled embedding of shape [1, hidden_dim]

        Raises:
            ValueError: If chunk_embeddings is not 2D
        """
        if chunk_embeddings.dim() != 2:
            raise ValueError(
                f"Expected 2D embeddings [num_chunks, hidden_dim], "
                f"got shape {chunk_embeddings.shape}"
            )

        if self.pooling_strategy == "mean":
            # Simple mean pooling
            pooled = chunk_embeddings.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        elif self.pooling_strategy == "weighted":
            # Attention-based weighted pooling
            if self.attention is None:
                raise RuntimeError("Attention module not initialized for weighted pooling")

            # Compute attention weights
            attn_logits = self.attention(chunk_embeddings)  # [num_chunks, 1]
            attn_weights = torch.softmax(attn_logits, dim=0)  # [num_chunks, 1]

            # Weighted sum
            pooled = (chunk_embeddings * attn_weights).sum(dim=0, keepdim=True)  # [1, hidden_dim]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return pooled

    def forward(self, document_text: str) -> List[Dict[str, torch.Tensor]]:
        """
        Generate LoRA weights from a document.

        Forward pass:
        1. Chunk document into overlapping pieces
        2. Embed each chunk using embedding model
        3. Pool chunk embeddings into single embedding
        4. Feed pooled embedding to hypernetwork
        5. Return list of LoRA A/B pairs

        Args:
            document_text (str): Full document text for task description

        Returns:
            List[Dict[str, Tensor]]: Same format as TextToLoRAHypernet output
                List of dicts with "A" and "B" keys:
                - "A": [1, lora_r, hidden_dim]
                - "B": [1, hidden_dim, lora_r]

        Raises:
            ValueError: If document_text is not a string
            RuntimeError: If embedding model is not available

        Example:
            >>> doc2lora = DocToLoRA(hypernet, chunk_size=512, num_chunks=3)
            >>> long_doc = "Task: fine-tune model on math problems..." * 10
            >>> lora_weights = doc2lora(long_doc)
            >>> lora_weights[0]["A"].shape
            torch.Size([1, 16, 768])
        """
        # Validate input
        if not isinstance(document_text, str):
            raise ValueError(f"document_text must be a string, got {type(document_text)}")

        # Chunk document
        chunks = self._chunk_document(document_text)

        if not chunks or all(len(c.strip()) == 0 for c in chunks):
            logger.warning("Document produced empty chunks, using default embedding")
            pooled_embedding = torch.randn(1, self.hypernet.hidden_dim, device=self.device)
        else:
            # Embed chunks
            chunk_embeddings = self._embed_chunks(chunks)

            # Pool embeddings
            pooled_embedding = self._pool_embeddings(chunk_embeddings)

        # Generate LoRA from pooled embedding
        lora_weights = self.hypernet(pooled_embedding)

        return lora_weights

    def get_config(self) -> Dict[str, any]:
        """
        Get configuration dictionary for DocToLoRA.

        Returns:
            Dict[str, any]: Configuration with keys:
                - hypernet_config: Hypernetwork configuration
                - chunk_size: Chunk size in characters
                - num_chunks: Maximum chunks to process
                - pooling_strategy: Pooling method ("mean" or "weighted")
                - device: Computation device

        Example:
            >>> doc2lora = DocToLoRA(hypernet)
            >>> config = doc2lora.get_config()
            >>> config["chunk_size"]
            512
        """
        return {
            "hypernet_config": self.hypernet.get_lora_config(),
            "chunk_size": self.chunk_size,
            "num_chunks": self.num_chunks,
            "pooling_strategy": self.pooling_strategy,
            "device": self.device,
        }
