"""
Text-to-LoRA Hypernetwork Module for AlignTune v3.3 Advanced Parameterization Suite

Generates LoRA A/B matrices from task description embeddings using a hypernetwork.
This module implements meta-training for generating task-specific LoRA weights.

Components:
- TextToLoRAHypernet: Hypernetwork that maps embeddings to LoRA weights
- TextToLoRATrainer: Meta-training loop for hypernetwork optimization
- DocToLoRA: Document-based specialization for generating LoRAs from long documents

Key features:
- Embedding → Dense layers → Shaped LoRA matrices (A, B)
- Support for multiple target modules
- CPU-compatible training and inference
- Proper LoRA initialization (A: N(0, std), B: zeros)
- Meta-training on task distributions

References:
- LoRA: Low-Rank Adaptation of Large Language Models
- Hypernetworks: Neural Hypernetworks as Data Generators
"""

from .hypernet import TextToLoRAHypernet
from .train import TextToLoRATrainer
from .doc2lora import DocToLoRA

__all__ = [
    "TextToLoRAHypernet",
    "TextToLoRATrainer",
    "DocToLoRA",
]
