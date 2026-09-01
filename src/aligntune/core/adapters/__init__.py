"""
AlignTune Advanced Adapters Module (v3.3)

Provides multiple LoRA variant implementations:
- STANDARD: Classic LoRA (LoRA from PEFT)
- MOA: Mixture of Adapters
- TEXT2LORA: Text-guided LoRA (Hypernetwork)
- DOC2LORA: Documentation-guided LoRA

Current implementation includes:
- Mixture of Adapters (MoA) with top-k gating
- TextToLoRA: Hypernetwork for generating LoRA from embeddings
- DocToLoRA: Document-based LoRA generation with chunking
"""

from .moa import MoARouter, MoALoraLayer
from .text2lora import TextToLoRAHypernet, TextToLoRATrainer, DocToLoRA

__all__ = [
    "MoARouter",
    "MoALoraLayer",
    "TextToLoRAHypernet",
    "TextToLoRATrainer",
    "DocToLoRA",
]
