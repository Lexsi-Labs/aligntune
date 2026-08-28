"""
PEFT Factory for routing configurations to the correct adapter.
"""
from typing import Any

from .lora import LoraAdapter


class PEFTFactory:
    """Factory to create the appropriate PEFTAdapter based on the configuration."""

    @staticmethod
    def get_adapter(config: Any) -> LoraAdapter:
        return LoraAdapter(config)
