"""
Dataset Adapter System for AlignTune.

This module provides a flexible system to handle different dataset formats
by automatically detecting schemas and mapping fields to expected training formats.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datasets import Dataset
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """Field mapping configuration for dataset adaptation."""
    prompt_field: str
    completion_field: str
    context_field: Optional[str] = None
    chosen_field: Optional[str] = None
    rejected_field: Optional[str] = None


class DatasetAdapter(ABC):
    """Base class for dataset adapters."""
    
    @abstractmethod
    def detect_format(self, dataset: Dataset) -> bool:
        """Check if this adapter can handle the dataset format."""
        pass
    
    @abstractmethod
    def get_field_mappings(self, training_type: str) -> FieldMapping:
        """Get field mappings for the specified training type."""
        pass
    
    @abstractmethod
    def adapt_dataset(self, dataset: Dataset, training_type: str, preserve_columns: Optional[List[str]] = None) -> Dataset:
        """
        Adapt dataset to expected format for training type.
        
        Args:
            dataset: The input dataset.
            training_type: 'sft' or 'dpo'/'rl'.
            preserve_columns: List of column names to keep from the original dataset.
                            If None, only the new columns (prompt/completion/etc) are kept.
        """
        pass

    def _get_columns_to_remove(self, dataset: Dataset, preserve_columns: Optional[List[str]]) -> List[str]:
        """Helper to calculate which columns to remove."""
        if preserve_columns is None:
            # Default behavior: remove all original columns
            return dataset.column_names
        
        # Keep specified columns
        return [col for col in dataset.column_names if col not in preserve_columns]


class AlpacaAdapter(DatasetAdapter):
    """Adapter for Alpaca format datasets."""
    
    def detect_format(self, dataset: Dataset) -> bool:
        """Check if dataset has Alpaca format fields."""
        fields = list(dataset.features.keys())
        return "instruction" in fields and "output" in fields
    
    def get_field_mappings(self, training_type: str) -> FieldMapping:
        """Get field mappings for Alpaca format."""
        if training_type == "sft":
            return FieldMapping(
                prompt_field="instruction",
                completion_field="output"
            )
        elif training_type == "dpo":
            return FieldMapping(
                prompt_field="instruction",
                chosen_field="output",
                rejected_field="rejected_output"  # May not exist
            )
        else:
            raise ValueError(f"Unsupported training type: {training_type}")
    
    def adapt_dataset(self, dataset: Dataset, training_type: str, preserve_columns: Optional[List[str]] = None) -> Dataset:
        """Adapt Alpaca dataset to expected format."""
        mappings = self.get_field_mappings(training_type)
        
        def map_fields(example):
            result = {}
            if training_type == "sft":
                # Create text field for SFT training
                instruction = example.get(mappings.prompt_field, "")
                input_text = example.get("input", "")
                output = example.get(mappings.completion_field, "")
                
                # Combine instruction and input if both exist
                if input_text:
                    prompt = f"{instruction}\n\nInput: {input_text}"
                else:
                    prompt = instruction
                
                result["text"] = f"{prompt}\n\nResponse: {output}"
                result["prompt"] = prompt
                result["completion"] = output
            elif training_type == "dpo":
                result["prompt"] = example.get(mappings.prompt_field, "")
                result["chosen"] = example.get(mappings.chosen_field, "")
                result["rejected"] = example.get(mappings.rejected_field, "")
            
            return result
        
        remove_cols = self._get_columns_to_remove(dataset, preserve_columns)
        return dataset.map(map_fields, remove_columns=remove_cols)


class DollyAdapter(DatasetAdapter):
    """Adapter for Dolly format datasets."""
    
    def detect_format(self, dataset: Dataset) -> bool:
        """Check if dataset has Dolly format fields."""
        fields = list(dataset.features.keys())
        return "instruction" in fields and "response" in fields
    
    def get_field_mappings(self, training_type: str) -> FieldMapping:
        """Get field mappings for Dolly format."""
        if training_type == "sft":
            return FieldMapping(
                prompt_field="instruction",
                completion_field="response",
                context_field="context"
            )
        elif training_type == "dpo":
            return FieldMapping(
                prompt_field="instruction",
                chosen_field="response",
                rejected_field="rejected_response"  # May not exist
            )
        else:
            raise ValueError(f"Unsupported training type: {training_type}")
    
    def adapt_dataset(self, dataset: Dataset, training_type: str, preserve_columns: Optional[List[str]] = None) -> Dataset:
        """Adapt Dolly dataset to expected format."""
        mappings = self.get_field_mappings(training_type)
        
        def map_fields(example):
            result = {}
            if training_type == "sft":
                # Create text field for SFT training
                instruction = example.get(mappings.prompt_field, "")
                context = example.get(mappings.context_field, "")
                response = example.get(mappings.completion_field, "")
                
                # Combine instruction and context if both exist
                if context:
                    prompt = f"Context: {context}\n\nInstruction: {instruction}"
                else:
                    prompt = instruction
                
                result["text"] = f"{prompt}\n\nResponse: {response}"
                result["prompt"] = prompt
                result["completion"] = response
            elif training_type == "dpo":
                result["prompt"] = example.get(mappings.prompt_field, "")
                result["chosen"] = example.get(mappings.chosen_field, "")
                result["rejected"] = example.get(mappings.rejected_field, "")
            
            return result
        
        remove_cols = self._get_columns_to_remove(dataset, preserve_columns)
        return dataset.map(map_fields, remove_columns=remove_cols)


class UltraChatAdapter(DatasetAdapter):
    """Adapter for UltraChat format datasets."""
    
    def detect_format(self, dataset: Dataset) -> bool:
        """Check if dataset has UltraChat format fields."""
        fields = list(dataset.features.keys())
        return "prompt" in fields and "messages" in fields
    
    def get_field_mappings(self, training_type: str) -> FieldMapping:
        """Get field mappings for UltraChat format."""
        if training_type == "sft":
            return FieldMapping(
                prompt_field="prompt",
                completion_field="messages"  # Will be processed specially
            )
        elif training_type == "dpo":
            return FieldMapping(
                prompt_field="prompt",
                chosen_field="messages",
                rejected_field="rejected_messages"  # May not exist
            )
        else:
            raise ValueError(f"Unsupported training type: {training_type}")
    
    def adapt_dataset(self, dataset: Dataset, training_type: str, preserve_columns: Optional[List[str]] = None) -> Dataset:
        """Adapt UltraChat dataset to expected format."""
        mappings = self.get_field_mappings(training_type)
        
        def map_fields(example):
            result = {}
            if training_type == "sft":
                # Process conversation format
                prompt = example.get(mappings.prompt_field, "")
                messages = example.get(mappings.completion_field, [])
                
                # Convert messages to text format
                conversation_text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        conversation_text += f"Human: {content}\n"
                    elif role == "assistant":
                        conversation_text += f"Assistant: {content}\n"
                
                result["text"] = conversation_text.strip()
                result["prompt"] = prompt
                result["completion"] = conversation_text.strip()
            elif training_type == "dpo":
                result["prompt"] = example.get(mappings.prompt_field, "")
                result["chosen"] = str(example.get(mappings.chosen_field, ""))
                result["rejected"] = str(example.get(mappings.rejected_field, ""))
            
            return result
        
        remove_cols = self._get_columns_to_remove(dataset, preserve_columns)
        return dataset.map(map_fields, remove_columns=remove_cols)


class CustomAdapter(DatasetAdapter):
    """Adapter for custom dataset formats with user-defined mappings."""
    
    def __init__(self, field_mappings: Dict[str, str]):
        self.field_mappings = field_mappings
    
    def detect_format(self, dataset: Dataset) -> bool:
        """Custom adapter always matches if mappings are provided."""
        return len(self.field_mappings) > 0
    
    def get_field_mappings(self, training_type: str) -> FieldMapping:
        """Get field mappings from user configuration."""
        if training_type == "sft":
            return FieldMapping(
                prompt_field=self.field_mappings.get("prompt", "prompt"),
                completion_field=self.field_mappings.get("completion", "completion")
            )
        elif training_type == "dpo":
            return FieldMapping(
                prompt_field=self.field_mappings.get("prompt", "prompt"),
                chosen_field=self.field_mappings.get("chosen", "chosen"),
                rejected_field=self.field_mappings.get("rejected", "rejected")
            )
        else:
            raise ValueError(f"Unsupported training type: {training_type}")
    
    def adapt_dataset(self, dataset: Dataset, training_type: str, preserve_columns: Optional[List[str]] = None) -> Dataset:
        """Adapt dataset using custom field mappings."""
        mappings = self.get_field_mappings(training_type)
        
        def map_fields(example):
            result = {}
            if training_type == "sft":
                prompt = example.get(mappings.prompt_field, "")
                completion = example.get(mappings.completion_field, "")
                result["text"] = f"{prompt}\n\nResponse: {completion}"
                result["prompt"] = prompt
                result["completion"] = completion
            elif training_type == "dpo":
                result["prompt"] = example.get(mappings.prompt_field, "")
                result["chosen"] = example.get(mappings.chosen_field, "")
                result["rejected"] = example.get(mappings.rejected_field, "")
            
            return result
        
        remove_cols = self._get_columns_to_remove(dataset, preserve_columns)
        return dataset.map(map_fields, remove_columns=remove_cols)


class DatasetSchemaDetector:
    """Automatic dataset schema detection and adapter selection."""
    
    def __init__(self):
        self.adapters = [
            AlpacaAdapter(),
            DollyAdapter(),
            UltraChatAdapter(),
        ]
    
    def detect_format(self, dataset: Dataset) -> str:
        """Auto-detect dataset format based on field names."""
        fields = list(dataset.features.keys())
        logger.info(f"Detected dataset fields: {fields}")
        
        # Check each adapter
        for adapter in self.adapters:
            if adapter.detect_format(dataset):
                adapter_name = adapter.__class__.__name__.replace("Adapter", "").lower()
                logger.info(f"Detected format: {adapter_name}")
                return adapter_name
        
        # Check for common patterns
        if "instruction" in fields and "response" in fields:
            return "dolly"
        elif "instruction" in fields and "output" in fields:
            return "alpaca"
        elif "prompt" in fields and "messages" in fields:
            return "ultrachat"
        elif "text" in fields:
            return "text"
        else:
            return "custom"
    
    def get_adapter(self, format_type: str, custom_mappings: Optional[Dict[str, str]] = None) -> DatasetAdapter:
        """Get appropriate adapter for the detected format."""
        if format_type == "alpaca":
            return AlpacaAdapter()
        elif format_type == "dolly":
            return DollyAdapter()
        elif format_type == "ultrachat":
            return UltraChatAdapter()
        elif format_type == "custom" and custom_mappings:
            return CustomAdapter(custom_mappings)
        else:
            raise ValueError(f"No adapter available for format: {format_type}")
    
    def adapt_dataset(self, dataset: Dataset, training_type: str, 
                     format_type: Optional[str] = None, 
                     custom_mappings: Optional[Dict[str, str]] = None,
                     preserve_columns: Optional[List[str]] = None) -> Dataset:
        """
        Adapt dataset to expected format for training.
        
        Args:
            dataset: Input dataset
            training_type: 'sft' or 'rl'/'dpo'
            format_type: Optional explicit format type
            custom_mappings: Optional custom field mappings
            preserve_columns: Optional list of columns to preserve (not remove)
        """
        # Auto-detect format if not specified
        if format_type is None:
            format_type = self.detect_format(dataset)
        
        # Get appropriate adapter
        adapter = self.get_adapter(format_type, custom_mappings)
        
        # Adapt dataset
        logger.info(f"Adapting dataset using {adapter.__class__.__name__} for {training_type} training")
        adapted_dataset = adapter.adapt_dataset(dataset, training_type)
        
        logger.info(f"Dataset adapted successfully. New fields: {list(adapted_dataset.features.keys())}")
        return adapted_dataset


# Global instance for easy access
schema_detector = DatasetSchemaDetector()