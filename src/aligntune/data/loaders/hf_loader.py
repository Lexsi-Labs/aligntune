from typing import Optional, Any
from datasets import load_dataset
from .base import BaseLoader

class HFLoader(BaseLoader):
    def __init__(
        self, 
        name: str, 
        config_name: Optional[str] = None, 
        split: Optional[str] = None, 
        **kwargs: Any
    ):
        """
        Args:
            name: The repo ID (e.g., 'openai/gsm8k')
            config_name: The specific configuration (e.g., 'main', 'socratic')
            split: Specific split to load (optional)
            **kwargs: Extra arguments for load_dataset (cache_dir, etc.)
        """
        self.name = name
        self.config_name = config_name
        self.split = split
        self.kwargs = kwargs

    def load(self):
        # Extract max_samples if present (AlignTune-specific, not for load_dataset)
        max_samples = self.kwargs.pop('max_samples', None)
        
        # We construct the args dynamically
        load_args = [self.name]
        if self.config_name:
            load_args.append(self.config_name)
            
        dataset = load_dataset(
            *load_args,
            split=self.split,
            **self.kwargs
        )
        
        # Apply max_samples after loading if specified
        if max_samples is not None and hasattr(dataset, '__len__'):
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
        
        return dataset