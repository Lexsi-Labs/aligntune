"""
Registry system for datasets, rewards, and tasks - FIXED VERSION.

This module provides a flexible registry system that allows easy registration
and retrieval of dataset loaders, reward functions, and task definitions.
"""

from typing import Dict, Any, Callable, List, Optional, Union
from abc import ABC, abstractmethod
import logging
import torch
from datasets import load_dataset, Dataset
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def _initialize_from_rewards_registry():
    """Initialize RL core registry from rewards module registry."""
    try:
        from ...rewards.registry import RewardRegistry as RewardsRegistry
        # Copy all registered rewards
        reward_count = 0
        for name in RewardsRegistry.list_rewards():
            try:
                reward_func_obj = RewardsRegistry.get_reward_function(name)
                # Extract the compute method which is callable
                if hasattr(reward_func_obj, 'compute'):
                    callable_func = reward_func_obj.compute
                elif callable(reward_func_obj):
                    callable_func = reward_func_obj
                else:
                    logger.warning(f"Reward function '{name}' is not callable, skipping")
                    continue
                RewardRegistry.register(name, callable_func)
                reward_count += 1
            except (ValueError, KeyError) as e:
                # Handle reward type parsing errors gracefully
                error_msg = str(e)
                if "Unknown reward type" in error_msg or "RewardT" in error_msg or "RewardType" in error_msg:
                    logger.debug(f"Could not initialize reward '{name}' from rewards registry: {e}")
                    # Don't log as warning for parsing errors - they're expected for some reward types
                    continue
                else:
                    logger.warning(f"Could not initialize reward '{name}' from rewards registry: {e}")
            except Exception as e:
                # Log other errors as warnings
                logger.warning(f"Could not initialize reward '{name}' from rewards registry: {e}")
        
        if reward_count > 0:
            logger.info(f"Initialized RL core registry with {reward_count} reward functions")
        else:
            logger.debug("No reward functions initialized from rewards registry")
    except ImportError as e:
        logger.debug(f"Rewards registry not available: {e}")
    except Exception as e:
        logger.warning(f"Could not initialize from rewards registry: {e}")


class DatasetRegistry:
    """Registry for dataset loaders and schema adapters."""
    
    _loaders: Dict[str, Callable] = {}
    _adapters: Dict[str, Callable] = {}
    
    @classmethod
    def register_loader(cls, name: str, loader_func: Callable):
        """Register a dataset loader function."""
        if not callable(loader_func):
            raise ValueError(f"Loader function for '{name}' must be callable")
        cls._loaders[name] = loader_func
        logger.info(f"Registered dataset loader: {name}")
    
    @classmethod
    def register_adapter(cls, schema: str, adapter_func: Callable):
        """Register a schema adapter function."""
        if not callable(adapter_func):
            raise ValueError(f"Adapter function for '{schema}' must be callable")
        cls._adapters[schema] = adapter_func
        logger.info(f"Registered schema adapter: {schema}")
    
    @classmethod
    def load_dataset(cls, name: str, **kwargs) -> Dataset:
        """Load dataset by name using registered loader."""
        
        # Extract custom parameters that aren't for HuggingFace load_dataset
        custom_params = {}
        hf_params = {}
        
        # Parameters in this set are treated as AlignTune-specific/custom and
        # are NOT forwarded to Hugging Face's `load_dataset` (or custom loaders
        # that in turn call `load_dataset`). This prevents errors where a HF
        # dataset `BuilderConfig` does not define the corresponding fields.
        #
        # Example: some configs (like McAuley-Lab/Amazon-Reviews-2023) do not
        # accept RL-specific arguments such as `field_mappings` or
        # `auto_detect_fields`. Passing them through would cause errors like:
        #   ValueError: BuilderConfig ... doesn't have a 'field_mappings' key.
        custom_param_names = {
            # Generic dataset controls
            'percent',
            'max_samples',
            'cache',
            'tokenizer',
            'max_length',
            'chat_template',
            'column_mapping',
            'system_prompt',
            # RL dataset-specific controls that should never hit HF builders
            'field_mappings',
            'format_type',
            'auto_detect_fields',
            'task_type',
            'weight',
            'dataset_num_proc',
            'pad_token',
            'label_pad_token_id',
            'truncation_mode',
            'padding_free',
            'precompute_ref_log_probs',
            'precompute_ref_batch_size',
            'tools',
        }
        
        for key, value in kwargs.items():
            if key in custom_param_names:
                custom_params[key] = value
            else:
                hf_params[key] = value
        
        # Load the base dataset
        if name in cls._loaders:
            dataset = cls._loaders[name](name=name, **hf_params)
        else:
            # Try HuggingFace datasets as fallback
            try:
                dataset = load_dataset(name, **hf_params)
                
                # Auto-detect and adapt Q&A format if needed
                if len(dataset) > 0:
                    sample_cols = dataset.column_names
                    # Check if it looks like Q&A format (has 'question' or 'query' but no 'prompt')
                    if ('question' in sample_cols or 'query' in sample_cols) and 'prompt' not in sample_cols:
                        system_prompt = custom_params.get('system_prompt')
                        logger.info(f"Auto-detected Q&A format, applying Q&A schema adapter")
                        if system_prompt:
                            logger.info(f"Using system prompt: {system_prompt[:50]}...")
                        dataset = _adapt_qa_schema(dataset, system_prompt=system_prompt)
                        
            except Exception as e:
                raise ValueError(
                    f"Failed to load dataset '{name}'. "
                    f"Available registered loaders: {list(cls._loaders.keys())}. "
                    f"Error: {e}"
                )
        
        # Apply custom processing
        
        # 1. Handle percent sampling
        if custom_params.get('percent') is not None:
            percent = custom_params['percent']
            if percent < 100:
                num_samples = int(len(dataset) * (percent / 100.0))
                dataset = dataset.select(range(num_samples))
                logger.info(f"Sampled {percent}% of dataset: {num_samples} samples")
        
        # 2. Handle max_samples
        if custom_params.get('max_samples') is not None:
            max_samples = custom_params['max_samples']
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                logger.info(f"Limited dataset to {max_samples} samples")
        
        # 3. Handle caching
        cache = custom_params.get('cache')
        if cache is not None and hasattr(cache, 'cache_exists'):
            # Generate cache key
            tokenizer = custom_params.get('tokenizer')
            tokenizer_name = getattr(tokenizer, 'name_or_path', 'unknown') if tokenizer else 'unknown'
            
            cache_key = cache.get_cache_key(
                dataset_name=name,
                split=hf_params.get('split', 'train'),
                tokenizer_name=tokenizer_name,
                max_length=custom_params.get('max_length', 512),
                template=custom_params.get('chat_template'),
                column_mapping=custom_params.get('column_mapping'),
                percent=custom_params.get('percent'),
                max_samples=custom_params.get('max_samples')
            )
            
            # Try to load from cache
            if cache.cache_exists(cache_key):
                try:
                    dataset = cache.load_from_cache(cache_key)
                    logger.info(f"Loaded dataset from cache: {cache_key}")
                    return dataset
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {e}. Loading fresh dataset.")
            
            # Save to cache after all processing
            try:
                cache.save_to_cache(dataset, cache_key, metadata={
                    'dataset_name': name,
                    'split': hf_params.get('split', 'train'),
                    'percent': custom_params.get('percent'),
                    'max_samples': custom_params.get('max_samples')
                })
            except Exception as e:
                logger.warning(f"Failed to save to cache: {e}")
        
        # 4. Handle column mapping
        column_mapping = custom_params.get('column_mapping')
        if column_mapping:
            def rename_columns(example):
                renamed = {}
                for new_key, old_key in column_mapping.items():
                    if old_key in example:
                        renamed[new_key] = example[old_key]
                    else:
                        renamed[new_key] = example.get(new_key, "")
                # Keep other columns
                for key, value in example.items():
                    if key not in column_mapping.values() and key not in renamed:
                        renamed[key] = value
                return renamed
            
            dataset = dataset.map(rename_columns)
            logger.info(f"Applied column mapping: {column_mapping}")
        
        return dataset
    
    @classmethod
    def adapt_schema(cls, dataset: Dataset, schema: str) -> Dataset:
        """Apply schema adapter to dataset."""
        if schema not in cls._adapters:
            logger.warning(f"No adapter found for schema '{schema}', returning original dataset")
            return dataset
        
        return cls._adapters[schema](dataset)
    
    @classmethod
    def list_loaders(cls) -> List[str]:
        """List all registered dataset loaders."""
        return list(cls._loaders.keys())
    
    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered schema adapters."""
        return list(cls._adapters.keys())


class RewardRegistry:
    """Registry for reward functions."""
    
    _functions: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, reward_func: Callable):
        """Register a reward function."""
        if not callable(reward_func):
            raise ValueError(f"Reward function '{name}' must be callable")
        cls._functions[name] = reward_func
        logger.info(f"Registered reward function: {name}")
    
    @classmethod
    def get_reward(cls, name: str, **params) -> Callable:
        """Get configured reward function."""
        if name not in cls._functions:
            raise ValueError(f"Unknown reward function: {name}. Available: {list(cls._functions.keys())}")
        
        reward_func = cls._functions[name]
        
        # If the function accepts parameters, return a configured version
        if params:
            def configured_reward(*args, **kwargs):
                return reward_func(*args, **kwargs, **params)
            return configured_reward
        
        return reward_func
    
    @classmethod
    def get_reward_function(cls, name: str, **params) -> Callable:
        """Alias for get_reward() for backward compatibility."""
        return cls.get_reward(name, **params)
    
    @classmethod
    def list_functions(cls) -> List[str]:
        """List all registered reward functions."""
        return list(cls._functions.keys())


# Initialize the registry after class definition
_initialize_from_rewards_registry()


class TaskRegistry:
    """Registry for task definitions."""
    
    _tasks: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, task_config: Dict[str, Any]):
        """Register a task definition."""
        if not isinstance(task_config, dict):
            raise ValueError(f"Task config for '{name}' must be a dictionary")
        
        # Validate required fields
        required_fields = ["description", "input_format", "output_format"]
        for field in required_fields:
            if field not in task_config:
                raise ValueError(f"Task config for '{name}' must include '{field}'")
        
        cls._tasks[name] = task_config
        logger.info(f"Registered task: {name}")
    
    @classmethod
    def get_task(cls, name: str) -> Dict[str, Any]:
        """Get task configuration."""
        if name not in cls._tasks:
            raise ValueError(f"Unknown task: {name}. Available: {list(cls._tasks.keys())}")
        
        return cls._tasks[name].copy()
    
    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered tasks."""
        return list(cls._tasks.keys())


# Built-in dataset loaders
def _load_hf_dataset(name: str, split: str = "train", **kwargs) -> Dataset:
    """Load dataset from HuggingFace Hub."""
    # Filter out custom parameters that HuggingFace doesn't understand
    hf_params = {}
    custom_param_names = {
        'percent', 'max_samples', 'cache', 'tokenizer', 
        'max_length', 'chat_template', 'column_mapping'
    }
    
    for key, value in kwargs.items():
        if key not in custom_param_names:
            hf_params[key] = value
    
    return load_dataset(name, split=split, **hf_params)


def _load_local_dataset(path: str, **kwargs) -> Dataset:
    """Load dataset from local path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")
    
    # Filter out custom parameters
    hf_params = {}
    custom_param_names = {
        'percent', 'max_samples', 'cache', 'tokenizer', 
        'max_length', 'chat_template', 'column_mapping'
    }
    
    for key, value in kwargs.items():
        if key not in custom_param_names:
            hf_params[key] = value
    
    if path.suffix == '.json':
        return load_dataset('json', data_files=str(path), **hf_params)
    elif path.suffix == '.csv':
        return load_dataset('csv', data_files=str(path), **hf_params)
    else:
        # Try to load as directory
        return load_dataset(str(path), **hf_params)


# Built-in schema adapters
def _adapt_chat_schema(dataset: Dataset) -> Dataset:
    """Adapt dataset to chat format."""
    def transform_example(example):
        if 'messages' in example:
            return example
        
        # Convert instruction/response format to messages
        if 'instruction' in example and 'response' in example:
            return {
                'messages': [
                    {'role': 'user', 'content': example['instruction']},
                    {'role': 'assistant', 'content': example['response']}
                ]
            }
        
        return example
    
    return dataset.map(transform_example)


def _adapt_hh_rlhf_schema(dataset: Dataset) -> Dataset:
    """Adapt dataset to HH-RLHF format."""
    def transform_example(example):
        if 'chosen' in example and 'rejected' in example:
            return {
                'prompt': example.get('prompt', ''),
                'chosen': example['chosen'],
                'rejected': example['rejected']
            }
        return example
    
    return dataset.map(transform_example)


def _adapt_ultrafeedback_schema(dataset: Dataset) -> Dataset:
    """Adapt dataset to UltraFeedback format."""
    def transform_example(example):
        if 'completions' in example:
            completions = example['completions']
            if len(completions) >= 2:
                return {
                    'prompt': example.get('prompt', ''),
                    'chosen': completions[0]['response'],
                    'rejected': completions[1]['response']
                }
        return example
    
    return dataset.map(transform_example)


def _adapt_alpaca_schema(dataset: Dataset) -> Dataset:
    """Adapt dataset to Alpaca format."""
    def transform_example(example):
        if 'instruction' in example and 'output' in example:
            return {
                'instruction': example['instruction'],
                'input': example.get('input', ''),
                'output': example['output']
            }
        return example
    
    return dataset.map(transform_example)


def _adapt_qa_schema(dataset: Dataset, system_prompt: Optional[str] = None) -> Dataset:
    """Auto-adapt Q&A datasets to prompt format for RL training.
    
    Detects common Q&A formats:
    - 'question' + 'answers' -> 'prompt': "Question: {question}\n\nAnswer:"
    - 'question' + 'answer' -> 'prompt': "Question: {question}\n\nAnswer:"
    - 'query' -> 'prompt': "Question: {query}\n\nAnswer:"
    
    Args:
        dataset: Dataset to adapt
        system_prompt: Optional system prompt to prepend to each question
    """
    if len(dataset) == 0:
        return dataset
    
    # Check first example to detect schema
    sample = dataset[0]
    columns = dataset.column_names
    
    # Detect Q&A format
    has_question = 'question' in columns
    has_query = 'query' in columns
    has_answers = 'answers' in columns or 'answer' in columns
    
    # Only adapt if we detect Q&A format and don't already have 'prompt'
    if 'prompt' in columns:
        return dataset  # Already formatted
    
    if not (has_question or has_query):
        return dataset  # Not a Q&A dataset
    
    def format_qa_example(example):
        """Format Q&A example to prompt format."""
        # Get question from various possible fields
        question = (
            example.get('question') or 
            example.get('query') or 
            example.get('instruction') or
            ''
        )
        
        # Format as prompt with optional system prompt
        if question and isinstance(question, str) and question.strip():
            if system_prompt:
                prompt = f"{system_prompt}\n\nQuestion: {question.strip()}\n\nAnswer:"
            else:
                prompt = f"Question: {question.strip()}\n\nAnswer:"
        else:
            prompt = ""
        
        return {"prompt": prompt}
    
    # Format dataset
    dataset = dataset.map(format_qa_example, batched=False)
    
    # Filter out empty prompts
    dataset = dataset.filter(lambda x: x['prompt'].strip())
    
    # Remove other columns (keep only 'prompt' for RL training)
    cols_to_remove = [c for c in dataset.column_names if c != 'prompt']
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    
    return dataset


# Built-in reward functions
def _length_reward(text: str, min_length: int = 10, max_length: int = 500, **kwargs) -> float:
    """Reward based on text length."""
    length = len(text.split())
    if length < min_length:
        return 0.0
    elif length > max_length:
        return 0.0
    else:
        # Normalize to 0-1 range
        return (length - min_length) / (max_length - min_length)


# Sentiment pipeline cache to avoid reloading
_sentiment_pipeline = None

def _sentiment_reward(text: str, positive_weight: float = 1.0, **kwargs) -> float:
    """Reward based on sentiment (requires additional dependencies)."""
    global _sentiment_pipeline
    
    try:
        from transformers import pipeline
        
        # Initialize pipeline once and reuse
        if _sentiment_pipeline is None:
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
        
        result = _sentiment_pipeline(text[:512])[0]  # Limit text length
        
        if result['label'].lower() == 'positive':
            return result['score'] * positive_weight
        else:
            return (1.0 - result['score']) * positive_weight
    except ImportError:
        logger.warning("Sentiment analysis not available, returning neutral reward")
        return 0.5
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")
        return 0.5


def _coherence_reward(text: str, **kwargs) -> float:
    """Simple coherence reward based on text structure."""
    sentences = text.split('.')
    if len(sentences) < 2:
        return 0.0
    
    # Simple heuristic: reward longer, well-structured text
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    return min(1.0, avg_sentence_length / 20.0)


def _safety_reward(text: str, strict: bool = True, **kwargs) -> float:
    """Reward for safety (basic implementation)."""
    # Simple keyword-based safety check
    unsafe_keywords = [
        'harmful', 'dangerous', 'illegal', 'violence', 'hate',
        'discrimination', 'explicit', 'inappropriate'
    ]
    
    text_lower = text.lower()
    unsafe_count = sum(1 for keyword in unsafe_keywords if keyword in text_lower)
    
    if strict:
        return 0.0 if unsafe_count > 0 else 1.0
    else:
        return max(0.0, 1.0 - (unsafe_count * 0.2))


# Built-in task definitions
CHAT_TASK = {
    "description": "Conversational AI task",
    "input_format": "messages",
    "output_format": "response",
    "evaluation_metrics": ["bleu", "rouge", "perplexity"]
}

CLASSIFICATION_TASK = {
    "description": "Text classification task",
    "input_format": "text",
    "output_format": "label",
    "evaluation_metrics": ["accuracy", "f1", "precision", "recall"]
}

SUMMARIZATION_TASK = {
    "description": "Text summarization task",
    "input_format": "document",
    "output_format": "summary",
    "evaluation_metrics": ["rouge", "bleu", "bertscore"]
}


# Initialize built-in components
def _initialize_builtins():
    """Initialize built-in dataset loaders, adapters, rewards, and tasks."""
    
    # Register dataset loaders
    DatasetRegistry.register_loader("huggingface", _load_hf_dataset)
    DatasetRegistry.register_loader("local", _load_local_dataset)
    
    # Register schema adapters
    DatasetRegistry.register_adapter("chat", _adapt_chat_schema)
    DatasetRegistry.register_adapter("hh-rlhf", _adapt_hh_rlhf_schema)
    DatasetRegistry.register_adapter("ultrafeedback", _adapt_ultrafeedback_schema)
    DatasetRegistry.register_adapter("alpaca", _adapt_alpaca_schema)
    DatasetRegistry.register_adapter("qa", _adapt_qa_schema)
    
    # Register reward functions
    RewardRegistry.register("length", _length_reward)
    RewardRegistry.register("sentiment", _sentiment_reward)
    RewardRegistry.register("coherence", _coherence_reward)
    RewardRegistry.register("safety", _safety_reward)
    
    # Register tasks
    TaskRegistry.register("conversation", CHAT_TASK)
    TaskRegistry.register("classification", CLASSIFICATION_TASK)
    TaskRegistry.register("summarization", SUMMARIZATION_TASK)
    
    logger.info("Initialized built-in registries")


# Initialize built-ins when module is imported
_initialize_builtins()