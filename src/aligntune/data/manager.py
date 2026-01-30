from typing import Optional, Dict, Union, Any, Callable
from datasets import Dataset, DatasetDict
from aligntune.data.loaders.resolver import LoaderResolver
from aligntune.data.schemas import TaskType
from aligntune.data.processors import (
    ColumnMapper, 
    SystemPromptInjector, 
    SplitGenerator, 
    CustomProcessor,
    preprocess_hh_rlhf,
    preprocess_ultrafeedback,
    preprocess_ultrachat,
    preprocess_human_like_dpo,
    preprocess_ultrafeedback_binarized,
    preprocess_mbpp,
    preprocess_humaneval,
)

class DataManager:
    """
    The High-Level Orchestrator with Auto-Detection + Custom Processing + Tokenizer Support.
    
    Logic:
    - If user provides processing_fn → use it
    - Else → auto-detect based on dataset name/format and apply built-in processor
    - Apply system prompt with chat template if tokenizer provided
    """
    
    # Dataset name patterns → processor mapping
    DATASET_PATTERNS = {
        'ultrachat': preprocess_ultrachat,
        'hh-rlhf': preprocess_hh_rlhf,
        'anthropic/hh': preprocess_hh_rlhf,
        'ultrafeedback': preprocess_ultrafeedback,
        'human-like-dpo': preprocess_human_like_dpo,
        'ultrafeedback_binarized': preprocess_ultrafeedback_binarized,
        # NEW: Code dataset patterns
        'mbpp': preprocess_mbpp,
        'humaneval': preprocess_humaneval,
        'human_eval': preprocess_humaneval,
        'openai_humaneval': preprocess_humaneval,
    }
    
    def __init__(
        self,
        task_type: Union[str, TaskType],
        column_mapping: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None,
        tokenizer=None,  # NEW: Add tokenizer parameter
        enable_thinking: bool = False,  # NEW: Add enable_thinking parameter
        processing_fn: Optional[Callable] = None,
        processing_batched: bool = False,
        val_split_ratio: float = 0.1,
        seed: int = 42,
        max_samples: int = None ,
        auto_detect: bool = True,  # Enable/disable auto-detection
    ):
        self.task_type = TaskType(task_type)
        self.user_processing_fn = processing_fn  # Store user function
        self.processing_batched = processing_batched
        self.auto_detect = auto_detect
        self.tokenizer = tokenizer  # NEW: Store tokenizer
        self.enable_thinking = enable_thinking  # NEW: Store enable_thinking
        
        # Initialize Processors Pipeline
        self.mapper = ColumnMapper(self.task_type, column_mapping)
        
        # NEW: Pass tokenizer and enable_thinking to SystemPromptInjector
        self.injector = SystemPromptInjector(
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking
        )
        
        self.splitter = SplitGenerator(val_split_ratio, seed,max_samples)
    
    def _detect_processor(self, dataset_name: str) -> Optional[Callable]:
        """Auto-detect processor based on dataset name."""
        if not self.auto_detect:
            return None
        
        dataset_name_lower = dataset_name.lower()
        
        # Sort patterns by length (longest first) to match specific patterns before general ones
        sorted_patterns = sorted(
            self.DATASET_PATTERNS.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for pattern, processor in sorted_patterns:
            if pattern in dataset_name_lower:
                print(f"🔍 Auto-detected dataset format: '{pattern}' in '{dataset_name}'")
                print(f"📝 Using built-in processor: {processor.__name__}")
                # Create a wrapper that passes the target task type
                def wrapper(example):
                    return processor(example, target_task=self.task_type.value)
                return wrapper
                
        return None
    
    def _detect_from_columns(self, dataset: Dataset) -> Optional[Callable]:
        """Auto-detect processor based on dataset columns."""
        if not self.auto_detect:
            return None
        
        columns = set(dataset.column_names)
        
        # Check for specific dataset formats
        if 'messages' in columns:
            first_example = dataset[0]
            messages = first_example.get('messages')
            
            if isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], dict) and 'role' in messages[0]:
                    print("🔍 Auto-detected chat format in 'messages' column")
                    print("📝 Using preprocess_ultrachat")
                    
                    def wrapper(example):
                        return preprocess_ultrachat(example, target_task=self.task_type.value)
                    
                    return wrapper
        
        # Check for HH-RLHF format
        if 'chosen' in columns and 'rejected' in columns:
            first_example = dataset[0]
            chosen = first_example.get('chosen', '')
            
            if isinstance(chosen, str) and '\n\nAssistant:' in chosen:
                print("🔍 Auto-detected HH-RLHF conversational format")
                print("📝 Using preprocess_hh_rlhf")
                
                def wrapper(example):
                    return preprocess_hh_rlhf(example, target_task=self.task_type.value)
                
                return wrapper
        
        # Check for UltraFeedback format
        if 'instruction' in columns and 'completions' in columns:
            print("🔍 Auto-detected UltraFeedback format")
            print("📝 Using preprocess_ultrafeedback")
            
            def wrapper(example):
                return preprocess_ultrafeedback(example, target_task=self.task_type.value)
            
            return wrapper
        
        return None
    
    def load_dataset(self, dataset_name_or_path: str, **loader_kwargs) -> DatasetDict:
        # 1. Load
        loader = LoaderResolver.resolve(dataset_name_or_path, **loader_kwargs)
        raw_data = loader.load()
        
        if isinstance(raw_data, Dataset):
            raw_data = DatasetDict({"train": raw_data})
        
        # 2. Determine which processing function to use
        # Priority: user_processing_fn > auto-detect from name > auto-detect from columns
        processing_fn = self.user_processing_fn
        detected_from_name = None


        if processing_fn is not None:
            processed_splits = {}
            for split_name, dataset in raw_data.items():
                custom_processor = CustomProcessor(processing_fn, self.processing_batched)
                dataset = custom_processor.process(dataset)
                processed_splits[split_name] = dataset
            dataset_dict = DatasetDict(processed_splits)
            dataset_dict = self.splitter.process(dataset_dict)
            return dataset_dict
        
        if processing_fn is None:
            # Try auto-detection from dataset name
            detected_from_name = self._detect_processor(dataset_name_or_path)
            processing_fn = detected_from_name
        
        # 3. Process Pipeline
        processed_splits = {}
        for split_name, dataset in raw_data.items():
            print(f"\n{'='*60}")
            print(f"Processing {split_name} split ({len(dataset)} examples)")
            print(f"{'='*60}")
            
            # If no processor yet, try detecting from columns
            if processing_fn is None and detected_from_name is None:
                processing_fn = self._detect_from_columns(dataset)
            
            # A. Apply auto-detected or user processing (BEFORE column mapping)
            if processing_fn is not None:
                custom_processor = CustomProcessor(processing_fn, self.processing_batched)
                dataset = custom_processor.process(dataset)
                
                # Clean up messages column if it exists
                if 'messages' in dataset.column_names:
                    dataset = dataset.remove_columns(['messages'])
                    print("✅ Removed 'messages' column")
            
            # B. Normalize Columns
            dataset = self.mapper.process(dataset)
            
            # C. Inject System Prompt WITH CHAT TEMPLATE (if tokenizer provided)
            dataset = self.injector.process(dataset)
            
            print(f"Final columns: {dataset.column_names}")
            processed_splits[split_name] = dataset
        
        dataset_dict = DatasetDict(processed_splits)
        
        # 4. Create Splits
        dataset_dict = self.splitter.process(dataset_dict)
        
        return dataset_dict