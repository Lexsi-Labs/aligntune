import logging
from typing import Dict, Optional, List, Any, Callable
from datasets import Dataset, DatasetDict
from aligntune.data.schemas import TaskType, TASK_SCHEMAS

logger = logging.getLogger(__name__)

class ColumnMapper:
    """Normalizes raw column names to the Task Schema."""
    
    def __init__(self, task_type: TaskType, user_mapping: Optional[Dict[str, str]] = None):
        self.task_type = TaskType(task_type)
        self.schema = TASK_SCHEMAS.get(self.task_type)
        self.user_mapping = user_mapping or {}

    def process(self, dataset: Dataset) -> Dataset:
        if not self.schema:
            return dataset

        current_cols = dataset.column_names
        rename_map = {}

        # 1. Apply User Mapping
        for src, tgt in self.user_mapping.items():
            # --- SAFETY CHECK 1: CODE TASKS ---
            # If the user maps a column to 'response' or 'completion' in a CODE task,
            # it will shadow 'test_cases' in the Evaluator, breaking Pass@K.
            # We explicitly ignore such mappings here.
            if self.task_type == TaskType.CODE and tgt in ['response', 'completion']:
                logger.warning(f"⚠️ CODE TASK SAFETY: Ignoring user mapping '{src}' -> '{tgt}'. "
                               f"This prevents masking 'test_cases' with reference code strings.")
                continue
            
            # --- SAFETY CHECK 2: Existing Columns ---
            # If the target column already exists (e.g., created by a preprocessor like preprocess_mbpp),
            # do not attempt to rename, as it might overwrite the preprocessed data with raw data.
            if tgt in current_cols:
                logger.debug(f"ℹ️ Target column '{tgt}' already exists. Skipping mapping from '{src}'.")
                continue

            if src in current_cols:
                rename_map[src] = tgt

        # 2. Apply Heuristics for missing columns
        for target in self.schema.required_columns:
            if target in current_cols or target in rename_map.values():
                continue

            possible_names = self.schema.column_heuristics.get(target, [])
            for synonym in possible_names:
                if synonym in current_cols and synonym not in rename_map:
                    rename_map[synonym] = target
                    break
        
        if rename_map:
            logger.info(f"Mapping columns: {rename_map}")
            dataset = dataset.rename_columns(rename_map)

        # 3. Validation
        missing = [c for c in self.schema.required_columns if c not in dataset.column_names]
        
        if self.schema == TASK_SCHEMAS[TaskType.SFT] and "messages" in dataset.column_names:
            missing = []

        if missing:
            logger.warning(f"Dataset missing required columns {missing} for task. "
                           f"Found: {dataset.column_names}. ")
            
        return dataset


class SystemPromptInjector:
    """Injects system prompts into text prompts or chat structures with tokenizer support."""
    
    def __init__(self, system_prompt: Optional[str], tokenizer=None, enable_thinking: bool = False):
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        
        # Debug logging
        if system_prompt:
            logger.info(f"SystemPromptInjector initialized with system prompt: {system_prompt[:50]}...")
            if tokenizer is not None:
                logger.info(f"✅ Tokenizer provided: {type(tokenizer).__name__}")
                logger.info(f"   Has apply_chat_template: {hasattr(tokenizer, 'apply_chat_template')}")
                if hasattr(tokenizer, 'chat_template'):
                    has_template = tokenizer.chat_template is not None
                    logger.info(f"   Chat template exists: {has_template}")
            else:
                logger.warning("⚠️ No tokenizer provided to SystemPromptInjector")

    def _apply_chat_template(self, messages):
        """Helper to apply chat template with fallbacks."""
        if not self.tokenizer or not hasattr(self.tokenizer, 'apply_chat_template'):
            return None
        
        # Try with enable_thinking first
        try:
            result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            logger.debug("✅ Chat template applied (with enable_thinking)")
            return result
        except TypeError:
            # enable_thinking not supported, try without it
            pass
        except Exception as e:
            logger.warning(f"⚠️ Chat template failed with enable_thinking: {e}")
            return None
        
        # Try without enable_thinking
        try:
            result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug("✅ Chat template applied (without enable_thinking)")
            return result
        except Exception as e:
            logger.warning(f"⚠️ Chat template failed: {e}")
            return None

    def process(self, dataset: Dataset) -> Dataset:
        if not self.system_prompt:
            logger.info("No system prompt provided, skipping injection")
            return dataset

        def _looks_like_chat_templated(text: str) -> bool:
            """Heuristic: detect if `prompt` is already chat-templated."""
            if not text:
                return False
            markers = (
                "<|begin_of_text|>",  # Llama 3+
                "<|start_header_id|>",  # Llama 3+
                "<|eot_id|>",  # Llama 3+
                "<|im_start|>",  # ChatML
                "<|im_end|>",  # ChatML
                "<<SYS>>",  # Llama 2 style
                "[INST]",  # Llama 2 style
            )
            return any(m in text for m in markers)

        def _inject_into_existing_template(prompt_text: str) -> Optional[str]:
            """
            If `prompt_text` is already templated, avoid wrapping it again.
            Instead, try to insert the system prompt into the existing system section.
            """
            if self.system_prompt in prompt_text:
                return prompt_text

            # Llama 3 template: insert before first <|eot_id|> (end of system message)
            sys_hdr = "<|start_header_id|>system<|end_header_id|>"
            eot = "<|eot_id|>"
            if sys_hdr in prompt_text and eot in prompt_text:
                eot_idx = prompt_text.find(eot)
                if eot_idx != -1:
                    return prompt_text[:eot_idx] + "\n" + self.system_prompt + "\n" + prompt_text[eot_idx:]

            # ChatML template: insert before first <|im_end|> (end of system message)
            chatml_sys = "<|im_start|>system"
            im_end = "<|im_end|>"
            if chatml_sys in prompt_text and im_end in prompt_text:
                end_idx = prompt_text.find(im_end)
                if end_idx != -1:
                    return prompt_text[:end_idx] + "\n" + self.system_prompt + "\n" + prompt_text[end_idx:]

            # Unknown template - safest is to return None (skip modification)
            return None

        def _inject(example):
            # Case 1: Chat messages format
            if "messages" in example and isinstance(example["messages"], list):
                logger.debug("Processing messages format")
                msgs = example["messages"]
                new_msgs = list(msgs)
                
                # Add system prompt if not present
                if new_msgs and new_msgs[0].get("role") != "system":
                    new_msgs.insert(0, {"role": "system", "content": self.system_prompt})
                
                # Always try to apply chat template
                formatted = self._apply_chat_template(new_msgs)
                if formatted is not None:
                    return {"prompt": formatted, "messages": new_msgs}
                
                # Fallback: return messages only
                logger.debug("Chat template not available, keeping messages format")
                return {"messages": new_msgs}
            
            # Case 2: Prompt format
            if "prompt" in example and isinstance(example["prompt"], str):
                logger.debug("Processing prompt format")
                prompt_text = example["prompt"]

                # Avoid double system prompt injection if prompt is already chat-templated.
                # This can happen if the dataset already contains templated prompts.
                if _looks_like_chat_templated(prompt_text):
                    patched = _inject_into_existing_template(prompt_text)
                    if patched is not None:
                        return {"prompt": patched}
                    return {"prompt": prompt_text}
                
                # Always try to apply chat template
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt_text}
                ]
                
                formatted = self._apply_chat_template(messages)
                if formatted is not None:
                    return {"prompt": formatted}
                
                # Fallback: plain text concatenation
                logger.debug("Chat template not available, using plain text concatenation")
                return {"prompt": f"{self.system_prompt}\n\n{prompt_text}"}
            
            # Case 3: Context format (fallback)
            if "context" in example and isinstance(example["context"], str):
                logger.debug("Processing context format")
                return {"context": f"{self.system_prompt}\n\n{example['context']}"}

            logger.debug("No suitable column found for injection")
            return example

        cols = dataset.column_names
        logger.info(f"Dataset columns before injection: {cols}")
        
        if any(c in cols for c in ["messages", "prompt", "context"]):
            logger.info(f"Injecting system prompt into dataset")
            result = dataset.map(_inject, desc="Injecting system prompt with chat template")
            logger.info(f"Dataset columns after injection: {result.column_names}")
            return result
        
        logger.warning(f"⚠️ No suitable columns found for system prompt injection. Columns: {cols}")
        return dataset

class CustomProcessor:
    """Applies a user-defined custom function to the dataset."""
    
    def __init__(self, func: Optional[Callable] = None, batched: bool = False):
        self.func = func
        self.batched = batched

    def process(self, dataset: Dataset) -> Dataset:
        if not self.func:
            return dataset
        
        logger.info(f"Applying custom processing function (batched={self.batched})...")
        # dataset.map automatically handles the function application
        return dataset.map(self.func, batched=self.batched, desc="Custom Processing")

class SplitGenerator:
    """
    Handles Train/Test/Validation splitting logic using size-based heuristics.
    """
    
    def __init__(self, val_split: float, seed: int = 42, max_samples=None):
        self.val_split = val_split
        self.seed = seed
        self.max_samples = max_samples

    def _apply_max_samples(self, dataset: Dataset, split_name: str) -> Dataset:
        """Apply max_samples limit to a dataset split."""
        if self.max_samples is None:
            return dataset
        
        if len(dataset) > self.max_samples:
            logger.info(f"Limiting {split_name} split from {len(dataset)} to {self.max_samples} samples")
            return dataset.select(range(self.max_samples))
        
        return dataset

    def _normalize_split_names(self, dataset_dict: DatasetDict) -> DatasetDict:
        """
        Normalize various split naming conventions to standard names.
        Maps variations like train_sft, train_gen, valid_sft, etc. to train/validation/test.
        """
        normalized = {}
        
        # Define mapping patterns for different split types
        train_patterns = ['train', 'train_sft', 'train_gen', 'train_prefs', 'training']
        val_patterns = ['validation', 'valid', 'val', 'valid_sft', 'valid_gen', 'dev', 'development']
        test_patterns = ['test', 'test_sft', 'test_gen', 'testing', 'eval', 'evaluation']
        
        # Track which standard splits we've found
        found_train = None
        found_val = None
        found_test = None
        
        for split_name, split_data in dataset_dict.items():
            split_lower = split_name.lower()
            
            # Check if this is a train split variant
            if any(pattern in split_lower for pattern in train_patterns) and found_train is None:
                normalized['train'] = split_data
                found_train = split_name
                logger.info(f"Mapping '{split_name}' -> 'train'")
            
            # Check if this is a validation split variant
            elif any(pattern in split_lower for pattern in val_patterns) and found_val is None:
                normalized['validation'] = split_data
                found_val = split_name
                logger.info(f"Mapping '{split_name}' -> 'validation'")
            
            # Check if this is a test split variant
            elif any(pattern in split_lower for pattern in test_patterns) and found_test is None:
                normalized['test'] = split_data
                found_test = split_name
                logger.info(f"Mapping '{split_name}' -> 'test'")
            
            # Keep other splits as-is
            else:
                normalized[split_name] = split_data
        
        return DatasetDict(normalized)

    def process(self, dataset_dict: DatasetDict) -> DatasetDict:
        # First, normalize split names
        dataset_dict = self._normalize_split_names(dataset_dict)
        
        # Check if dataset already has proper splits (train + at least one of val/test)
        has_train = "train" in dataset_dict
        has_val = "validation" in dataset_dict
        has_test = "test" in dataset_dict
        
        # If dataset already has train and at least one evaluation split, keep it as is
        if has_train and (has_val or has_test):
            logger.info("Dataset already has splits (train + validation/test). Keeping existing splits.")
            
            # Apply max_samples to existing splits if specified
            if self.max_samples is not None:
                new_splits = {}
                for split_name, split_data in dataset_dict.items():
                    new_splits[split_name] = self._apply_max_samples(split_data, split_name)
                return DatasetDict(new_splits)
            
            return dataset_dict
        
        # Sort splits by size (descending: Largest -> Smallest)
        sorted_splits = sorted(dataset_dict.items(), key=lambda item: len(item[1]), reverse=True)
        num_splits = len(sorted_splits)
        
        if num_splits == 0:
            return dataset_dict

        new_splits = {}
        
        try:
            # === CASE 1: Only 1 Split (Train) ===
            if num_splits == 1:
                # Logic: Create 3 splits (8:1:1 approx)
                main_ds = sorted_splits[0][1]
                
                # Apply max_samples to the original dataset before splitting
                if self.max_samples is not None and len(main_ds) > self.max_samples:
                    logger.info(f"Limiting dataset from {len(main_ds)} to {self.max_samples} before splitting")
                    main_ds = main_ds.select(range(self.max_samples))
                
                total_len = len(main_ds)
                
                if total_len < 3:
                    logger.warning("Dataset too small to split. Returning as train.")
                    return DatasetDict({"train": main_ds})

                logger.info(f"Single split found ({total_len} samples). Generating Train/Val/Test (80/10/10).")
                
                # 1. Peel off Test Set (10%)
                split1 = main_ds.train_test_split(test_size=self.val_split, seed=self.seed)
                test_ds = split1["test"]
                remaining_ds = split1["train"]
                
                # 2. Peel off Validation Set
                val_ratio = self.val_split / (1.0 - self.val_split)
                split2 = remaining_ds.train_test_split(test_size=val_ratio, seed=self.seed)
                
                new_splits["train"] = split2["train"]
                new_splits["validation"] = split2["test"]
                new_splits["test"] = test_ds

            # === CASE 2: 2 Splits (Train + Test/Other) ===
            elif num_splits == 2:
                train_ds = sorted_splits[0][1] # Largest
                test_ds = sorted_splits[1][1]  # Smallest
                
                logger.info(f"Two splits found. Largest ({len(train_ds)})->Train, Smallest ({len(test_ds)})->Test.")
                
                # Apply max_samples before creating validation split
                if self.max_samples is not None and len(train_ds) > self.max_samples:
                    logger.info(f"Limiting train split from {len(train_ds)} to {self.max_samples} before validation split")
                    train_ds = train_ds.select(range(self.max_samples))
                
                # Create Validation from Train
                if len(train_ds) > 1:
                    logger.info("Splitting Train to create Validation.")
                    split = train_ds.train_test_split(test_size=self.val_split, seed=self.seed)
                    new_splits["train"] = split["train"]
                    new_splits["validation"] = split["test"]
                else:
                    new_splits["train"] = train_ds
                    logger.warning("Train set too small to split validation.")
                
                # Apply max_samples to test split
                new_splits["test"] = self._apply_max_samples(test_ds, "test")

            # === CASE 3: 3+ Splits ===
            else:
                new_splits["train"] = self._apply_max_samples(sorted_splits[0][1], "train")      # Largest
                new_splits["validation"] = self._apply_max_samples(sorted_splits[-1][1], "validation") # Smallest
                new_splits["test"] = self._apply_max_samples(sorted_splits[-2][1], "test")       # 2nd Smallest
                
                logger.info(f"3+ splits found. Assigned by size: "
                            f"Train={len(new_splits['train'])}, "
                            f"Test={len(new_splits['test'])}, "
                            f"Validation={len(new_splits['validation'])}")

        except Exception as e:
            logger.error(f"Error during size-based splitting: {e}. Returning original.")
            return dataset_dict
        
        return DatasetDict(new_splits)# ============================================================================
# DATASET-SPECIFIC PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_hh_rlhf(example: Dict[str, Any], target_task: str = "dpo") -> Dict[str, Any]:
    """Preprocess Anthropic/hh-rlhf dataset."""
    result = {}
    chosen_text = example.get('chosen', '')
    rejected_text = example.get('rejected', '')
    
    if '\n\nAssistant:' in chosen_text:
        prompt = chosen_text.split('\n\nAssistant:')[0] + '\n\nAssistant:'
    else:
        prompt = chosen_text
    
    if target_task == "dpo":
        chosen_response = chosen_text.split('\n\nAssistant:')[-1].strip() if '\n\nAssistant:' in chosen_text else chosen_text
        rejected_response = rejected_text.split('\n\nAssistant:')[-1].strip() if '\n\nAssistant:' in rejected_text else rejected_text
        result['prompt'] = prompt.strip()
        result['chosen'] = chosen_response
        result['rejected'] = rejected_response
    elif target_task == "sft":
        chosen_response = chosen_text.split('\n\nAssistant:')[-1].strip() if '\n\nAssistant:' in chosen_text else chosen_text
        result['prompt'] = prompt.strip()
        result['completion'] = chosen_response
    else:
        chosen_response = chosen_text.split('\n\nAssistant:')[-1].strip() if '\n\nAssistant:' in chosen_text else chosen_text
        result['prompt'] = prompt.strip()
        result['response'] = chosen_response
    return result


def preprocess_ultrafeedback(example: Dict[str, Any], target_task: str = "dpo") -> Dict[str, Any]:
    """Preprocess openbmb/UltraFeedback dataset."""
    result = {}
    prompt = example.get('instruction', '').strip()
    completions = example.get('completions', [])
    if not completions:
        return {'prompt': prompt, 'chosen': '', 'rejected': ''}
    
    sorted_completions = sorted(completions, key=lambda x: x.get('overall', 0), reverse=True)
    
    if target_task == "dpo":
        result['prompt'] = prompt
        result['chosen'] = sorted_completions[0].get('response', '').strip() if len(sorted_completions) > 0 else ''
        result['rejected'] = sorted_completions[-1].get('response', '').strip() if len(sorted_completions) > 1 else ''
    elif target_task == "sft":
        result['prompt'] = prompt
        result['completion'] = sorted_completions[0].get('response', '').strip()
    else:
        result['prompt'] = prompt
        result['response'] = sorted_completions[0].get('response', '').strip()
    return result


def preprocess_ultrachat(example: Dict[str, Any], target_task: str = "sft") -> Dict[str, Any]:
    """Preprocess HuggingFaceH4/ultrachat_200k dataset."""
    result = {}
    messages = example.get('messages', [])
    if not messages:
        return {'prompt': '', 'completion': ''}
    
    user_messages = [msg['content'] for msg in messages if msg.get('role') == 'user']
    assistant_messages = [msg['content'] for msg in messages if msg.get('role') == 'assistant']
    
    if target_task == "sft":
        result['prompt'] = user_messages[-1].strip() if user_messages else ''
        result['completion'] = assistant_messages[-1].strip() if assistant_messages else ''
    elif target_task == "dpo":
        result['prompt'] = user_messages[-1].strip() if user_messages else ''
        result['chosen'] = assistant_messages[-1].strip() if assistant_messages else ''
        result['rejected'] = ''
    else:
        result['prompt'] = user_messages[-1].strip() if user_messages else ''
        result['response'] = assistant_messages[-1].strip() if assistant_messages else ''
    return result


def preprocess_human_like_dpo(example: Dict[str, Any], target_task: str = "dpo") -> Dict[str, Any]:
    """Preprocess HumanLLMs/Human-Like-DPO-Dataset."""
    result = {}
    if target_task == "dpo":
        result['prompt'] = example.get('prompt', '').strip()
        result['chosen'] = example.get('chosen', '').strip()
        result['rejected'] = example.get('rejected', '').strip()
    elif target_task == "sft":
        result['prompt'] = example.get('prompt', '').strip()
        result['completion'] = example.get('chosen', '').strip()
    else:
        result['prompt'] = example.get('prompt', '').strip()
        result['response'] = example.get('chosen', '').strip()
    return result


def preprocess_ultrafeedback_binarized(example: Dict[str, Any], target_task: str = "grpo") -> Dict[str, Any]:
    """Preprocess HuggingFaceH4/ultrafeedback_binarized dataset for GRPO."""
    result = {}
    
    # Extract the actual user question from the prompt
    raw_prompt = example.get('prompt', '').strip()
    
    # Extract just the question part (after "Q: " and before " \nA:")
    if '\nQ: ' in raw_prompt and '\nA:' in raw_prompt:
        question_start = raw_prompt.find('\nQ: ') + 4
        question_end = raw_prompt.find('\nA:', question_start)
        prompt_text = raw_prompt[question_start:question_end].strip()
    else:
        prompt_text = raw_prompt
    
    # Extract chosen response
    chosen = example.get('chosen', [])
    if isinstance(chosen, list) and len(chosen) > 0:
        chosen_text = chosen[-1].get('content', '') if isinstance(chosen[-1], dict) else ''
    elif isinstance(chosen, str):
        chosen_text = chosen
    else:
        chosen_text = ''
    
    # Extract rejected response
    rejected = example.get('rejected', [])
    if isinstance(rejected, list) and len(rejected) > 0:
        rejected_text = rejected[-1].get('content', '') if isinstance(rejected[-1], dict) else ''
    elif isinstance(rejected, str):
        rejected_text = rejected
    else:
        rejected_text = ''
    
    # CRITICAL: Format as conversation list for GRPO
    # GRPO expects: prompt = [{"role": "user", "content": "..."}]
    result['prompt'] = [{"role": "user", "content": prompt_text}]
    
    # Keep other fields as needed
    if 'prompt_id' in example:
        result['prompt_id'] = example['prompt_id']
    
    # Note: Don't include 'chosen', 'rejected', 'response' for GRPO
    # GRPO generates its own completions
    
    return result

# ============================================================================
# CODE DATASET PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_mbpp(example: Dict[str, Any], target_task: str = "code") -> Dict[str, Any]:
    """
    Preprocess MBPP.
    FIX: For 'code' task, avoid returning 'completion' or mapping 'code' to 'response'
    so the evaluator can correctly identify 'test_cases' as the target column.
    """
    result = {}
    description = example.get('text', '').strip()
    test_list = example.get('test_list', [])
    reference_code = example.get('code', '').strip()
    
    if target_task == "code":       
        formatted_prompt = (
            f"Complete the following Python function:\n\n"
            f"{description}\n\n"
            f"Provide only the function implementation in Python."
        )

        result['prompt'] = formatted_prompt
        result['test_cases'] = test_list
        result['reference_code'] = reference_code 
        
        if 'task_id' in example:
            result['task_id'] = example['task_id']
            
        # Match 'correct' version exactly by adding original_text
        if 'text' in example:
            result['original_text'] = example['text']
    
    elif target_task == "sft":
        result['prompt'] = description
        result['completion'] = reference_code
    
    else:
        result['prompt'] = description
        result['response'] = reference_code
    
    return result


def preprocess_humaneval(example: Dict[str, Any], target_task: str = "code") -> Dict[str, Any]:
    """Preprocess HumanEval."""
    result = {}
    prompt = example.get('prompt', '').strip()
    test_code = example.get('test', '')
    canonical_solution = example.get('canonical_solution', '').strip()
    entry_point = example.get('entry_point', '')
    
    if target_task == "code":
        result['prompt'] = prompt
        result['test_cases'] = test_code
        result['reference_code'] = canonical_solution
        result['entry_point'] = entry_point
        
        if 'task_id' in example:
            result['task_id'] = example['task_id']
    
    elif target_task == "sft":
        result['prompt'] = prompt
        result['completion'] = canonical_solution
    
    else:
        result['prompt'] = prompt
        result['response'] = canonical_solution
    
    return result


def preprocess_codex_eval(example: Dict[str, Any], target_task: str = "code") -> Dict[str, Any]:
    """Generic preprocessor for Codex-style evaluation datasets."""
    result = {}
    
    prompt = (
        example.get('prompt') or 
        example.get('text') or 
        example.get('description') or 
        example.get('problem') or
        ''
    ).strip()
    
    test_cases = (
        example.get('test_cases') or
        example.get('tests') or
        example.get('test_list') or
        example.get('test') or
        []
    )
    
    reference = (
        example.get('code') or
        example.get('canonical_solution') or
        example.get('solution') or
        example.get('reference') or
        ''
    ).strip()
    
    if target_task == "code":
        result['prompt'] = prompt
        result['test_cases'] = test_cases
        result['reference_code'] = reference
    elif target_task == "sft":
        result['prompt'] = prompt
        result['completion'] = reference
    else:
        result['prompt'] = prompt
        result['response'] = reference
    
    return result