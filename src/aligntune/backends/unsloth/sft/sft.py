"""
Enhanced Unsloth SFT Backend Implementation with Task Type Support.

This module provides task-aware training using Unsloth optimizations,
supporting multiple task types:
- Instruction Following
- Supervised Fine-Tuning
- Text Generation
- Chat Completion

Note: Classification tasks are not supported by Unsloth backend.
"""

import logging
import time
import yaml
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader

from aligntune.core.sft.trainer_base import SFTTrainerBase
from aligntune.core.sft.config import TaskType
from aligntune.core.sft.evaluator import EnhancedEvaluator
from aligntune.core.dataset_adapters import schema_detector
from aligntune.core.precision_handler import PrecisionHandler
logger = logging.getLogger(__name__)


class TaskFormatter:
    """Handles task-specific dataset formatting."""
    
    @staticmethod
    def format_instruction_following(examples: Dict, dataset_cfg) -> Dict[str, List[str]]:
        """Format dataset for instruction following with structured prompts."""
        texts = []
        for i in range(len(examples[dataset_cfg.instruction_column])):
            instruction = examples[dataset_cfg.instruction_column][i]
            response = examples[dataset_cfg.response_column][i]
            context = examples.get(dataset_cfg.context_column, [""] * len(examples[dataset_cfg.instruction_column]))[i]
            
            if context:
                text = (
                    f"<|im_start|>system\nContext: {context}<|im_end|>\n"
                    f"<|im_start|>user\n{instruction}<|im_end|>\n"
                    f"<|im_start|>assistant\n{response}<|im_end|>"
                )
            else:
                text = (
                    f"<|im_start|>user\n{instruction}<|im_end|>\n"
                    f"<|im_start|>assistant\n{response}<|im_end|>"
                )
            texts.append(text)
        return {"text": texts}
    
    @staticmethod
    def format_supervised_fine_tuning(examples: Dict, dataset_cfg) -> Dict[str, List[str]]:
        """Format dataset for standard supervised fine-tuning."""
        texts = []
        for i in range(len(examples.get("instruction", examples.get("text", [])))):
            instruction = examples.get("instruction", [""])[i] if "instruction" in examples else ""
            input_text = examples.get("input", [""])[i] if "input" in examples else ""
            output = examples.get("response", examples.get("output", [""]))[i]
            
            if instruction and output:
                if input_text:
                    text = (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Input:\n{input_text}\n\n"
                        f"### Response:\n{output}"
                    )
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            else:
                # Fallback to simple text
                text = examples.get("text", [""])[i]
            
            texts.append(text)
        return {"text": texts}
    
    
    @staticmethod
    def format_text_generation(examples: Dict, dataset_cfg) -> Dict[str, List[str]]:
        """Format dataset for text generation (simple text continuation or input-output pairs)."""
        texts = []
        text_field = dataset_cfg.text_column if hasattr(dataset_cfg, 'text_column') else "text"
        
        # Check if we have input-output pairs (e.g., for summarization, Q&A)
        input_field = text_field
        output_field = None
        
        # Special handling for SQuAD format (question + context + answers)
        if "question" in examples and "context" in examples and "answers" in examples:
            questions = examples.get("question", [])
            contexts = examples.get("context", [])
            answers_list = examples.get("answers", [])
            
            for i in range(len(questions)):
                question = questions[i] if i < len(questions) else ""
                context = contexts[i] if i < len(contexts) else ""
                answers = answers_list[i] if i < len(answers_list) else {}
                
                # Extract answer text from answers dict
                answer_text = ""
                if isinstance(answers, dict) and "text" in answers:
                    answer_list = answers["text"]
                    if isinstance(answer_list, list) and len(answer_list) > 0:
                        answer_text = answer_list[0]
                elif isinstance(answers, str):
                    answer_text = answers
                
                if question and context:
                    # Format as: Question + Context -> Answer
                    formatted = f"Question: {question.strip()}\n\nContext: {context.strip()}"
                    if answer_text:
                        formatted += f"\n\nAnswer: {answer_text.strip()}"
                    texts.append(formatted)
            return {"text": texts}
        
        # Look for common output field names
        if "summary" in examples:
            output_field = "summary"
        elif "output" in examples:
            output_field = "output"
        elif "completion" in examples:
            output_field = "completion"
        elif "target" in examples:
            output_field = "target"
        
        # If we have both input and output, format as prompt-completion
        if output_field and input_field in examples:
            input_texts = examples.get(input_field, [])
            output_texts = examples.get(output_field, [])
            
            for i in range(len(input_texts)):
                input_text = input_texts[i] if i < len(input_texts) else ""
                output_text = output_texts[i] if i < len(output_texts) else ""
                
                if input_text and isinstance(input_text, str) and input_text.strip():
                    if output_text and isinstance(output_text, str) and output_text.strip():
                        # Format as: input -> output (for training)
                        formatted = f"{input_text.strip()}\n\nSummary: {output_text.strip()}"
                    else:
                        # Only input available
                        formatted = input_text.strip()
                    texts.append(formatted)
        else:
            # Simple text continuation (original behavior)
            for text in examples.get(input_field, []):
                # Only add non-empty text
                if text and isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        
        # Ensure we always return something, even if all were empty
        if not texts:
            texts = ["[NO_VALID_TEXT]"]
        
        return {"text": texts}
    
    @staticmethod
    def format_chat_completion(examples: Dict, dataset_cfg) -> Dict[str, List[str]]:
        """Format dataset for chat completion with conversation format."""
        texts = []
        messages_field = dataset_cfg.messages_column if hasattr(dataset_cfg, 'messages_column') else "messages"
        
        for messages in examples.get(messages_field, []):
            if isinstance(messages, list):
                # Format as conversation
                conversation = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conversation.append(f"{role}: {content}")
                texts.append("\n".join(conversation))
            else:
                # Fallback to simple text
                texts.append(str(messages) if messages else "")
        return {"text": texts}


class UnslothSFTTrainer(SFTTrainerBase):
    """Enhanced SFT trainer using Unsloth with task type support."""
    
    # Supported task types for Unsloth
    SUPPORTED_TASKS = [
        TaskType.INSTRUCTION_FOLLOWING,
        TaskType.SUPERVISED_FINE_TUNING,
        TaskType.TEXT_GENERATION,
        TaskType.CHAT_COMPLETION
    ]
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task_type = self._get_task_type()
        self.training_config = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.training_history = []
        self.logging_manager = None
        self.evaluator = None
        self.unsloth_model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.eval_dataset = None  # Already exists - no need to add
        self.custom_evaluator = None  # ADD THIS LINE (for BaseEvaluator)
        
        # Validate task type
        self._validate_task_type()
        
        logger.info(f"Initialized UnslothSFTTrainer for task: {self.task_type.value}")
    
    def _get_task_type(self) -> TaskType:
        """Extract task type from config."""
        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'task_type'):
            task_type = self.config.dataset.task_type
        elif hasattr(self.config, 'train') and hasattr(self.config.train, 'task_type'):
            task_type = self.config.train.task_type
        else:
            # Default to supervised fine-tuning
            task_type = TaskType.SUPERVISED_FINE_TUNING
        
        # Convert string to enum if needed
        if isinstance(task_type, str):
            task_type = TaskType(task_type.lower())
        
        return task_type
    
    def _validate_task_type(self):
        """Validate that the task type is supported by Unsloth."""
        if self.task_type not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Task type {self.task_type.value} is not supported by Unsloth backend. "
                f"Supported tasks: {[t.value for t in self.SUPPORTED_TASKS]}. "
                f"Use TRL backend for classification tasks."
            )
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth is available."""
        try:
            import unsloth
            from unsloth import FastLanguageModel
            from trl import SFTTrainer, SFTConfig, ModelConfig
            return True
        except ImportError:
            return False
    
    def setup_model(self) -> None:
        """Setup Unsloth-optimized model with task-aware configuration."""
        try:
            import unsloth
            from unsloth import FastLanguageModel
            from trl import SFTConfig, ModelConfig
            from transformers import AutoTokenizer
            
            logger.info(f"Setting up Unsloth model for task: {self.task_type.value}")
            logger.info(f"Model: {self.config.model.name_or_path}")
            
            # Configure Unsloth model parameters
            # === UNIFIED PRECISION HANDLING ===
            precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
            precision = PrecisionHandler.validate_precision(precision)
            PrecisionHandler.log_precision_info(precision, "Unsloth SFT")
            dtype = PrecisionHandler.get_torch_dtype(precision)

            # Configure Unsloth model parameters
            # Unsloth only accepts load_in_4bit as boolean, not BitsAndBytesConfig parameters
            model_kwargs = {
                "max_seq_length": self.config.model.max_seq_length,
                "dtype": dtype,  # Use dtype from PrecisionHandler instead of None
                "load_in_4bit": self.config.model.quantization.get("load_in_4bit", False) if self.config.model.quantization else False,
            }
            
            # Unsloth handles quantization internally - don't pass BitsAndBytesConfig parameters
            # Parameters like bnb_4bit_compute_dtype, bnb_4bit_quant_type, etc. are not supported
            
            # Load model with Unsloth optimizations
            try:
                self.unsloth_model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.config.model.name_or_path,
                    **model_kwargs
                )
            except (SyntaxError, RuntimeError, Exception) as e:
                # Handle Unsloth compiled module syntax errors (may be wrapped in RuntimeError)
                error_msg = str(e)
                if "unsloth_compiled_module" in error_msg or "unexpected indent" in error_msg.lower():
                    logger.error(f"Unsloth compilation error: {e}")
                    logger.info("This is a known Unsloth issue with compiled modules.")
                    logger.info("Suggested fixes:")
                    logger.info("  1. Clear Unsloth cache: rm -rf ~/.cache/unsloth/")
                    logger.info("  2. Retry training after clearing cache")
                    logger.info("  3. Use TRL backend instead: backend='trl'")
                    raise RuntimeError(
                        f"Unsloth compiled module syntax error: {e}\n"
                        f"This is a known Unsloth library issue. "
                        f"Please clear the Unsloth cache and retry, or use TRL backend instead."
                    ) from e
                else:
                    # Re-raise if it's a different error
                    raise
            
            # Detect model architecture for appropriate target modules
            model_type = self.unsloth_model.config.model_type.lower()
            target_modules = self._get_target_modules_for_architecture(model_type)
            
            # Validate target modules exist in the model
            available_modules = [name for name, _ in self.unsloth_model.named_modules()]
            valid_target_modules = [tm for tm in target_modules if any(tm in name for name in available_modules)]
            
            if not valid_target_modules:
                # Try to auto-detect target modules
                logger.warning(f"None of the target modules {target_modules} found in model. Attempting auto-detection...")
                valid_target_modules = self._auto_detect_target_modules(available_modules, model_type)
            
            if not valid_target_modules:
                raise ValueError(
                    f"Target modules {target_modules} not found in the base model (type: {model_type}). "
                    f"Available modules include: {available_modules[:10]}... "
                    f"Please check the target modules and try again, or use a different model architecture."
                )
            
            logger.info(f"Using target modules for {model_type}: {valid_target_modules}")
            
            # Configure model for training with LoRA
            self.unsloth_model = FastLanguageModel.get_peft_model(
                self.unsloth_model,
                r=16,  # LoRA rank
                target_modules=valid_target_modules,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            
            # Setup tokenizer based on task type
            self._setup_tokenizer_for_task()
            
            logger.info("Unsloth model setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Unsloth model: {e}")
            raise
    
    def _get_target_modules_for_architecture(self, model_type: str) -> List[str]:
        """Get appropriate target modules based on model architecture."""
        if "qwen" in model_type or "qwen2" in model_type:
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        elif "llama" in model_type:
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in model_type:
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        elif "phi" in model_type:
            return ["q_proj", "k_proj", "v_proj", "dense"]
        elif "gemma" in model_type:
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        elif "gpt2" in model_type or "gpt" in model_type or "dialogpt" in model_type:
            # GPT-2 and DialoGPT use different module names
            return ["c_attn", "c_proj", "c_fc"]
        else:
            # Try to auto-detect by checking model structure
            # Default to common attention modules, but will be validated
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def _auto_detect_target_modules(self, available_modules: List[str], model_type: str) -> List[str]:
        """Auto-detect target modules from available modules."""
        # Common patterns for attention modules
        patterns = [
            # GPT-2 style
            ("c_attn", "c_proj"),
            # LLaMA/Qwen style
            ("q_proj", "k_proj", "v_proj", "o_proj"),
            # BERT style
            ("query", "key", "value"),
            # Generic attention
            ("attn", "attention"),
        ]
        
        detected = []
        for pattern_group in patterns:
            matches = [mod for mod in available_modules if any(p in mod.lower() for p in pattern_group)]
            if matches:
                # Get unique module names (remove duplicates)
                unique_matches = list(set([m.split('.')[-1] for m in matches if '.' in m]))
                if unique_matches:
                    detected.extend(unique_matches[:4])  # Limit to 4 modules
                    break
        
        return detected if detected else []
    
    def _setup_tokenizer_for_task(self):
        """Setup tokenizer with task-specific configurations."""
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # If a chat template is explicitly provided in config, try to apply it
        # so `tokenizer.apply_chat_template(...)` works across SFT task types.
        try:
            from unsloth import FastLanguageModel
            cfg_template = getattr(self.config.dataset, "chat_template", None)
            has_template = hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None

            def _normalize_template_name(name: Optional[str]) -> Optional[str]:
                if not name:
                    return None
                v = str(name).strip().lower()
                if v in {"auto", "none"}:
                    return None
                # common aliases used in configs
                if v in {"llama3", "llama_3", "llama-3"}:
                    return "llama-3"
                if v in {"llama2", "llama_2", "llama-2"}:
                    return "llama-2"
                return name

            normalized = _normalize_template_name(cfg_template)
            if normalized and not has_template:
                mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
                self.tokenizer = FastLanguageModel.get_chat_template(
                    self.tokenizer,
                    chat_template=normalized,
                    mapping=mapping,
                )
                logger.info(f"Applied configured chat template for SFT: {normalized}")
        except Exception as e:
            logger.debug(f"Skipping configured chat template application: {e}")
        
        # Add chat template for chat completion tasks
        if self.task_type == TaskType.CHAT_COMPLETION:
            logger.info("Applying Unsloth chat template...")
            from unsloth import FastLanguageModel
            
            # Get chat template from config if available
            chat_template = None
            if hasattr(self.config.dataset, 'chat_template'):
                chat_template = self.config.dataset.chat_template
                
            # Default mapping
            mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
            
            try:
                self.tokenizer = FastLanguageModel.get_chat_template(
                    self.tokenizer,
                    chat_template=chat_template if chat_template else "llama-3", # Default to llama-3 if not specified
                    mapping=mapping,
                )
                logger.info(f"Applied chat template: {chat_template if chat_template else 'llama-3 (default)'}")
            except Exception as e:
                logger.warning(f"Failed to apply Unsloth chat template: {e}. Falling back to manual template.")
                if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                    self.tokenizer.chat_template = (
                        "{% for message in messages %}"
                        "{{ message['role'] }}: {{ message['content'] }}\n"
                        "{% endfor %}Assistant:"
                    )
                    logger.info("Added manual chat template to tokenizer")
        
        # Add special tokens for instruction following if needed
        elif self.task_type == TaskType.INSTRUCTION_FOLLOWING:
            has_chat_template = (
                hasattr(self.tokenizer, "apply_chat_template") and
                hasattr(self.tokenizer, "chat_template") and
                self.tokenizer.chat_template is not None
            )
            if not has_chat_template:
                special_tokens = ["<|im_start|>", "<|im_end|>"]
                num_added = self.tokenizer.add_special_tokens({
                    'additional_special_tokens': special_tokens
                })
                if num_added > 0:
                    self.unsloth_model.resize_token_embeddings(len(self.tokenizer))
                    logger.info(f"Added {num_added} special tokens for instruction following")
    
    def _apply_chat_template_safe(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> Optional[str]:
        """Apply tokenizer chat template with broad compatibility."""
        if not self.tokenizer or not hasattr(self.tokenizer, "apply_chat_template"):
            return None
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except TypeError:
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                return None
        except Exception:
            return None

    def _format_with_chat_template(self, examples: Dict[str, Any], ds_cfg) -> Dict[str, List[str]]:
        """
        Build a `text` field using the model's chat template when available.
        Falls back to legacy string formatting if templating fails.
        """
        system_prompt = getattr(ds_cfg, "system_prompt", None)
        texts: List[str] = []

        def _prepend_system_if_missing(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
            if system_prompt and (not msgs or msgs[0].get("role") != "system"):
                return [{"role": "system", "content": system_prompt}] + msgs
            return msgs

        if self.task_type == TaskType.CHAT_COMPLETION:
            messages_field = ds_cfg.messages_column if hasattr(ds_cfg, "messages_column") else "messages"
            for messages in examples.get(messages_field, []):
                if isinstance(messages, list):
                    msgs = _prepend_system_if_missing(messages)
                    templated = self._apply_chat_template_safe(msgs, add_generation_prompt=False)
                    if templated is not None:
                        texts.append(templated)
                        continue
                    conversation = [f"{m.get('role','user')}: {m.get('content','')}" for m in msgs]
                    texts.append("\n".join(conversation))
                else:
                    texts.append(str(messages) if messages else "")
            return {"text": texts}

        if self.task_type in [TaskType.INSTRUCTION_FOLLOWING, TaskType.SUPERVISED_FINE_TUNING]:
            instr_col = getattr(ds_cfg, "instruction_column", "instruction")
            resp_col = getattr(ds_cfg, "response_column", "response")
            ctx_col = getattr(ds_cfg, "context_column", "context")
            input_col = getattr(ds_cfg, "input_column", "input")

            instructions = examples.get(instr_col, examples.get("instruction", []))
            responses = examples.get(resp_col, examples.get("response", examples.get("output", [])))

            n = min(len(instructions), len(responses))
            for i in range(n):
                instruction = instructions[i]
                response = responses[i]

                context = ""
                if ctx_col in examples and i < len(examples.get(ctx_col, [])):
                    context = examples.get(ctx_col, [""])[i] or ""
                input_text = ""
                if input_col in examples and i < len(examples.get(input_col, [])):
                    input_text = examples.get(input_col, [""])[i] or ""

                sys_parts: List[str] = []
                if system_prompt:
                    sys_parts.append(str(system_prompt).strip())
                if context and str(context).strip():
                    sys_parts.append(f"Context: {str(context).strip()}")

                messages: List[Dict[str, str]] = []
                if sys_parts:
                    messages.append({"role": "system", "content": "\n\n".join(sys_parts)})

                user_content = str(instruction) if instruction is not None else ""
                if input_text and str(input_text).strip():
                    user_content = f"{user_content}\n\n{str(input_text).strip()}"
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": str(response) if response is not None else ""})

                templated = self._apply_chat_template_safe(messages, add_generation_prompt=False)
                if templated is not None:
                    texts.append(templated)
                else:
                    # Fallback to legacy formatting (inline)
                    if self.task_type == TaskType.INSTRUCTION_FOLLOWING:
                        if context and str(context).strip():
                            texts.append(
                                f"<|im_start|>system\nContext: {str(context).strip()}<|im_end|>\n"
                                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                                f"<|im_start|>assistant\n{str(response) if response is not None else ''}<|im_end|>"
                            )
                        else:
                            texts.append(
                                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                                f"<|im_start|>assistant\n{str(response) if response is not None else ''}<|im_end|>"
                            )
                    else:
                        texts.append(f"### Instruction:\n{user_content}\n\n### Response:\n{str(response) if response is not None else ''}")

            if not texts:
                texts = ["[NO_VALID_TEXT]"]
            return {"text": texts}

        return {"text": examples.get("text", ["[NO_VALID_TEXT]"])}
    
    def setup_data(self) -> None:
        """Setup and prepare dataset for training (required by abstract base class)."""
        self.setup_dataset()

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step (required by abstract base class)."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_model() first.")
        
        # The actual training step is handled by TRL's SFTTrainer
        return {"loss": 0.0, "learning_rate": 0.0}

    def setup_dataset(self) -> None:
        """Setup and prepare dataset with task-aware formatting."""
        try:
            from datasets import load_dataset

            ds_cfg = self.config.dataset

            logger.info(f"Loading dataset: {ds_cfg.name} for task: {self.task_type.value}")

            # Prepare load_dataset arguments
            load_kwargs = {}
            
            # Add split parameter
            if hasattr(ds_cfg, 'split') and ds_cfg.split:
                load_kwargs['split'] = ds_cfg.split
            else:
                load_kwargs['split'] = "train"
            
            # Handle dataset config/subset (for datasets like wikitext that require it)
            dataset_name = ds_cfg.name
            dataset_config = None
            
            if hasattr(ds_cfg, 'subset') and ds_cfg.subset:
                dataset_config = ds_cfg.subset
                logger.info(f"Using dataset config/subset: {dataset_config}")
            elif hasattr(ds_cfg, 'config') and ds_cfg.config:
                dataset_config = ds_cfg.config
                logger.info(f"Using dataset config: {dataset_config}")
            
            # Load dataset with or without config
            # Special handling for SQuAD (both v1.1 and v2.0) which use deprecated 'List' feature type
            if "squad" in dataset_name.lower():
                logger.info(f"Detected SQuAD dataset. Using workaround for deprecated 'List' feature type...")
                try:
                    # Method 1: Try loading with builder to bypass feature validation
                    from datasets import load_dataset_builder
                    builder = load_dataset_builder(dataset_name, dataset_config if dataset_config else None)
                    
                    # Download and prepare
                    builder.download_and_prepare()
                    
                    # Load from cache, bypassing feature type validation
                    split_name = load_kwargs.get('split', 'train')
                    dataset = builder.as_dataset(split=split_name)
                    
                    logger.info(f"Successfully loaded {dataset_name} using builder workaround")
                except Exception as e_builder:
                    logger.warning(f"Builder method failed: {e_builder}. Trying alternative approach...")
                    try:
                        # Method 2: Try loading with ignore_verifications (if supported)
                        load_kwargs_alt = load_kwargs.copy()
                        load_kwargs_alt['download_mode'] = 'force_redownload'
                        if dataset_config:
                            dataset = load_dataset(dataset_name, dataset_config, **load_kwargs_alt)
                        else:
                            dataset = load_dataset(dataset_name, **load_kwargs_alt)
                        logger.info(f"Successfully loaded {dataset_name} with force_redownload")
                    except Exception as e2:
                        logger.warning(f"Force redownload failed: {e2}. Trying feature patching...")
                        try:
                            # Method 3: Patch Features.from_dict to handle List -> Sequence conversion
                            import datasets
                            import json
                            original_from_dict = datasets.Features.from_dict
                            
                            def patched_from_dict(features_dict):
                                """Patch to convert 'List' to 'Sequence' in feature dict."""
                                # Convert to string, replace, convert back
                                features_str = json.dumps(features_dict)
                                features_str = features_str.replace('"List"', '"Sequence"')
                                features_str = features_str.replace("'List'", "'Sequence'")
                                features_dict = json.loads(features_str)
                                return original_from_dict(features_dict)
                            
                            # Temporarily patch
                            datasets.Features.from_dict = patched_from_dict
                            try:
                                if dataset_config:
                                    dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)
                                else:
                                    dataset = load_dataset(dataset_name, **load_kwargs)
                                logger.info(f"Successfully loaded {dataset_name} with feature patching")
                            finally:
                                # Restore original
                                datasets.Features.from_dict = original_from_dict
                        except Exception as e3:
                            logger.error(f"All SQuAD loading methods failed. Last error: {e3}")
                            raise ValueError(
                                f"Dataset '{dataset_name}' uses deprecated 'List' feature type that cannot be automatically fixed. "
                                f"Please try: pip install --upgrade datasets>=2.14.0, "
                                f"or consider using an alternative Q&A dataset like 'allenai/sciq' or 'deepmind/code_contests'. "
                                f"Error: {str(e3)[:200]}"
                            ) from e3
            else:
                try:
                    if dataset_config:
                        dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)
                    else:
                        dataset = load_dataset(dataset_name, **load_kwargs)
                except (ValueError, TypeError) as e:
                    error_msg = str(e)
                    # Handle deprecated 'List' feature type for other datasets
                    if "Feature type 'List' not found" in error_msg or "'List'" in error_msg:
                        logger.warning(f"Dataset '{dataset_name}' uses deprecated 'List' feature type. "
                                     f"Trying to load with download_mode='force_redownload'...")
                        try:
                            load_kwargs_retry = load_kwargs.copy()
                            load_kwargs_retry['download_mode'] = 'force_redownload'
                            if dataset_config:
                                dataset = load_dataset(dataset_name, dataset_config, **load_kwargs_retry)
                            else:
                                dataset = load_dataset(dataset_name, **load_kwargs_retry)
                            logger.info(f"Successfully loaded dataset with updated features")
                        except Exception as e2:
                            logger.error(f"Failed to load dataset: {e2}")
                            raise ValueError(
                                f"Dataset '{dataset_name}' uses deprecated 'List' feature type. "
                                f"Please try: pip install --upgrade datasets>=2.14.0. "
                                f"Original error: {error_msg[:200]}"
                            ) from e
                    elif "Config name is missing" in error_msg or "config" in error_msg.lower():
                        logger.error(f"Dataset '{dataset_name}' requires a config/subset name. "
                                f"Add 'subset' or 'config' parameter to your DatasetConfig. "
                                f"Error: {error_msg}")
                        raise ValueError(
                            f"Dataset '{dataset_name}' requires a config/subset name. "
                            f"Please add 'subset' or 'config' to your DatasetConfig. "
                            f"Original error: {error_msg}"
                        )
                    else:
                        raise

            logger.info(f"Dataset loaded: {len(dataset)} samples")

            # Apply subset selection
            if hasattr(ds_cfg, 'max_samples') and ds_cfg.max_samples:
                original_len = len(dataset)
                dataset = dataset.select(range(min(ds_cfg.max_samples, len(dataset))))
                logger.info(f"Applied max_samples limit: {original_len} -> {len(dataset)} samples")
            elif hasattr(ds_cfg, 'percent') and ds_cfg.percent:
                original_len = len(dataset)
                max_samples = int(len(dataset) * ds_cfg.percent / 100)
                dataset = dataset.select(range(max_samples))
                logger.info(f"Applied percent limit ({ds_cfg.percent}%): {original_len} -> {len(dataset)} samples")

            # Auto-detect schema if enabled
            if hasattr(ds_cfg, 'auto_detect_fields') and ds_cfg.auto_detect_fields:
                logger.info("Auto-detecting dataset schema...")
                detected_format = schema_detector.detect_format(dataset)
                if detected_format:
                    logger.info(f"Auto-detected dataset format: {detected_format}")
                    if hasattr(ds_cfg, 'format_type'):
                        ds_cfg.format_type = detected_format

            # Apply field mappings
            field_mappings = None
            if hasattr(ds_cfg, 'field_mappings') and ds_cfg.field_mappings:
                field_mappings = ds_cfg.field_mappings
                logger.info(f"Applying field mappings: {field_mappings}")
            elif hasattr(ds_cfg, 'column_mapping') and ds_cfg.column_mapping:
                field_mappings = ds_cfg.column_mapping
                logger.info(f"Applying column mappings: {field_mappings}")
            
            if field_mappings:
                # Only rename columns that exist in the dataset
                valid_mappings = {k: v for k, v in field_mappings.items() if k in dataset.column_names}
                if valid_mappings:
                    dataset = dataset.rename_columns(valid_mappings)
                    logger.info(f"Renamed columns: {valid_mappings}")
                else:
                    logger.warning(f"No valid column mappings found. Available columns: {dataset.column_names}")

            # Log dataset structure
            logger.info(f"Dataset columns: {dataset.column_names}")
            if len(dataset) > 0:
                logger.info(f"Sample data (first item): {list(dataset[0].keys())}")

            # Task-specific formatting
            dataset = self._format_dataset_for_task(dataset, ds_cfg)

            # Split train/test
            # Always create eval split if dataset has at least 2 samples (minimum for split)
            # For very small datasets, use a smaller test_size to ensure eval set has at least 1 sample
            if len(dataset) >= 2:
                if len(dataset) >= 10:
                    test_size = 0.1  # 10% for larger datasets
                elif len(dataset) >= 5:
                    test_size = 0.2  # 20% for medium datasets (ensures at least 1 eval sample)
                else:
                    test_size = 0.2  # 20% for very small datasets (2-4 samples -> at least 1 eval)
                
                split_ds = dataset.train_test_split(test_size=test_size, seed=42)
                self.train_dataset = split_ds["train"]
                self.eval_dataset = split_ds["test"]
                logger.info(f"Split dataset: {len(self.train_dataset)} train, {len(self.eval_dataset)} eval")
            else:
                # Only skip split if dataset has less than 2 samples
                self.train_dataset = dataset
                self.eval_dataset = None
                logger.warning(f"Dataset too small for split (only {len(dataset)} samples), using all for training. No eval dataset created.")

            logger.info(f"Dataset prepared: {len(self.train_dataset)} training samples")
            if self.eval_dataset:
                logger.info(f"Evaluation dataset: {len(self.eval_dataset)} samples")

        except Exception as e:
            logger.error(f"Failed to setup dataset: {e}")
            raise
    
    def _format_dataset_for_task(self, dataset, ds_cfg):
        """Format dataset based on task type."""
        logger.info(f"Formatting dataset for task: {self.task_type.value}")
        
        # Select appropriate formatter
        can_template = self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template")

        if self.task_type == TaskType.INSTRUCTION_FOLLOWING:
            formatter = self._format_with_chat_template if can_template else TaskFormatter.format_instruction_following
        elif self.task_type == TaskType.SUPERVISED_FINE_TUNING:
            formatter = self._format_with_chat_template if can_template else TaskFormatter.format_supervised_fine_tuning
        elif self.task_type == TaskType.TEXT_GENERATION:
            formatter = TaskFormatter.format_text_generation
        elif self.task_type == TaskType.CHAT_COMPLETION:
            formatter = self._format_with_chat_template if can_template else TaskFormatter.format_chat_completion
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # Apply formatting
        try:
            formatted_dataset = dataset.map(
                lambda examples: formatter(examples, ds_cfg),
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Formatting for {self.task_type.value}"
            )
            
            # Verify formatting
            sample = formatted_dataset[0]
            logger.info(f"Sample formatted text (first 200 chars): {sample['text'][:200]}")
            
            return formatted_dataset
        except Exception as e:
            logger.error(f"Error formatting dataset: {e}")
            # Fallback to simple formatting
            logger.warning("Using fallback formatting")
            return self._fallback_format(dataset, ds_cfg)
    
    def _fallback_format(self, dataset, ds_cfg):
        """Fallback formatting when task-specific formatting fails."""
        def simple_format(examples):
            texts = []
            # Try common column names
            for i in range(len(list(examples.values())[0])):
                text = ""
                if "instruction" in examples and "response" in examples:
                    text = f"{examples['instruction'][i]}\n{examples['response'][i]}"
                elif "text" in examples:
                    text = examples["text"][i]
                else:
                    # Use first text-like column
                    first_col = list(examples.keys())[0]
                    text = str(examples[first_col][i])
                
                texts.append(text)
            return {"text": texts}
        
        return dataset.map(
            simple_format,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def setup_trainer(self) -> None:
        """Setup TRL SFTTrainer with task-aware configuration."""
        try:
            from trl import SFTTrainer, SFTConfig, ModelConfig
            
            logger.info(f"Setting up TRL SFTTrainer for task: {self.task_type.value}")
            
            # Get training parameters
            num_epochs = getattr(self.config, 'num_epochs', None) or \
                        getattr(self.config.train, 'epochs', 3) if hasattr(self.config, 'train') else 3
            batch_size = getattr(self.config, 'batch_size', None) or \
                        getattr(self.config.train, 'per_device_batch_size', 2) if hasattr(self.config, 'train') else 2
            grad_accum = getattr(self.config.train, 'gradient_accumulation_steps', 1) if hasattr(self.config, 'train') else 1
            lr = getattr(self.config.train, 'learning_rate', 2e-4) if hasattr(self.config, 'train') else 2e-4
            save_steps = getattr(self.config.train, 'save_interval', 500) if hasattr(self.config, 'train') else 500
            output_dir = getattr(self.config.logging, 'output_dir', './output') if hasattr(self.config, 'logging') else './output'
            # === UNIFIED PRECISION HANDLING ===
            precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
            precision_args = PrecisionHandler.get_training_args_precision(precision)

                # Create SFT configuration
            
            # Create SFT configuration
            self.training_config = SFTConfig(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                logging_steps=10,
                save_steps=save_steps,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                **precision_args,  
                dataloader_pin_memory=False,
                report_to="none" if not hasattr(self.config, 'logging') else "tensorboard"
            )
            
            # Get max_seq_length
            max_seq_len = getattr(self.config.model, 'max_seq_length', 2048) if hasattr(self.config, 'model') else 2048
            
            # Task-specific trainer settings
            packing = False  # Generally disable packing for better quality
            if self.task_type == TaskType.TEXT_GENERATION:
                packing = True  # Can enable packing for simple text generation
            
            # Create trainer
            self.trainer = SFTTrainer(
                model=self.unsloth_model,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=self.training_config,
                dataset_text_field="text",
                max_seq_length=max_seq_len,
                dataset_num_proc=2,
                packing=packing,
            )
            
            logger.info(f"TRL SFTTrainer setup completed for {self.task_type.value}")
            logger.info(f"Training config: {num_epochs} epochs, batch_size={batch_size}, lr={lr}")
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise
    
    def train(self) -> Dict[str, Any]:
        """Execute training with Unsloth optimizations."""
        try:
            logger.info(f"Starting Unsloth SFT training for task: {self.task_type.value}")
            start_time = time.time()
            
            # Setup components
            self.setup_model()
            self.setup_dataset()
            self.setup_trainer()
            
            # Suppress Unsloth's informational warning about num_items_in_batch
            # This is a known limitation with Qwen2 models and gradient accumulation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*num_items_in_batch.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*Qwen2ForCausalLM does not accept.*", category=UserWarning)
                # Start training
                training_result = self.trainer.train()
            
            # Save model
            output_dir = self.config.logging.output_dir if hasattr(self.config, 'logging') else './output'
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            training_time = time.time() - start_time
            
            # Compile results
            results = {
                "task_type": self.task_type.value,
                "training_time": training_time,
                "final_loss": training_result.training_loss,
                "total_steps": training_result.global_step,
                "model_path": output_dir,
                "training_history": self.training_history,
            }
            
            logger.info(f"Unsloth SFT training completed in {training_time:.2f} seconds")
            logger.info(f"Task: {self.task_type.value}, Final loss: {training_result.training_loss:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self,eval_dataset=None) -> Dict[str, Any]:
        """Evaluate the trained model with task-specific metrics."""
        try:
            if not self.eval_dataset:
                logger.warning("No evaluation dataset available")
                return {}
            
            if eval_dataset:
                self.eval_dataset=eval_dataset
            
            logger.info(f"Evaluating Unsloth SFT model for task: {self.task_type.value}")
            
            # Suppress Unsloth's informational warning about num_items_in_batch
            # This is a known limitation with Qwen2 models and gradient accumulation
            # It doesn't affect functionality, just makes gradient accumulation slightly less accurate
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*num_items_in_batch.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*Qwen2ForCausalLM does not accept.*", category=UserWarning)
                # Run basic evaluation from TRL trainer
                eval_results = self.trainer.evaluate()
            
            # Add task-specific metrics
            eval_results["task_type"] = self.task_type.value
            
            # Calculate perplexity from loss (if available)
            if "eval_loss" in eval_results:
                try:
                    eval_results["eval_perplexity"] = float(
                        torch.exp(torch.tensor(float(eval_results["eval_loss"]))).item()
                    )
                except Exception as e:
                    logger.debug(f"Could not calculate perplexity: {e}")
            
            # Add comprehensive quality metrics using SFTEvaluator
            try:
                from aligntune.core.sft.evaluator import SFTEvaluator
                
                if self.unsloth_model and self.tokenizer and self.eval_dataset:
                    evaluator = SFTEvaluator(self.config)
                    quality_metrics = evaluator.evaluate(
                        model=self.unsloth_model,
                        tokenizer=self.tokenizer,
                        dataset=self.eval_dataset,
                        config=self.config
                    )
                    
                    # Merge quality metrics (avoid duplicates)
                    for key, value in quality_metrics.items():
                        if key not in eval_results:  # Don't override existing metrics
                            eval_results[key] = value
                    
                    logger.info(f"Added {len(quality_metrics)} quality metrics")
            except Exception as e:
                logger.warning(f"Could not compute quality metrics: {e}")
            
            logger.info(f"Evaluation results: {len(eval_results)} metrics computed")
            return eval_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def generate_samples(self, num_samples: int = 5) -> List[Dict[str, str]]:
        """Generate task-specific sample outputs."""
        try:
            logger.info(f"Generating {num_samples} samples for task: {self.task_type.value}")
            
            # Get task-specific prompts
            sample_prompts = self._get_sample_prompts_for_task(num_samples)
            
            samples = []
            for i, prompt in enumerate(sample_prompts):
                try:
                    # Tokenize input
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    
                    # Move to device
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = self.unsloth_model.generate(
                            **inputs,
                            max_new_tokens=100,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decode response
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()
                    
                    samples.append({
                        "task_type": self.task_type.value,
                        "prompt": prompt,
                        "response": response,
                        "sample_id": i + 1
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to generate sample {i+1}: {e}")
                    samples.append({
                        "task_type": self.task_type.value,
                        "prompt": prompt,
                        "response": f"Generation failed: {e}",
                        "sample_id": i + 1
                    })
            
            logger.info(f"Generated {len(samples)} samples")
            return samples
            
        except Exception as e:
            logger.error(f"Sample generation failed: {e}")
            return []
    
    def _get_sample_prompts_for_task(self, num_samples: int) -> List[str]:
        """Get task-specific sample prompts for generation."""
        if self.task_type == TaskType.INSTRUCTION_FOLLOWING:
            prompts = [
                "<|im_start|>user\nExplain machine learning.<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nWrite a Python function.<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nDescribe deep learning.<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nHow does training work?<|im_end|>\n<|im_start|>assistant\n"
            ]
        elif self.task_type == TaskType.CHAT_COMPLETION:
            prompts = [
                "Human: Hello!\nAssistant:",
                "Human: How are you?\nAssistant:",
                "Human: Explain quantum computing.\nAssistant:",
                "Human: Tell me a joke.\nAssistant:",
                "Human: What's the weather?\nAssistant:"
            ]
        elif self.task_type == TaskType.TEXT_GENERATION:
            prompts = [
                "The future of AI is",
                "In a world where",
                "The key to success is",
                "Technology has changed",
                "The most important skill is"
            ]
        else:  # SUPERVISED_FINE_TUNING
            prompts = [
                "### Instruction:\nExplain machine learning.\n\n### Response:\n",
                "### Instruction:\nWrite code.\n\n### Response:\n",
                "### Instruction:\nWhat is AI?\n\n### Response:\n",
                "### Instruction:\nDescribe NLP.\n\n### Response:\n",
                "### Instruction:\nHow to train models?\n\n### Response:\n"
            ]
        
        return prompts[:num_samples]
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model with task metadata."""
        try:
            save_path = path or (self.config.logging.output_dir if hasattr(self.config, 'logging') else './output')
            
            logger.info(f"Saving Unsloth model to: {save_path}")
            
            # Save using Unsloth's optimized saving
            self.unsloth_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save training configuration with task type
            config_path = Path(save_path) / "training_config.yaml"
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
            config_dict['task_type'] = self.task_type.value
            
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Model saved successfully to: {save_path}")
            logger.info(f"Task type: {self.task_type.value}")
            
            # Store for push_to_hub
            self._last_save_path = save_path
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load a trained model with task type detection."""
        try:
            logger.info(f"Loading Unsloth model from: {path}")
            
            # Try to load task type from config
            config_path = Path(path) / "training_config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    saved_config = yaml.safe_load(f)
                    if 'task_type' in saved_config:
                        self.task_type = TaskType(saved_config['task_type'])
                        logger.info(f"Loaded task type: {self.task_type.value}")
            
            import unsloth
            from unsloth import FastLanguageModel
            
            # Load model and tokenizer
            self.unsloth_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=self.config.model.max_seq_length if hasattr(self.config, 'model') else 2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise