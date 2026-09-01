import logging
from typing import Dict, Optional, List, Any, Callable
from datasets import Dataset, DatasetDict
from aligntune.data.schemas import TaskType, TaskSchema, TASK_SCHEMAS

logger = logging.getLogger(__name__)

class ColumnMapper:
    """Normalizes raw column names to the Task Schema."""

    def __init__(self, task_type: TaskType, user_mapping: Optional[Dict[str, str]] = None):
        self.task_type = TaskType(task_type)
        self.schema = TASK_SCHEMAS.get(self.task_type)
        self.user_mapping = user_mapping or {}

    @staticmethod
    def _normalize_sft_messages(dataset: Dataset) -> Dataset:
        """Normalize preserved ShareGPT turns into TRL-compatible messages."""
        import json

        # CuratorKIT emits the canonical SFT answer as ``output``. This is
        # especially important when a DPO row's ``chosen`` answer was mapped
        # by CuratorKIT before reaching AlignTune. Populate the canonical
        # ``completion`` field before constructing assistant messages.
        if "output" in dataset.column_names:
            if "completion" not in dataset.column_names:
                dataset = dataset.rename_column("output", "completion")
            else:
                def fill_completion(example: Dict[str, Any]) -> Dict[str, Any]:
                    if not str(example.get("completion") or "").strip():
                        example["completion"] = example.get("output") or ""
                    return example

                dataset = dataset.map(
                    fill_completion,
                    desc="Filling SFT completion from canonical output",
                )

        conversation_column = next(
            (
                column
                for column in ("messages", "conversations", "conversation")
                if column in dataset.column_names
            ),
            None,
        )

        def coerce_messages(value: Any) -> List[Dict[str, str]]:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return []
            if isinstance(value, dict):
                value = value.get("messages", value.get("conversations", []))
            if not isinstance(value, list):
                return []

            role_map = {
                "human": "user",
                "user": "user",
                "gpt": "assistant",
                "assistant": "assistant",
                "bot": "assistant",
                "system": "system",
            }
            messages = []
            for turn in value:
                if not isinstance(turn, dict):
                    continue
                raw_role = turn.get("role", turn.get("from", turn.get("speaker")))
                role = role_map.get(str(raw_role).lower()) if raw_role is not None else None
                content = turn.get("content", turn.get("value", turn.get("text")))
                if role and content is not None:
                    messages.append({"role": role, "content": str(content)})
            return messages

        if conversation_column:
            def add_source_messages(example: Dict[str, Any]) -> Dict[str, Any]:
                messages = coerce_messages(example.get(conversation_column))
                if messages:
                    example["messages"] = messages
                return example

            dataset = dataset.map(
                add_source_messages,
                desc="Normalizing ShareGPT conversations",
            )
            if conversation_column != "messages" and "messages" in dataset.column_names:
                dataset = dataset.remove_columns(conversation_column)
            if "messages" in dataset.column_names:
                return dataset

        if not {"prompt", "completion"}.issubset(dataset.column_names):
            return dataset

        def build_messages(example: Dict[str, Any]) -> Dict[str, Any]:
            metadata = example.get("metadata") or {}
            turns = metadata.get("turns", []) if isinstance(metadata, dict) else []
            source_system = metadata.get("system_prompt") if isinstance(metadata, dict) else None

            messages: List[Dict[str, str]] = []
            if source_system:
                messages.append({"role": "system", "content": str(source_system)})
            messages.extend([
                {"role": "user", "content": str(example.get("prompt", ""))},
                {"role": "assistant", "content": str(example.get("completion", ""))},
            ])
            if isinstance(turns, list):
                messages.extend(
                    turn for turn in turns
                    if isinstance(turn, dict)
                    and turn.get("role") in {"system", "user", "assistant"}
                    and "content" in turn
                )
            example["messages"] = messages
            return example

        return dataset.map(build_messages, desc="Normalizing conversational SFT data")

    @staticmethod
    def _apply_schema_heuristics(dataset: Dataset, schema: "TaskSchema") -> Dataset:
        """Rename raw synonym columns to a schema's required column names."""
        current_cols = dataset.column_names
        rename_map: Dict[str, str] = {}
        for target in schema.required_columns:
            if target in current_cols or target in rename_map.values():
                continue
            for synonym in schema.column_heuristics.get(target, []):
                if synonym in current_cols and synonym not in rename_map:
                    rename_map[synonym] = target
                    break
        if rename_map:
            logger.info(f"Mapping columns: {rename_map}")
            dataset = dataset.rename_columns(rename_map)
        return dataset

    @staticmethod
    def _combine_sft_input(dataset: Dataset) -> Dataset:
        """Combine an optional instruction input/context into ``prompt``."""
        if "prompt" not in dataset.column_names or "input" not in dataset.column_names:
            return dataset

        def combine(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt = str(example.get("prompt") or "").strip()
            context = str(example.get("input") or "").strip()
            if context and context not in prompt:
                example["prompt"] = f"{prompt}\n\n{context}" if prompt else context
            return example

        return dataset.map(combine, desc="Combining SFT instruction context")

    @staticmethod
    def _column_is_missing_or_blank(dataset: Dataset, column: str) -> bool:
        """True if `column` is absent, or present but empty for the first row.

        CuratorKIT's generic curator step pre-creates canonical columns (e.g.
        `prompt`) as empty placeholders before task-specific normalization
        runs, so a plain "not in column_names" check misses that case.
        """
        if column not in dataset.column_names:
            return True
        if len(dataset) == 0:
            return False
        value = dataset[0].get(column)
        return value is None or value == ""

    @staticmethod
    def _normalize_prompt_task(dataset: Dataset) -> Dataset:
        """Normalize prompt/reward-style tasks while preserving extra columns."""
        if ColumnMapper._column_is_missing_or_blank(dataset, "prompt") and "input" in dataset.column_names:
            if dataset[0].get("input"):
                dataset = dataset.remove_columns("prompt") if "prompt" in dataset.column_names else dataset
                dataset = dataset.rename_column("input", "prompt")

        # Same Anthropic/hh-rlhf style fallback as _normalize_preference: some
        # datasets only ship full conversation strings under `chosen`/`rejected`
        # (used e.g. by Online DPO, which only needs the prompt half).
        if (ColumnMapper._column_is_missing_or_blank(dataset, "prompt")
                and "chosen" in dataset.column_names
                and len(dataset) > 0
                and isinstance(dataset[0].get("chosen"), str)
                and ColumnMapper._split_anthropic_style_conversation(dataset[0]["chosen"]) is not None):

            def extract_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
                split = ColumnMapper._split_anthropic_style_conversation(example["chosen"])
                if split is not None:
                    example["prompt"] = split[0]
                return example

            dataset = dataset.map(extract_prompt, desc="Extracting Anthropic-style prompt from chosen")

        return dataset

    @staticmethod
    def _split_anthropic_style_conversation(text: str) -> Optional[tuple]:
        """Split a hh-rlhf style ``...\\n\\nHuman: ...\\n\\nAssistant: ...`` string
        into ``(prompt, completion)`` at the LAST ``\\n\\nAssistant:`` marker."""
        marker = "\n\nAssistant:"
        idx = text.rfind(marker)
        if idx == -1:
            return None
        return text[: idx + len(marker)], text[idx + len(marker):]

    @staticmethod
    def _normalize_preference(dataset: Dataset) -> Dataset:
        """Normalize explicit DPO rows to consistently plain or conversational fields."""
        import ast

        prompt_missing = ColumnMapper._column_is_missing_or_blank(dataset, "prompt")
        if prompt_missing and "instruction" in dataset.column_names and len(dataset) and dataset[0].get("instruction"):
            if "prompt" in dataset.column_names:
                dataset = dataset.remove_columns("prompt")
            dataset = dataset.rename_column("instruction", "prompt")
            prompt_missing = False
        if prompt_missing and "input" in dataset.column_names and len(dataset) and dataset[0].get("input"):
            if "prompt" in dataset.column_names:
                dataset = dataset.remove_columns("prompt")
            dataset = dataset.rename_column("input", "prompt")

        # Some preference datasets (e.g. Anthropic/hh-rlhf) only ship full
        # conversation strings under `chosen`/`rejected`, with no separate
        # `prompt` column. Derive it by splitting at the last "\n\nAssistant:"
        # turn marker, the standard hh-rlhf preprocessing recipe.
        if (ColumnMapper._column_is_missing_or_blank(dataset, "prompt")
                and all(c in dataset.column_names for c in ("chosen", "rejected"))
                and len(dataset) > 0
                and isinstance(dataset[0].get("chosen"), str)
                and ColumnMapper._split_anthropic_style_conversation(dataset[0]["chosen"]) is not None):

            def split_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
                chosen_split = ColumnMapper._split_anthropic_style_conversation(example["chosen"])
                rejected_split = ColumnMapper._split_anthropic_style_conversation(example["rejected"])
                if chosen_split is None or rejected_split is None:
                    return example
                prompt, chosen_completion = chosen_split
                _, rejected_completion = rejected_split
                example["prompt"] = prompt
                example["chosen"] = chosen_completion
                example["rejected"] = rejected_completion
                return example

            dataset = dataset.map(split_conversation, desc="Extracting Anthropic-style prompt from chosen/rejected")

        missing = [column for column in ("prompt", "chosen", "rejected")
                   if column not in dataset.column_names]
        if missing:
            logger.warning("Preference dataset missing canonical columns: %s", missing)

        def parse_message_list(value: Any) -> Any:
            """Restore CuratorKIT's Python-repr conversation strings safely."""
            if not isinstance(value, str) or not value.lstrip().startswith("["):
                return value
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return value
            if not isinstance(parsed, list):
                return value
            if not all(
                isinstance(message, dict)
                and isinstance(message.get("role"), str)
                and isinstance(message.get("content"), str)
                for message in parsed
            ):
                return value
            return parsed

        def normalize_preference_row(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt = parse_message_list(example.get("prompt"))
            chosen = parse_message_list(example.get("chosen"))
            rejected = parse_message_list(example.get("rejected"))

            if not isinstance(prompt, (str, list)) or not prompt:
                raise ValueError(
                    "Implicit string DPO is not supported because it has no explicit "
                    "prompt. Provide prompt, chosen, and rejected explicitly."
                )

            values = (prompt, chosen, rejected)
            if all(isinstance(value, str) for value in values):
                example["prompt"] = prompt
                example["chosen"] = chosen
                example["rejected"] = rejected
                return example

            if all(isinstance(value, list) and value for value in values):
                example["prompt"] = prompt
                example["chosen"] = chosen
                example["rejected"] = rejected
                return example

            raise ValueError(
                "DPO rows must use one consistent schema: prompt/chosen/rejected "
                "must all be strings or all be non-empty message lists."
            )

        if all(column in dataset.column_names for column in ("prompt", "chosen", "rejected")):
            dataset = dataset.map(normalize_preference_row, desc="Normalizing DPO data")
        return dataset

    @staticmethod
    def _normalize_kto(dataset: Dataset) -> Dataset:
        """Normalize KTO rows to ``prompt/completion/label``."""
        rename_map = {}
        if "prompt" not in dataset.column_names:
            for source in ("instruction", "input", "question"):
                if source in dataset.column_names:
                    rename_map[source] = "prompt"
                    break
        if "completion" not in dataset.column_names:
            for source in ("output", "response", "answer"):
                if source in dataset.column_names:
                    rename_map[source] = "completion"
                    break
        if rename_map:
            dataset = dataset.rename_columns(rename_map)
        missing = [column for column in ("prompt", "completion", "label")
                   if column not in dataset.column_names]
        if missing:
            logger.warning("KTO dataset missing canonical columns: %s", missing)
        return dataset

    @staticmethod
    def _normalize_distillation(dataset: Dataset, task_type: TaskType) -> Dataset:
        """Shape online/offline and self-distillation rows explicitly."""
        if task_type in {TaskType.DISTILLATION_ONLINE, TaskType.DISTILLATION_OFFLINE}:
            import json

            if "prompt" not in dataset.column_names and "instruction" in dataset.column_names:
                dataset = dataset.rename_column("instruction", "prompt")
            if "completion" not in dataset.column_names and "output" in dataset.column_names:
                dataset = dataset.rename_column("output", "completion")

            # The generic schema heuristics may have renamed a plain prompt to
            # ``messages``. Only keep it as-is when it is actually a list of
            # role/content dictionaries.
            if "messages" in dataset.column_names:
                sample_messages = dataset[0]["messages"] if len(dataset) else None
                if isinstance(sample_messages, list):
                    return dataset
                if "prompt" in dataset.column_names:
                    dataset = dataset.remove_columns("messages")
                else:
                    dataset = dataset.rename_column("messages", "prompt")

            if "prompt" not in dataset.column_names:
                return dataset

            def make_messages(example: Dict[str, Any]) -> Dict[str, Any]:
                prompt = example.get("prompt", [])
                if isinstance(prompt, str):
                    try:
                        parsed = json.loads(prompt)
                    except (TypeError, json.JSONDecodeError):
                        parsed = None
                    if isinstance(parsed, list):
                        prompt_messages = parsed
                    else:
                        prompt_messages = [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list):
                    prompt_messages = [
                        message for message in prompt
                        if isinstance(message, dict) and message.get("role") in {"system", "user", "assistant"}
                    ]
                else:
                    prompt_messages = []

                if task_type == TaskType.DISTILLATION_OFFLINE:
                    completion = example.get("completion", example.get("response", ""))
                    if completion and not any(
                        message.get("role") == "assistant" for message in prompt_messages
                    ):
                        prompt_messages.append({"role": "assistant", "content": str(completion)})
                example["messages"] = prompt_messages
                return example

            return dataset.map(make_messages, desc="Normalizing distillation messages")

        if task_type in {TaskType.DISTILLATION_SDFT, TaskType.DISTILLATION_SDPO}:
            if "prompt" not in dataset.column_names and "instruction" in dataset.column_names:
                dataset = dataset.rename_column("instruction", "prompt")
            if "privileged_context" not in dataset.column_names:
                logger.warning("%s dataset has no privileged_context column", task_type.value)
            return dataset

        return dataset

    def _normalize_by_task(self, dataset: Dataset) -> Dataset:
        """Apply the explicit post-CuratorKIT normalizer for this task."""
        if self.task_type == TaskType.SFT:
            dataset = self._combine_sft_input(dataset)
            return self._normalize_sft_messages(dataset)
        if self.task_type == TaskType.DISTILLATION_OFFLINE:
            # Offline distillation consumes the same full user/assistant
            # conversations as SFT, including Alpaca's optional input context.
            # DISTILLATION_OFFLINE's own schema maps raw synonyms straight to
            # "messages" (for datasets that already ship prompt/completion
            # pairs), so the generic rename step in process() never renames
            # Alpaca's "instruction"/"output" to "prompt"/"completion" -
            # apply SFT's own heuristics here so _combine_sft_input/
            # _normalize_sft_messages (which require "prompt"/"completion")
            # have something to work with.
            dataset = self._apply_schema_heuristics(dataset, TASK_SCHEMAS[TaskType.SFT])
            dataset = self._combine_sft_input(dataset)
            return self._normalize_sft_messages(dataset)
        if self.task_type == TaskType.DPO:
            return self._normalize_preference(dataset)
        if self.task_type == TaskType.KTO:
            return self._normalize_kto(dataset)
        if self.task_type in {TaskType.GRPO, TaskType.PPO, TaskType.ES}:
            return self._normalize_prompt_task(dataset)
        if self.task_type in {
            TaskType.DISTILLATION_ONLINE,
            TaskType.DISTILLATION_OFFLINE,
            TaskType.DISTILLATION_SDFT,
            TaskType.DISTILLATION_SDPO,
        }:
            return self._normalize_distillation(dataset, self.task_type)
        return dataset

    def _process_distillation_online_messages(self, dataset: Dataset) -> Dataset:
        """
        For online distillation, extract user prompts from messages.
        If messages contain both user and assistant, extract just the user prompts.
        Creates a cleaned messages list with only user turns.
        Handles both list and JSON string formats for messages.
        """
        import json
        if "messages" not in dataset.column_names:
            return dataset

        def _extract_user_prompts(example):
            messages = example.get("messages", [])

            # Handle JSON string format
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except (json.JSONDecodeError, TypeError):
                    logger.debug(f"Failed to parse messages as JSON string")
                    return example

            if not isinstance(messages, list):
                return example

            # Extract only user messages (excluding assistant responses)
            user_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"]

            if not user_messages:
                # If no user messages, skip this example
                return None

            # Create messages list with only user turns (ensure it's a list)
            example["messages"] = user_messages
            return example

        result = dataset.map(_extract_user_prompts, desc="Extracting prompts for online distillation")
        # Filter out None results
        result = result.filter(lambda x: x is not None, desc="Removing examples without user prompts")

        original_size = len(dataset)
        filtered_size = len(result)
        if original_size - filtered_size > 0:
            logger.info(f"Online Distillation: Extracted prompts from {original_size} examples, kept {filtered_size}")

        return result

    def _process_distillation_offline_messages(self, dataset: Dataset) -> Dataset:
        """
        For offline distillation, validate that messages contain both user and assistant.
        Handles both list and JSON string formats for messages.
        """
        import json
        if "messages" not in dataset.column_names:
            return dataset

        def _has_user_and_assistant(example):
            messages = example.get("messages", [])

            # Handle JSON string format
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except (json.JSONDecodeError, TypeError):
                    logger.debug(f"Failed to parse messages as JSON string for offline distillation")
                    return False

            if not isinstance(messages, list):
                return False

            has_user = any(isinstance(msg, dict) and msg.get("role") == "user" for msg in messages)
            has_assistant = any(isinstance(msg, dict) and msg.get("role") == "assistant" for msg in messages)

            if not (has_user and has_assistant):
                logger.debug(
                    f"Offline distillation requires both user and assistant messages. "
                    f"Found: user={has_user}, assistant={has_assistant}"
                )
                return False

            return True

        original_size = len(dataset)
        result = dataset.filter(_has_user_and_assistant, desc="Filtering examples for offline distillation")
        filtered_size = len(result)

        if original_size - filtered_size > 0:
            logger.info(f"Offline Distillation: Filtered {original_size - filtered_size} examples without proper message format")

        return result

    def _validate_distillation_sdft_sdpo(self, dataset: Dataset) -> Dataset:
        """
        For SDFT/SDPO, validate that both prompt and privileged_context exist.
        """
        required_cols = ["prompt", "privileged_context"]

        def _has_required_fields(example):
            for col in required_cols:
                val = example.get(col)
                if val is None:
                    logger.debug(f"Missing required column: {col}")
                    return False
                if isinstance(val, str) and len(val.strip()) == 0:
                    logger.debug(f"Empty {col}")
                    return False
                if isinstance(val, list) and len(val) == 0:
                    logger.debug(f"Empty {col}")
                    return False
            return True

        original_size = len(dataset)
        result = dataset.filter(_has_required_fields, desc="Filtering examples for SDFT/SDPO distillation")
        filtered_size = len(result)

        if original_size - filtered_size > 0:
            logger.info(f"SDFT/SDPO Distillation: Filtered {original_size - filtered_size} examples missing prompt or privileged_context")

        return result

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
                logger.warning(f"CODE TASK SAFETY: Ignoring user mapping '{src}' -> '{tgt}'. "
                               f"This prevents masking 'test_cases' with reference code strings.")
                continue

            # --- SAFETY CHECK 2: Existing Columns ---
            # If the target column already exists (e.g., created by a preprocessor like preprocess_mbpp),
            # do not attempt to rename, as it might overwrite the preprocessed data with raw data.
            if tgt in current_cols:
                logger.debug(f"Target column '{tgt}' already exists. Skipping mapping from '{src}'.")
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

        # CuratorKIT has already detected the raw format. Normalize its
        # canonical fields into the selected trainer schema.
        dataset = self._normalize_by_task(dataset)

        # 3. Apply Distillation-specific validation
        if self.task_type == TaskType.DISTILLATION_ONLINE:
            dataset = self._process_distillation_online_messages(dataset)
        elif self.task_type == TaskType.DISTILLATION_OFFLINE:
            dataset = self._process_distillation_offline_messages(dataset)
        elif self.task_type in [TaskType.DISTILLATION_SDFT, TaskType.DISTILLATION_SDPO]:
            dataset = self._validate_distillation_sdft_sdpo(dataset)

        # 4. Validation
        missing = [c for c in self.schema.required_columns if c not in dataset.column_names]

        if self.schema == TASK_SCHEMAS[TaskType.SFT] and "messages" in dataset.column_names:
            missing = []

        if missing:
            logger.warning(f"Dataset missing required columns {missing} for task. "
                           f"Found: {dataset.column_names}. ")

        return dataset


class SystemPromptInjector:
    """Injects system prompts into text prompts or chat structures with tokenizer support."""
    
    def __init__(
        self,
        system_prompt: Optional[str],
        tokenizer=None,
        enable_thinking: bool = False,
        task_type: Optional[TaskType] = None,
    ):
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.task_type = TaskType(task_type) if task_type is not None else None
        
        # Debug logging
        if system_prompt:
            logger.info(f"SystemPromptInjector initialized with system prompt: {system_prompt[:50]}...")
            if tokenizer is not None:
                logger.info(f"Tokenizer provided: {type(tokenizer).__name__}")
                logger.info(f"   Has apply_chat_template: {hasattr(tokenizer, 'apply_chat_template')}")
                if hasattr(tokenizer, 'chat_template'):
                    has_template = tokenizer.chat_template is not None
                    logger.info(f"   Chat template exists: {has_template}")
            else:
                logger.warning("No tokenizer provided to SystemPromptInjector")

    def process(self, dataset: Dataset) -> Dataset:
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
            # Start with copy of all columns (important for KTO, etc.)
            result = {k: v for k, v in example.items()}

            # DPO supports either all-string fields or all message-list fields.
            # Keep string DPO plain and give conversational DPO template kwargs.
            if self.task_type == TaskType.DPO:
                prompt = example.get("prompt")
                if isinstance(prompt, str):
                    if self.system_prompt and not prompt.startswith(self.system_prompt):
                        result["prompt"] = f"{self.system_prompt}\n\n{prompt}"
                    return result

                if isinstance(prompt, list):
                    prompt_messages = list(prompt)
                    if (
                        self.system_prompt
                        and prompt_messages
                        and prompt_messages[0].get("role") != "system"
                    ):
                        prompt_messages.insert(
                            0, {"role": "system", "content": self.system_prompt}
                        )
                    result["prompt"] = prompt_messages
                    result["chat_template_kwargs"] = {
                        "enable_thinking": bool(self.enable_thinking)
                    }
                    return result

            # Case 1: Chat messages format (prompt is messages list)
            if "messages" in example and isinstance(example["messages"], list):
                logger.debug("Processing messages format")
                msgs = example["messages"]
                new_msgs = list(msgs)

                # Add system prompt if not present
                if (
                    self.system_prompt
                    and new_msgs
                    and new_msgs[0].get("role") != "system"
                ):
                    new_msgs.insert(0, {"role": "system", "content": self.system_prompt})

                # Keep the dataset model-agnostic. TRL applies the tokenizer
                # template during tokenization using these per-row kwargs.
                result["messages"] = new_msgs
                result["chat_template_kwargs"] = {
                    "enable_thinking": bool(self.enable_thinking)
                }
                return result

            # Case 2: Conversational preference prompt. Keep the list shape so
            # TRL can apply its chosen/rejected conversation handling later.
            if "prompt" in example and isinstance(example["prompt"], list):
                logger.debug("Processing conversational prompt format")
                prompt_messages = [
                    message for message in example["prompt"]
                    if isinstance(message, dict)
                    and message.get("role") in {"system", "user", "assistant"}
                ]
                if prompt_messages and prompt_messages[0].get("role") != "system":
                    if self.system_prompt:
                        prompt_messages.insert(
                            0, {"role": "system", "content": self.system_prompt}
                        )
                result["prompt"] = prompt_messages
                result["chat_template_kwargs"] = {
                    "enable_thinking": bool(self.enable_thinking)
                }
                return result

            # Case 3: Prompt format (prompt is string)
            if (
                self.system_prompt
                and "prompt" in example
                and isinstance(example["prompt"], str)
            ):
                logger.debug("Processing prompt format")
                prompt_text = example["prompt"]

                # Avoid double system prompt injection if prompt is already chat-templated.
                # This can happen if the dataset already contains templated prompts.
                if _looks_like_chat_templated(prompt_text):
                    patched = _inject_into_existing_template(prompt_text)
                    if patched is not None:
                        result["prompt"] = patched
                        return result
                    return result

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt_text}
                ]
                completion = example.get("completion")
                if isinstance(completion, str) and completion:
                    messages.append({"role": "assistant", "content": completion})
                result["messages"] = messages
                result["chat_template_kwargs"] = {
                    "enable_thinking": bool(self.enable_thinking)
                }

                return result

            # Case 4: Context format (fallback)
            if "context" in example and isinstance(example["context"], str):
                logger.debug("Processing context format")
                result["context"] = f"{self.system_prompt}\n\n{example['context']}"
                return result

            logger.debug("No suitable column found for injection")
            return result

        cols = dataset.column_names
        logger.info(f"Dataset columns before injection: {cols}")
        
        if any(c in cols for c in ["messages", "prompt", "context"]):
            logger.info("Injecting system prompt and chat-template kwargs into dataset")
            result = dataset.map(_inject, desc="Injecting system prompt into messages")
            logger.info(f"Dataset columns after injection: {result.column_names}")
            return result
        
        logger.warning(f"⚠️ No suitable columns found for system prompt injection. Columns: {cols}")
        return dataset
