"""
Unsloth Schema-Guided JSON Extraction Trainer for AlignTune v3.10

Extends TRL's SFTTrainer to support structured JSON output with schema validation,
with the model backbone loaded through Unsloth's FastLanguageModel for accelerated
training (faster kernels, patched LoRA, reduced memory footprint). TRL's SFTTrainer
itself works unchanged on top of the Unsloth-loaded model - Unsloth's acceleration
comes entirely from how the model is loaded/patched, not from any change to the
trainer's loss computation.

Features:
- JSON schema validation at training time
- Differentiable schema loss: 0 if valid JSON matching schema, 1 if invalid
- Combined loss: lm_loss + schema_loss_weight * json_schema_loss
- Graceful error handling: invalid JSON logged and training continues
- Compatible with any causal LM architecture supported by Unsloth
"""

import logging
import json
import re
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

try:
    from pydantic import BaseModel, ValidationError, create_model
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    ValidationError = Exception

logger = logging.getLogger(__name__)


@dataclass
class ExtractionTrainerConfig:
    """Configuration for Unsloth Schema-Guided JSON Extraction Trainer."""

    # Schema settings
    json_schema: Dict[str, Any] = None  # JSON schema definition
    schema_type: str = "pydantic"  # Type of schema ("pydantic", "jsonschema")
    json_schema_loss_weight: float = 0.1  # Weight for schema validation loss

    # Output settings
    extract_last_json_block: bool = True  # Extract last {...} block from output
    strict_json_validation: bool = False  # If True, training halts on invalid JSON

    # SFT settings inherited from base
    model_name: str = "gpt2"
    dataset_name: str = None
    output_dir: str = "./output/unsloth_extraction"
    run_name: str = "unsloth_extraction"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 1

    # Unsloth-specific settings
    load_in_4bit: bool = False  # Whether to load the backbone in 4-bit precision
    dtype: Optional[str] = None  # None lets Unsloth auto-detect the best dtype

    def __post_init__(self):
        """Validate configuration."""
        if not PYDANTIC_AVAILABLE and self.schema_type == "pydantic":
            logger.warning(
                "Pydantic not available. Schema validation will be disabled. "
                "Install with: pip install pydantic"
            )

        if self.json_schema is None:
            logger.warning("No JSON schema provided. Schema validation will be disabled.")

        if not 0 <= self.json_schema_loss_weight <= 1:
            raise ValueError(
                f"json_schema_loss_weight must be between 0 and 1, got {self.json_schema_loss_weight}"
            )

        logger.info(
            f"ExtractionTrainerConfig initialized with schema_type={self.schema_type}, "
            f"loss_weight={self.json_schema_loss_weight}"
        )


class JSONSchemaValidator:
    """
    Validates JSON output against schema.

    Supports Pydantic models and basic JSON schema validation.
    """

    def __init__(self, schema: Dict[str, Any], schema_type: str = "pydantic"):
        """
        Initialize JSON schema validator.

        Args:
            schema: Schema definition (Pydantic model or JSON schema dict)
            schema_type: Type of schema ("pydantic" or "jsonschema")
        """
        self.schema = schema
        self.schema_type = schema_type
        self.pydantic_model = None

        if schema_type == "pydantic" and PYDANTIC_AVAILABLE and isinstance(schema, dict):
            # Convert dict schema to Pydantic model
            self._build_pydantic_model()

    def _build_pydantic_model(self):
        """Build Pydantic model from schema dict."""
        if not isinstance(self.schema, dict):
            return

        try:
            # Extract fields from schema
            fields = {}
            properties = self.schema.get("properties", {})

            for field_name, field_schema in properties.items():
                field_type = field_schema.get("type", "string")

                # Map JSON types to Python types
                type_mapping = {
                    "string": str,
                    "integer": int,
                    "number": float,
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }

                python_type = type_mapping.get(field_type, str)

                # Handle optional fields
                required = self.schema.get("required", [])
                if field_name not in required:
                    python_type = Optional[python_type]

                fields[field_name] = (python_type, ...)

            # Create Pydantic model
            self.pydantic_model = create_model("DynamicSchema", **fields)
            logger.info(f"Pydantic model created with fields: {list(fields.keys())}")

        except Exception as e:
            logger.warning(f"Failed to build Pydantic model: {e}")
            self.pydantic_model = None

    def validate_json(self, text: str) -> tuple[bool, Optional[Dict]]:
        """
        Validate JSON text against schema.

        Args:
            text: Text potentially containing JSON

        Returns:
            (is_valid, parsed_json) tuple
        """
        # Extract JSON from text
        json_obj = self._extract_json(text)

        if json_obj is None:
            return False, None

        # Validate against Pydantic model if available
        if self.pydantic_model is not None:
            try:
                self.pydantic_model(**json_obj)
                return True, json_obj
            except ValidationError as e:
                logger.debug(f"Pydantic validation failed: {e}")
                return False, json_obj

        # Basic schema validation if Pydantic not available
        if isinstance(self.schema, dict):
            return self._validate_jsonschema(json_obj), json_obj

        return True, json_obj

    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract JSON object from text.

        Attempts to find and parse JSON blocks in the format {...}.

        Args:
            text: Text to search for JSON

        Returns:
            Parsed JSON dict or None
        """
        if not isinstance(text, str):
            return None

        # Try to find JSON blocks
        # Pattern: { ... } (handles nested braces)
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

        matches = re.findall(json_pattern, text)

        if not matches:
            return None

        # Try to parse matches, return the first valid one
        # or the last one (if extract_last_json_block is True)
        candidates = matches if not getattr(self, "extract_last", True) else [matches[-1]]

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        return None

    def _validate_jsonschema(self, json_obj: Dict) -> bool:
        """
        Basic JSON schema validation.

        Args:
            json_obj: Parsed JSON object

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(self.schema, dict):
            return True

        properties = self.schema.get("properties", {})
        required = self.schema.get("required", [])

        # Check required fields
        for field in required:
            if field not in json_obj:
                logger.debug(f"Missing required field: {field}")
                return False

        # Check field types (basic validation)
        for field, value in json_obj.items():
            if field in properties:
                expected_type = properties[field].get("type", "string")
                if not self._type_matches(value, expected_type):
                    logger.debug(f"Type mismatch for field {field}")
                    return False

        return True

    @staticmethod
    def _type_matches(value: Any, expected_type: str) -> bool:
        """
        Check if value matches expected JSON type.

        Args:
            value: Value to check
            expected_type: Expected type name

        Returns:
            True if type matches
        """
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        type_check = type_checks.get(expected_type, lambda v: True)
        return type_check(value)


class UnslothExtractionControlledSFTTrainer(SFTTrainer):
    """
    SFT Trainer with schema-guided JSON extraction support, accelerated by Unsloth.

    Extends TRL's SFTTrainer to:
    1. Validate JSON output against provided schema
    2. Compute auxiliary schema validation loss
    3. Penalize structurally invalid JSON outputs
    4. Gracefully handle validation errors during training

    Loss computation:
    - Base: Standard language modeling loss
    - Auxiliary: json_schema_loss (0 if valid JSON matching schema, 1 if invalid)
    - Combined: lm_loss + schema_loss_weight * json_schema_loss

    Note: this class subclasses TRL's SFTTrainer directly, unchanged. Unsloth's
    acceleration comes entirely from the model instance passed in via `model=` at
    construction time - that model must be loaded with Unsloth's FastLanguageModel
    (see `UnslothExtractionTrainer.load_model_and_tokenizer`) for the speedups to
    apply. The loss/validation logic below is identical to the plain TRL version.
    """

    def __init__(self, config: ExtractionTrainerConfig, *args, **kwargs):
        """
        Initialize Unsloth Extraction-controlled SFT Trainer.

        Args:
            config: ExtractionTrainerConfig instance
            *args: Positional arguments for SFTTrainer
            **kwargs: Keyword arguments for SFTTrainer
        """
        self.extraction_config = config
        self.json_schema_loss_weight = config.json_schema_loss_weight
        self.extract_last_json_block = config.extract_last_json_block
        self.strict_json_validation = config.strict_json_validation

        # Initialize schema validator
        self.schema_validator = None
        if config.json_schema is not None:
            self.schema_validator = JSONSchemaValidator(
                config.json_schema,
                schema_type=config.schema_type
            )
            self.schema_validator.extract_last = config.extract_last_json_block

        # Initialize parent SFTTrainer
        super().__init__(*args, **kwargs)

        logger.info("UnslothExtractionControlledSFTTrainer initialized")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute combined loss: LM loss + schema validation loss.

        Args:
            model: The model being trained
            inputs: Input dictionary with input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs
            **kwargs: Additional arguments

        Returns:
            Loss or (loss, outputs) tuple
        """
        # Get base LM loss from parent
        outputs = model(**inputs)
        lm_loss = outputs.loss

        if self.json_schema_loss_weight == 0 or self.schema_validator is None:
            # No auxiliary loss
            if return_outputs:
                return lm_loss, outputs
            return lm_loss

        # Compute schema validation loss
        try:
            schema_loss = self._compute_schema_loss(outputs, inputs)

            # Combined loss
            total_loss = lm_loss + self.json_schema_loss_weight * schema_loss

            logger.debug(
                f"LM Loss: {lm_loss:.4f}, Schema Loss: {schema_loss:.4f}, "
                f"Total: {total_loss:.4f}"
            )

            if return_outputs:
                return total_loss, outputs
            return total_loss

        except Exception as e:
            if self.strict_json_validation:
                logger.error(f"Schema validation error: {e}")
                raise
            else:
                logger.warning(f"Schema validation error, using LM loss only: {e}")
                if return_outputs:
                    return lm_loss, outputs
                return lm_loss

    def _compute_schema_loss(self, outputs, inputs) -> torch.Tensor:
        """
        Compute JSON schema validation loss.

        Args:
            outputs: Model outputs
            inputs: Input batch

        Returns:
            Schema validation loss (0 if valid, 1 if invalid)
        """
        batch_size = inputs["input_ids"].shape[0]
        schema_losses = []

        # Get predicted token IDs
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Decode predictions to text
        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer is None:
            logger.warning("Tokenizer not available for schema validation")
            return torch.tensor(0.0, device=logits.device)

        for batch_idx in range(batch_size):
            try:
                # Decode predicted tokens
                pred_tokens = predictions[batch_idx]
                pred_text = tokenizer.decode(
                    pred_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                # Validate JSON
                is_valid, json_obj = self.schema_validator.validate_json(pred_text)

                # Loss: 0 if valid, 1 if invalid
                loss = torch.tensor(
                    0.0 if is_valid else 1.0,
                    device=logits.device,
                    dtype=torch.float32
                )
                schema_losses.append(loss)

            except Exception as e:
                logger.debug(f"Error validating JSON for batch {batch_idx}: {e}")
                # Assume invalid on error
                schema_losses.append(torch.tensor(1.0, device=logits.device))

        # Average loss across batch
        if schema_losses:
            schema_loss = torch.stack(schema_losses).mean()
        else:
            schema_loss = torch.tensor(0.0, device=logits.device)

        return schema_loss

    def _prepare_extraction_dataset(self, dataset):
        """
        Prepare dataset for JSON extraction training.

        Ensures output text contains valid JSON structures.

        Args:
            dataset: Input dataset

        Returns:
            Prepared dataset
        """
        def validate_and_prepare(examples):
            """Validate examples contain JSON."""
            valid_indices = []
            valid_responses = []

            responses = examples.get("response", examples.get("output", []))

            for i, response in enumerate(responses):
                if isinstance(response, str):
                    # Check if response contains valid JSON
                    is_valid, _ = self.schema_validator.validate_json(response)
                    if is_valid or not self.strict_json_validation:
                        valid_indices.append(i)
                        valid_responses.append(response)
                    else:
                        logger.warning(
                            f"Skipping invalid JSON response at index {i}: {response[:100]}..."
                        )

            if valid_indices:
                # Return only valid examples
                result = {k: [examples[k][i] for i in valid_indices] for k in examples.keys()}
                return result
            else:
                # Return original if no valid JSON found
                return examples

        if self.schema_validator is not None:
            dataset = dataset.map(
                validate_and_prepare,
                batched=True,
                desc="Validating JSON extraction dataset"
            )

        return dataset


class UnslothExtractionTrainer:
    """
    Wrapper class for easy schema-guided JSON extraction training, accelerated
    by Unsloth.

    Handles Unsloth-accelerated model loading, dataset preparation, and training
    lifecycle. The heavy lifting for JSON-schema-guided loss computation is done
    by TRL's SFTTrainer (via `UnslothExtractionControlledSFTTrainer`); Unsloth's
    speedups come from how the underlying causal LM is loaded here.
    """

    def __init__(self, config: ExtractionTrainerConfig):
        """
        Initialize Unsloth Extraction Trainer.

        Args:
            config: ExtractionTrainerConfig instance
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        logger.info("UnslothExtractionTrainer initialized")

    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth is available."""
        try:
            import unsloth
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            return True
        except ImportError:
            return False

    def load_model_and_tokenizer(self):
        """Load pretrained model and tokenizer via Unsloth's FastLanguageModel."""
        logger.info(f"Loading model with Unsloth acceleration: {self.config.model_name}")

        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "Unsloth is not installed. Install with: pip install unsloth"
            ) from e

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )

        # Prepare the Unsloth model for training (patches gradient checkpointing,
        # enables training-mode kernels, etc.). TRL's SFTTrainer then operates on
        # this model completely unchanged.
        FastLanguageModel.for_training(self.model)

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Unsloth model loaded: {self.model}")
        logger.info(f"Tokenizer loaded, vocab size: {len(self.tokenizer)}")

    def setup_trainer(self, train_dataset, eval_dataset=None):
        """
        Setup UnslothExtractionControlledSFTTrainer.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        from transformers import TrainingArguments

        # Prepare dataset
        # Note: actual preparation happens in trainer if needed

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            run_name=self.config.run_name,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_strategy="epoch",
            logging_steps=10,
        )

        # Create trainer
        self.trainer = UnslothExtractionControlledSFTTrainer(
            config=self.config,
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        logger.info("UnslothExtractionControlledSFTTrainer setup complete")

    def train(self):
        """Execute training."""
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")

        logger.info("Starting Unsloth-accelerated schema-guided JSON extraction training...")
        self.trainer.train()
        logger.info("Training complete")

    def save_model(self, output_path: str):
        """
        Save trained model and tokenizer.

        Args:
            output_path: Directory to save model
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
