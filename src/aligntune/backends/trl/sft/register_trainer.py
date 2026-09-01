"""
Register/Formality Control Trainer for AlignTune v3.10

Extends SFTTrainer to support register control tokens for fine-grained tone/formality control.
Enables single model to switch between formal, legal, technical, and simple language registers.

Registers special tokens: [REGISTER: formal], [REGISTER: legal], [REGISTER: technical], [REGISTER: simple]
Prepends control token to instruction during SFT training.
Optional auxiliary loss penalizes register mismatch during training.
"""

import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class RegisterTrainerConfig:
    """Configuration for Register/Formality Control Trainer."""

    # Register control settings
    registers: List[str] = None  # List of register names: ["formal", "legal", "technical", "simple"]
    register_column: str = "register"  # Dataset column containing register label
    prepend_register_token: bool = True  # Whether to prepend [REGISTER: *] token to instruction
    register_loss_weight: float = 0.0  # Weight for auxiliary register classification loss (0 = disabled)

    # SFT settings inherited from base
    model_name: str = "gpt2"
    dataset_name: str = None
    output_dir: str = "./output"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.registers is None:
            self.registers = ["formal", "legal", "technical", "simple"]

        # Validate register names
        valid_registers = {"formal", "legal", "technical", "simple"}
        for reg in self.registers:
            if reg not in valid_registers:
                raise ValueError(f"Invalid register: {reg}. Must be one of {valid_registers}")

        # Validate register loss weight
        if not 0 <= self.register_loss_weight <= 1:
            raise ValueError(f"register_loss_weight must be between 0 and 1, got {self.register_loss_weight}")

        logger.info(f"RegisterTrainerConfig initialized with registers: {self.registers}")


class RegisterControlledSFTTrainer(SFTTrainer):
    """
    SFT Trainer with register/formality control tokens.

    Extends TRL's SFTTrainer to:
    1. Add special [REGISTER: *] tokens to tokenizer
    2. Prepend control tokens to instructions during training
    3. Optionally compute auxiliary register classification loss
    4. Learn to condition output style on register token

    The model learns to produce different formality levels based on the register token:
    - [REGISTER: formal] → formal business language
    - [REGISTER: legal] → legal document language
    - [REGISTER: technical] → technical/jargon-heavy language
    - [REGISTER: simple] → simple/accessible language
    """

    def __init__(self, config: RegisterTrainerConfig, *args, **kwargs):
        """
        Initialize Register-controlled SFT Trainer.

        Args:
            config: RegisterTrainerConfig instance
            *args: Positional arguments for SFTTrainer
            **kwargs: Keyword arguments for SFTTrainer
        """
        self.register_config = config
        self.register_tokens = [f"[REGISTER: {reg}]" for reg in config.registers]
        self.register_token_ids = {}

        # Extract SFT-specific kwargs
        self.register_loss_weight = config.register_loss_weight
        self.register_column = config.register_column
        self.prepend_register_token = config.prepend_register_token

        # Initialize parent SFTTrainer
        super().__init__(*args, **kwargs)

        # Register special tokens with tokenizer
        self._register_special_tokens()

    def _register_special_tokens(self):
        """Register [REGISTER: *] special tokens with tokenizer."""
        logger.info("Registering register control tokens with tokenizer...")

        # Get the number of tokens before adding
        initial_vocab_size = len(self.tokenizer)

        # Add special tokens to tokenizer
        new_tokens = self.register_tokens
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )

        logger.info(
            f"Added {num_added} register tokens. "
            f"Vocab size: {initial_vocab_size} -> {len(self.tokenizer)}"
        )

        # Get token IDs for each register
        for token in self.register_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            register_name = token.split(": ")[1].rstrip("]")
            self.register_token_ids[register_name] = token_id
            logger.debug(f"Register token '{register_name}': ID {token_id}")

        # Resize model embeddings to match new vocab
        if hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Model embeddings resized to {len(self.tokenizer)}")

    def _format_instruction_with_register(
        self, instruction: str, register: str, context: Optional[str] = None
    ) -> str:
        """
        Prepend register token to instruction.

        Args:
            instruction: The user instruction
            register: Register name (e.g., "formal", "legal")
            context: Optional context/system prompt

        Returns:
            Formatted instruction with prepended register token
        """
        if not self.prepend_register_token:
            return instruction

        if register not in self.register_config.registers:
            logger.warning(f"Unknown register: {register}. Using 'formal' as default.")
            register = "formal"

        register_token = f"[REGISTER: {register}]"

        if context:
            return f"{register_token} [CONTEXT: {context}] {instruction}"
        else:
            return f"{register_token} {instruction}"

    def _prepare_dataset_with_registers(self, dataset):
        """
        Prepare dataset with register tokens prepended to instructions.

        Args:
            dataset: Input dataset with 'instruction', 'response', and optional 'register' columns

        Returns:
            Dataset with register tokens integrated
        """
        if self.register_column not in dataset.column_names:
            logger.warning(
                f"Column '{self.register_column}' not found in dataset. "
                f"Using 'formal' as default register for all samples."
            )
            # Add default register column
            dataset = dataset.map(
                lambda x: {**x, self.register_column: "formal"},
                desc="Adding default register column"
            )

        def format_with_register(examples):
            """Format examples with register tokens."""
            formatted_texts = []

            for i in range(len(examples.get("instruction", examples.get("text", [])))):
                instruction = examples.get("instruction", [""])[i] if "instruction" in examples else ""
                response = examples.get("response", examples.get("output", [""]))[i]
                context = examples.get("context", [""])[i] if "context" in examples else None
                register = examples.get(self.register_column, ["formal"])[i]

                # Prepend register token
                formatted_instruction = self._format_instruction_with_register(
                    instruction, register, context
                )

                if formatted_instruction and response:
                    text = f"{formatted_instruction}\n{response}"
                else:
                    text = examples.get("text", [""])[i] if i < len(examples.get("text", [])) else ""

                formatted_texts.append(text)

            return {"text": formatted_texts}

        dataset = dataset.map(
            format_with_register,
            batched=True,
            desc="Formatting with register tokens"
        )

        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute combined loss: LM loss + optional register classification loss.

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

        if self.register_loss_weight == 0 or self.register_column not in inputs:
            # No auxiliary loss, return base loss
            if return_outputs:
                return lm_loss, outputs
            return lm_loss

        # Compute auxiliary register classification loss (optional)
        # This loss encourages the model to properly use the register token
        try:
            register_ids = inputs.get("register_token_ids", None)

            if register_ids is not None:
                # Predict which register should have been used
                # based on output token distributions
                predicted_registers = self._predict_register_from_logits(outputs.logits)
                register_loss = F.cross_entropy(
                    predicted_registers,
                    register_ids,
                    reduction="mean"
                )

                # Combined loss
                total_loss = lm_loss + self.register_loss_weight * register_loss

                logger.debug(
                    f"LM Loss: {lm_loss:.4f}, Register Loss: {register_loss:.4f}, "
                    f"Total: {total_loss:.4f}"
                )

                if return_outputs:
                    return total_loss, outputs
                return total_loss
        except Exception as e:
            logger.warning(f"Error computing register loss, using LM loss only: {e}")

        if return_outputs:
            return lm_loss, outputs
        return lm_loss

    def _predict_register_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Predict register from output logits.

        Simple heuristic: look at which register token appears early in the sequence.
        This is a placeholder for more sophisticated register prediction.

        Args:
            logits: Output logits from model

        Returns:
            Predicted register token IDs
        """
        batch_size = logits.shape[0]
        predicted_registers = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        # Check for register token presence in first 10 tokens
        register_token_ids_list = list(self.register_token_ids.values())

        for batch_idx in range(batch_size):
            # Get argmax predictions for first 10 tokens
            first_tokens = logits[batch_idx, :10].argmax(dim=1)

            # Check if any register token appears
            for token_id in register_token_ids_list:
                if token_id in first_tokens:
                    predicted_registers[batch_idx] = list(self.register_token_ids.values()).index(token_id)
                    break

        return predicted_registers


class RegisterTrainer:
    """
    Wrapper class for easy register-controlled SFT training.

    Handles model loading, dataset preparation, and training lifecycle.
    """

    def __init__(self, config: RegisterTrainerConfig):
        """
        Initialize Register Trainer.

        Args:
            config: RegisterTrainerConfig instance
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        logger.info("RegisterTrainer initialized")

    def load_model_and_tokenizer(self):
        """Load pretrained model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded: {self.model}")
        logger.info(f"Tokenizer loaded, vocab size: {len(self.tokenizer)}")

    def setup_trainer(self, train_dataset, eval_dataset=None):
        """
        Setup RegisterControlledSFTTrainer.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        from transformers import TrainingArguments

        # Prepare dataset with register tokens
        train_dataset = self.trainer._prepare_dataset_with_registers(train_dataset) if self.trainer else None

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_strategy="epoch",
            logging_steps=10,
        )

        # Create trainer (this initialization will happen inside trainer)
        self.trainer = RegisterControlledSFTTrainer(
            config=self.config,
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        logger.info("RegisterControlledSFTTrainer setup complete")

    def train(self):
        """Execute training."""
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")

        logger.info("Starting register-controlled SFT training...")
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
