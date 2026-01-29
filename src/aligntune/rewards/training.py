"""
Reward Model Training System for AlignTune.


This module provides comprehensive reward model training capabilities with strict validation,
no fallbacks, and production-ready implementations using TRL's RewardTrainer.
"""

import logging
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from trl import RewardTrainer, RewardConfig
import numpy as np
from dataclasses import dataclass

from .core import CompositeReward, RewardFunction
from .registry import RewardRegistry

logger = logging.getLogger(__name__)


class RewardModelDataset(Dataset):
    """
    PyTorch Dataset for reward model training data.
    
    Handles tokenization, padding, and reward score preparation with strict validation.
    """
    
    def __init__(
        self,
        texts: List[str],
        reward_scores: List[float],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        reference_texts: Optional[List[str]] = None
    ):
        """
        Initialize dataset with strict validation.
        
        Args:
            texts: List of input texts
            reward_scores: List of reward scores (must match texts length)
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            reference_texts: Optional reference texts for comparison
        """
        # Strict validation
        if not texts:
            raise ValueError("texts cannot be empty")
        if not reward_scores:
            raise ValueError("reward_scores cannot be empty")
        if len(texts) != len(reward_scores):
            raise ValueError(
                f"texts and reward_scores must have same length. "
                f"Got {len(texts)} texts and {len(reward_scores)} scores"
            )
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        # Validate reward scores are numeric
        for i, score in enumerate(reward_scores):
            if not isinstance(score, (int, float)):
                raise TypeError(f"reward_scores[{i}] must be numeric, got {type(score)}")
            if np.isnan(score) or np.isinf(score):
                raise ValueError(f"reward_scores[{i}] is invalid: {score}")
        
        # Validate reference texts if provided
        if reference_texts is not None:
            if len(reference_texts) != len(texts):
                raise ValueError(
                    f"reference_texts length ({len(reference_texts)}) "
                    f"must match texts length ({len(texts)})"
                )
        
        self.texts = texts
        self.reward_scores = reward_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reference_texts = reference_texts
        
        logger.info(f"Created RewardModelDataset with {len(texts)} samples")
        logger.info(f"Reward score range: [{min(reward_scores):.3f}, {max(reward_scores):.3f}]")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item with tokenization and error handling.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        if idx >= len(self.texts):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.texts)}")
        
        text = self.texts[idx]
        reward_score = self.reward_scores[idx]
        
        # Tokenize text
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        except Exception as e:
            raise RuntimeError(f"Tokenization failed for text at index {idx}: {e}") from e
        
        # Prepare labels (reward scores)
        labels = torch.tensor([reward_score], dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


class RewardModelValidator:
    """
    Strict validation for reward model configurations.
    
    No fallbacks, no assumptions - fail fast with clear errors.
    """
    
    @staticmethod
    def validate_reward_source(source_config) -> None:
        """
        Validate reward model source with NO fallbacks.
        
        Args:
            source_config: RewardModelSourceConfig to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not source_config:
            raise ValueError("reward_model_source must be specified")
        
        sources = [
            source_config.model_name,
            source_config.model_path,
            source_config.training_config
        ]
        source_count = sum(x is not None for x in sources)
        
        if source_count == 0:
            raise ValueError(
                "Must specify exactly one reward source: "
                "model_name (HF Hub), model_path (local), or training_config (custom)"
            )
        if source_count > 1:
            raise ValueError(
                f"Multiple reward sources specified. Found: "
                f"model_name={source_config.model_name}, "
                f"model_path={source_config.model_path}, "
                f"training_config={source_config.training_config}. "
                "Specify exactly ONE."
            )
    
    @staticmethod
    def validate_training_config(config) -> None:
        """
        Validate training configuration with NO defaults.
        
        Args:
            config: RewardModelTrainingConfig to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not config.training_texts:
            raise ValueError("training_texts cannot be empty")
        if not isinstance(config.training_texts, list):
            raise TypeError(f"training_texts must be list, got {type(config.training_texts)}")
        if len(config.training_texts) < 10:
            raise ValueError(f"Need at least 10 training texts, got {len(config.training_texts)}")
        if not config.reward_functions:
            raise ValueError("reward_functions cannot be empty")
        if not config.base_model_name:
            raise ValueError("base_model_name cannot be empty")
        if not config.output_dir:
            raise ValueError("output_dir cannot be empty")
        
        # Validate reward functions exist in registry
        for func_name in config.reward_functions:
            try:
                RewardRegistry.get_reward_function(func_name)
            except KeyError:
                available = list(RewardRegistry.list_reward_functions().keys())
                raise ValueError(
                    f"Reward function '{func_name}' not found in registry. "
                    f"Available: {available}"
                )
        
        # Validate optional fields if provided
        if config.reference_texts and len(config.reference_texts) != len(config.training_texts):
            raise ValueError("reference_texts length must match training_texts")
        if config.reward_weights:
            if len(config.reward_weights) != len(config.reward_functions):
                raise ValueError("reward_weights length must match reward_functions")
            if not all(w > 0 for w in config.reward_weights):
                raise ValueError("All reward_weights must be positive")
        
        # Validate training params
        if config.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {config.num_epochs}")
        if config.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {config.batch_size}")
        if config.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {config.max_length}")
    
    @staticmethod
    def validate_hf_model_access(model_name: str) -> None:
        """
        Validate HuggingFace model is accessible.
        
        Args:
            model_name: Model name to validate
            
        Raises:
            ValueError: If model is not accessible
        """
        try:
            from huggingface_hub import model_info
            info = model_info(model_name)
            if not info:
                raise ValueError(f"Model {model_name} not found on HuggingFace Hub")
            logger.info(f"✅ Model {model_name} is accessible on HuggingFace Hub")
        except Exception as e:
            raise ValueError(f"Cannot access model {model_name} on HuggingFace Hub: {e}") from e
    
    @staticmethod
    def validate_local_path(path: str) -> None:
        """
        Validate local path exists and is readable.
        
        Args:
            path: Path to validate
            
        Raises:
            FileNotFoundError: If path doesn't exist
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path_obj.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        # Check for required model files
        required_files = ['config.json', 'pytorch_model.bin']
        for file_name in required_files:
            file_path = path_obj / file_name
            if not file_path.exists():
                # Try alternative names
                alt_files = ['model.safetensors', 'pytorch_model-00001-of-00001.bin']
                found = any((path_obj / alt).exists() for alt in alt_files)
                if not found:
                    raise FileNotFoundError(f"Required model file not found: {file_name}")
        
        logger.info(f"✅ Local model path validated: {path}")


class RewardModelLoader:
    """
    Comprehensive reward model loader with strict validation.
    
    Handles HF Hub, local, and custom trained models with proper error handling.
    """
    
    def __init__(self):
        """Initialize loader."""
        self.logger = logging.getLogger(__name__)
    
    def load_from_huggingface(self, model_name: str, **kwargs) -> AutoModelForSequenceClassification:
        """
        Load reward model from HuggingFace Hub with validation.
        
        Args:
            model_name: Model name on HuggingFace Hub
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded reward model
            
        Raises:
            ValueError: If model loading fails
        """
        self.logger.info(f"Loading reward model from HuggingFace Hub: {model_name}")
        
        # Validate model access
        RewardModelValidator.validate_hf_model_access(model_name)
        
        try:
            # Load tokenizer first to validate
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
            
            # Load model with validation
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,  # Reward models typically have 1 output
                trust_remote_code=kwargs.get('trust_remote_code', False),
                torch_dtype=kwargs.get('torch_dtype', torch.float16),
                device_map=kwargs.get('device_map', 'auto'),
                ignore_mismatched_sizes=kwargs.get('ignore_mismatched_sizes', False)
            )
            
            # Validate model architecture
            if model.config.num_labels != 1:
                self.logger.warning(
                    f"Model has {model.config.num_labels} labels, expected 1 for reward model"
                )
            
            self.logger.info(f"✅ Reward model loaded successfully: {model.config.architectures}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e
    
    def load_from_local(self, model_path: str, **kwargs) -> AutoModelForSequenceClassification:
        """
        Load reward model from local path with validation.
        
        Args:
            model_path: Local path to model directory
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded reward model
            
        Raises:
            FileNotFoundError: If path doesn't exist
        """
        self.logger.info(f"Loading reward model from local path: {model_path}")
        
        # Validate path
        RewardModelValidator.validate_local_path(model_path)
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.logger.info(f"✅ Tokenizer loaded from {model_path}")
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=1,
                trust_remote_code=kwargs.get('trust_remote_code', False),
                torch_dtype=kwargs.get('torch_dtype', torch.float16),
                device_map=kwargs.get('device_map', 'auto')
            )
            
            self.logger.info(f"✅ Local reward model loaded successfully")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load local model from {model_path}: {e}") from e


class RewardModelTrainer:
    """
    Main reward model trainer using TRL's RewardTrainer.
    
    Generates training data from reward functions and trains neural reward models.
    """
    
    def __init__(
        self,
        base_model_name: str,
        reward_functions: Optional[List[RewardFunction]] = None,
        composite_weights: Optional[List[float]] = None
    ):
        """
        Initialize trainer with strict validation.
        
        Args:
            base_model_name: Base model for reward model training
            reward_functions: List of reward functions to use
            composite_weights: Weights for composite reward
        """
        self.logger = logging.getLogger(__name__)
        
        # Validate inputs
        if not base_model_name:
            raise ValueError("base_model_name cannot be empty")
        
        self.base_model_name = base_model_name
        self.reward_functions = reward_functions or []
        self.composite_weights = composite_weights
        
        # Create composite reward if functions provided
        if self.reward_functions:
            self.composite_reward = CompositeReward(
                self.reward_functions,
                self.composite_weights
            )
            self.logger.info(f"Created composite reward with {len(self.reward_functions)} functions")
        else:
            self.composite_reward = None
            self.logger.warning("No reward functions provided - will need manual reward scores")
    
    def generate_training_data(
        self,
        texts: List[str],
        references: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> RewardModelDataset:
        """
        Generate training data using composite reward functions.
        
        Args:
            texts: List of input texts
            references: Optional reference texts
            batch_size: Batch size for reward computation
            
        Returns:
            RewardModelDataset with computed rewards
            
        Raises:
            ValueError: If no reward functions available
        """
        if not self.composite_reward:
            raise ValueError("No reward functions available for training data generation")
        
        self.logger.info(f"Generating training data for {len(texts)} texts")
        
        # Compute rewards in batches
        try:
            if references:
                reward_scores = self.composite_reward.batch_compute(
                    texts, references, batch_size=batch_size
                )
            else:
                reward_scores = self.composite_reward.batch_compute(
                    texts, batch_size=batch_size
                )
        except Exception as e:
            raise RuntimeError(f"Failed to compute rewards: {e}") from e
        
        # Validate computed rewards
        if len(reward_scores) != len(texts):
            raise ValueError(
                f"Reward computation mismatch: {len(texts)} texts, {len(reward_scores)} scores"
            )
        
        # Check for invalid scores
        invalid_scores = [i for i, score in enumerate(reward_scores) 
                         if np.isnan(score) or np.isinf(score)]
        if invalid_scores:
            raise ValueError(f"Invalid reward scores at indices: {invalid_scores}")
        
        self.logger.info(f"✅ Generated {len(reward_scores)} reward scores")
        self.logger.info(f"Reward range: [{min(reward_scores):.3f}, {max(reward_scores):.3f}]")
        
        # Create tokenizer for dataset
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}") from e
        
        return RewardModelDataset(
            texts=texts,
            reward_scores=reward_scores,
            tokenizer=tokenizer,
            reference_texts=references
        )
    
    def train_reward_model(
        self,
        training_data: RewardModelDataset,
        output_dir: str,
        num_epochs: int = 3,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        **kwargs
    ) -> str:
        """
        Train reward model using TRL's RewardTrainer.
        
        Args:
            training_data: Training dataset
            output_dir: Directory to save trained model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            **kwargs: Additional training arguments
            
        Returns:
            Path to trained model
            
        Raises:
            RuntimeError: If training fails
        """
        self.logger.info(f"Training reward model for {num_epochs} epochs")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Validate training data
        if len(training_data) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load base model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=1,
                torch_dtype=torch.float32
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create training arguments with defaults
            default_args = {
                "logging_steps": 10,
                "save_steps": 100,
                "eval_strategy": "no",
                "save_total_limit": 2,
                "remove_unused_columns": False,
                "report_to": "none",  # Disable wandb logging
                "fp16": False,  # Disable mixed precision
                "bf16": False,  # Disable bfloat16
            }
            
            # Update defaults with kwargs (kwargs take precedence)
            default_args.update(kwargs)
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                **default_args
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=training_data,
                tokenizer=tokenizer
            )
            
            # Train model
            self.logger.info("Starting reward model training...")
            trainer.train()
            
            # Save model
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # Save metadata
            metadata = {
                "base_model": self.base_model_name,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "training_samples": len(training_data),
                "reward_functions": [f.__class__.__name__ for f in self.reward_functions] if self.reward_functions else [],
                "composite_weights": self.composite_weights
            }
            
            with open(output_path / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✅ Reward model training completed: {output_dir}")
            return output_dir
            
        except Exception as e:
            self.logger.error(f"❌ Reward model training failed: {e}")
            raise RuntimeError(f"Training failed: {e}") from e
    def train_reward_model_torch_loop(
        self,
        training_data: RewardModelDataset,
        output_dir: str,
        num_epochs: int = 3,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        **kwargs
    ) -> str:
        """
        Train reward model using pure PyTorch, with full try/except logging
        identical to the original Trainer-based implementation.
        """

        import os
        import sys
        from pathlib import Path
        from tqdm import tqdm

        # # Disable Unsloth
        # os.environ["DISABLE_UNSLOTH"] = "1"
        # os.environ["USE_UNSLOTH"] = "0"
        # for mod in list(sys.modules.keys()):
        #     if "unsloth" in mod.lower():
        #         del sys.modules[mod]

        self.logger.info(f"Training reward model (PyTorch) for {num_epochs} epochs")
        self.logger.info(f"Output directory: {output_dir}")

        if len(training_data) == 0:
            raise ValueError("Training data cannot be empty")

        try:
            # --------------------------------------------------------------
            # Setup: device, directories
            # --------------------------------------------------------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # --------------------------------------------------------------
            # Load model
            # --------------------------------------------------------------
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.base_model_name,
                    num_labels=1,
                    torch_dtype=torch.float32
                ).to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to load base model {self.base_model_name}: {e}")

            # --------------------------------------------------------------
            # Load tokenizer
            # --------------------------------------------------------------
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # --------------------------------------------------------------
            # DataLoader
            # --------------------------------------------------------------
            try:
                dataloader = torch.utils.data.DataLoader(
                    training_data,
                    batch_size=batch_size,
                    shuffle=True
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create DataLoader: {e}")

            # --------------------------------------------------------------
            # Optimizer
            # --------------------------------------------------------------
            try:
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize AdamW optimizer: {e}")

            # --------------------------------------------------------------
            # Pure PyTorch Training Loop
            # --------------------------------------------------------------
            self.logger.info("Starting PyTorch reward model training...")
            model.train()
            global_step = 0

            for epoch in tqdm(range(num_epochs)):
                epoch_loss = 0.0

                for step, batch in enumerate(dataloader):
                    try:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)
                    except Exception as e:
                        raise RuntimeError(f"Failed to move batch to device: {e}")

                    try:
                        print(input_ids.shape,attention_mask.shape)
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                    except Exception as e:
                        raise RuntimeError(f"Forward pass failed at step {step}: {e}")

                    try:
                        loss = outputs.loss / gradient_accumulation_steps
                        loss.backward()
                    except Exception as e:
                        raise RuntimeError(f"Backward pass failed at step {step}: {e}")

                    epoch_loss += loss.item()

                    if (step + 1) % gradient_accumulation_steps == 0:
                        try:
                            optimizer.step()
                            optimizer.zero_grad()
                        except Exception as e:
                            raise RuntimeError(f"Optimizer step failed at step {step}: {e}")

                        global_step += 1

                self.logger.info(f"Epoch {epoch+1}/{num_epochs} — Loss: {epoch_loss:.4f}")

            # --------------------------------------------------------------
            # Save trained model + tokenizer
            # --------------------------------------------------------------
            try:
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            except Exception as e:
                raise RuntimeError(f"Failed to save trained model/tokenizer: {e}")

            # --------------------------------------------------------------
            # Save training metadata
            # --------------------------------------------------------------
            try:
                metadata = {
                    "base_model": self.base_model_name,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "training_samples": len(training_data),
                    "reward_functions": [
                        f.__class__.__name__ for f in self.reward_functions
                    ] if self.reward_functions else [],
                    "composite_weights": self.composite_weights,
                    "trainer": "pure_pytorch"
                }

                with open(output_path / "training_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
                print(metadata)
            except Exception as e:
                raise RuntimeError(f"Failed to save training metadata: {e}")

            self.logger.info(f"✅ Reward model training completed (PyTorch): {output_dir}")
            return output_dir

        except Exception as e:
            self.logger.error(f"❌ Reward model training failed: {e}")
            raise RuntimeError(f"Training failed: {e}") from e

    
    
    def create_scalable_pipeline(
        self,
        training_texts: List[str],
        output_dir: str,
        reference_texts: Optional[List[str]] = None,
        **training_kwargs
    ) -> str:
        """
        Create scalable training pipeline.
        
        Args:
            training_texts: Texts for training
            output_dir: Output directory
            reference_texts: Optional reference texts
            **training_kwargs: Training parameters
            
        Returns:
            Path to trained model
        """
        self.logger.info("Creating scalable reward model training pipeline")
        
        # Generate training data
        training_data = self.generate_training_data(
            texts=training_texts,
            references=reference_texts,
            batch_size=training_kwargs.get('data_batch_size', 32)
        )
        
        # Train model
        model_path = self.train_reward_model(
            training_data=training_data,
            output_dir=output_dir,
            **training_kwargs
        )
        
        return model_path
