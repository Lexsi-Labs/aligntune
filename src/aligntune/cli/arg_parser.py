"""
CLI Argument Parser Module for AlignTune.


This module provides modular argument parsing functions for different
components of the CLI, extracted from the main.py file.
"""

import argparse
from typing import Dict, Any, Optional, List


def add_base_args(parser: argparse.ArgumentParser) -> None:
    """Add base arguments common to all training types."""
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    model_group.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization"
    )
    model_group.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    model_group.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization"
    )
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The output directory where the model predictions and checkpoints will be written"
    )
    output_group.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory"
    )
    
    # Logging configuration
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="Tensorboard log dir"
    )
    logging_group.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X updates steps"
    )
    logging_group.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps"
    )
    logging_group.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X updates steps"
    )
    logging_group.add_argument(
        "--wandb_project",
        type=str,
        help="Weights & Biases project name"
    )
    logging_group.add_argument(
        "--wandb_run_name",
        type=str,
        help="Weights & Biases run name"
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-specific arguments."""
    
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_type",
        type=str,
        help="Model type (e.g., 'gpt2', 'llama')"
    )
    model_group.add_argument(
        "--config_name",
        type=str,
        help="Pretrained config name or path if not the same as model_name"
    )
    model_group.add_argument(
        "--tokenizer_name",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    model_group.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use one of the fast tokenizer"
    )
    model_group.add_argument(
        "--cache_dir",
        type=str,
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )
    model_group.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="The specific model version to use (can be a branch name, tag name or commit id)"
    )
    model_group.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Will use the token generated when running `huggingface-cli login`"
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training-specific arguments."""
    
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform"
    )
    training_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/TPU core/CPU for training"
    )
    training_group.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/TPU core/CPU for evaluation"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="The initial learning rate for AdamW optimizer"
    )
    training_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer"
    )
    training_group.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer"
    )
    training_group.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer"
    )
    training_group.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer"
    )
    training_group.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm"
    )
    training_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Linear warmup over warmup_ratio fraction of total steps"
    )
    training_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="The scheduler type to use"
    )
    training_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    training_group.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision instead of 32-bit"
    )
    training_group.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 (mixed) precision instead of 32-bit"
    )
    training_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading"
    )
    training_group.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        help="Whether or not to pin memory for DataLoader"
    )
    training_group.add_argument(
        "--remove_unused_columns",
        action="store_true",
        help="Remove columns not required by the model when using an nlp.Dataset"
    )
    training_group.add_argument(
        "--label_smoothing_factor",
        type=float,
        default=0.0,
        help="The label smoothing epsilon to apply"
    )
    training_group.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0: set total number of training steps to perform"
    )
    training_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass"
    )


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add dataset-specific arguments."""
    
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to use"
    )
    dataset_group.add_argument(
        "--dataset_config_name",
        type=str,
        help="The configuration name of the dataset to use"
    )
    dataset_group.add_argument(
        "--train_file",
        type=str,
        help="The input training data file (a jsonlines or csv file)"
    )
    dataset_group.add_argument(
        "--validation_file",
        type=str,
        help="An optional input validation data file (a jsonlines or csv file)"
    )
    dataset_group.add_argument(
        "--test_file",
        type=str,
        help="An optional input test data file (a jsonlines or csv file)"
    )
    dataset_group.add_argument(
        "--max_train_samples",
        type=int,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value"
    )
    dataset_group.add_argument(
        "--max_eval_samples",
        type=int,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value"
    )
    dataset_group.add_argument(
        "--max_test_samples",
        type=int,
        help="For debugging purposes or quicker training, truncate the number of test examples to this value"
    )
    dataset_group.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming mode"
    )
    dataset_group.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
    dataset_group.add_argument(
        "--validation_split_percentage",
        type=int,
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split"
    )
    dataset_group.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing"
    )
    dataset_group.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization"
    )
    dataset_group.add_argument(
        "--packing",
        action="store_true",
        help="Whether to pack sequences together"
    )


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    """Add backend-specific arguments."""
    
    backend_group = parser.add_argument_group("Backend Configuration")
    backend_group.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "trl", "unsloth"],
        help="Backend to use for training"
    )
    backend_group.add_argument(
        "--algorithm",
        type=str,
        choices=["dpo", "ppo", "grpo", "gspo"],
        help="RL algorithm to use (for RL training)"
    )
    backend_group.add_argument(
        "--training_type",
        type=str,
        default="sft",
        choices=["sft", "dpo", "ppo", "grpo", "gspo"],
        help="Type of training to perform"
    )


def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    """Add evaluation-specific arguments."""
    
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run evaluation on the validation set"
    )
    eval_group.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set"
    )
    eval_group.add_argument(
        "--evaluation_strategy",
        type=str,
        default="no",
        choices=["no", "steps", "epoch"],
        help="The evaluation strategy to use"
    )
    eval_group.add_argument(
        "--metric_for_best_model",
        type=str,
        help="The metric to use to compare two different models"
    )
    eval_group.add_argument(
        "--greater_is_better",
        action="store_true",
        help="Whether the `metric_for_best_model` should be maximized or not"
    )
    eval_group.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Whether or not to load the best model found during training at the end of training"
    )
    eval_group.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="The checkpoint save strategy to use"
    )


def create_base_parser() -> argparse.ArgumentParser:
    """Create a base argument parser with common arguments."""
    
    parser = argparse.ArgumentParser(
        description="AlignTune: A comprehensive fine-tuning library for SFT and RL training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    add_base_args(parser)
    add_model_args(parser)
    add_training_args(parser)
    add_dataset_args(parser)
    add_backend_args(parser)
    add_evaluation_args(parser)
    
    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = create_base_parser()
    return parser.parse_args(args)


def args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert parsed arguments to dictionary."""
    
    return vars(args)
