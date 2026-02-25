from .meta_es_trainer_utils import *
from .vllm_evaluator_math import compute_value_function_gsm8k, get_gsm8k_splits
from .vllm_evaluator import compute_value_function_vllm
import os
import sys
import argparse
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
import wandb
import logging


logger = logging.getLogger(__name__)


class TRLMetaEsTrainer:
    def __init__(self, config):
        self.config = config
        self.state = None
        self.inner_train = None
        self.validation = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if required dependencies are available for Meta-ES training."""
        try:
            import torch
            import numpy as np
            return True
        except ImportError:
            return False

    def setup_data(self):
        """Setup datasets based on dataset name."""
        dataset_name = self._get_config_value(
            self.config.datasets, 'name', 'dataset_name', default='mbpp')

        logger.info(f"📚 Loading {dataset_name} dataset splits...")

        if dataset_name.lower() == 'gsm8k':
            self.inner_train, self.validation = get_gsm8k_splits(
                test_size=0.2, seed=42)
        elif dataset_name.lower() == 'mbpp':
            self.inner_train, self.validation = get_mbpp_splits(
                test_size=0.2, seed=42)
        else:
            # Normal dataset setup for other datasets
            from datasets import load_dataset
            split = self._get_config_value(
                self.config.dataset, 'split', default='train')

            try:
                dataset = load_dataset(dataset_name, split=split)
            except ValueError as e:
                if "Config name is missing" in str(e):
                    logger.info(
                        f"Dataset requires config, trying with 'main' config...")
                    try:
                        dataset = load_dataset(
                            dataset_name, 'main', split=split)
                    except BaseException:
                        logger.info(f"Trying alternative loading method...")
                        dataset = load_dataset(dataset_name, 'main')
                        if split and split in dataset:
                            dataset = dataset[split]
                        else:
                            dataset = dataset['train']
                else:
                    raise

            # Split into train and validation
            from sklearn.model_selection import train_test_split
            dataset_list = list(dataset)
            self.inner_train, self.validation = train_test_split(
                dataset_list, test_size=0.2, random_state=42
            )

        validation_list = list(self.validation)
        logger.info(f"   Inner train: {len(self.inner_train)} problems")
        logger.info(f"   Validation: {len(validation_list)} problems\n")

    def train(self):
        """Execute Meta-ES training."""
        # Load or initialize ES state
        resume_path = self._get_config_value(
            self.config.train, 'resume', default=None)

        if resume_path:
            logger.info(f"📂 Resuming from checkpoint: {resume_path}")
            self.state = load_es_checkpoint(resume_path)
        else:
            seed = self._get_config_value(
                self.config.train, 'seed', default=42)
            init_scale = self._get_config_value(
                self.config.train, 'init_scale', default=0.01)
            logger.info(f"🎲 Initializing ES state with seed={seed}")
            self.state = initialize_es_state(
                dim=380, init_scale=init_scale, seed=seed)

        # Setup datasets
        self.setup_data()

        # Get training parameters
        meta_iterations = self._get_config_value(
            self.config.train, 'meta_iterations', default=15)
        patience = self._get_config_value(
            self.config.train, 'patience', default=5)
        min_delta = self._get_config_value(
            self.config.train, 'min_delta', default=0.001)

        validation_list = list(self.validation)

        # ES training loop with progress bar
        pbar = tqdm(
            range(self.state.meta_iteration, meta_iterations),
            desc="ES Iterations",
            initial=self.state.meta_iteration,
            total=meta_iterations,
            unit="it",
        )

        for iteration in pbar:
            # Run ES iteration
            self.state = run_es_iteration(
                self.state, self.config, validation_list)

            # Update progress bar with current best fitness
            pbar.set_postfix(
                {"best_fitness": f"{self.state.best_fitness:.4f}"})

        pbar.close()

        # Final summary
        output_dir = self._get_config_value(
            self.config.logging,
            'output_dir',
            default='./es_meta_learning')

        logger.info(f"\n{'=' * 60}")
        logger.info("ES Meta-Learning Complete!")
        logger.info(f"{'=' * 60}")
        logger.info(f"Final best fitness: {self.state.best_fitness:.4f}")
        logger.info(f"Total iterations: {self.state.meta_iteration}")
        logger.info(
            f"Best mirror map saved to: {output_dir}/best_mirror_map_params.pt")
        logger.info(f"{'=' * 60}\n")

        return {
            "best_fitness": self.state.best_fitness,
            "total_iterations": self.state.meta_iteration,
            "model_path": output_dir,
        }

    def _get_config_value(self, config_obj, *attr_names, default=None):
        """Safely get config value from multiple possible attribute names."""
        if isinstance(config_obj, dict):
            for attr_name in attr_names:
                if attr_name in config_obj:
                    return config_obj[attr_name]
        else:
            for attr_name in attr_names:
                if hasattr(config_obj, attr_name):
                    return getattr(config_obj, attr_name)
        return default
