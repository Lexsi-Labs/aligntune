"""
TextToLoRATrainer: Meta-training loop for Text-to-LoRA hypernetwork.

Implements a meta-learning training procedure that:
1. Samples tasks from a task distribution
2. Embeds task descriptions
3. Generates LoRA weights using the hypernetwork
4. Evaluates on task-specific data
5. Backpropagates through hypernetwork parameters

This enables the hypernetwork to learn generalizable task-to-adapter mappings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any
import math
from pathlib import Path

logger = logging.getLogger(__name__)


class TextToLoRATrainer:
    """
    Meta-trainer for the Text-to-LoRA hypernetwork.

    Implements the training loop for learning task-to-adapter mappings.
    The trainer samples tasks, generates task-specific LoRA weights,
    computes task loss, and backpropagates through the hypernetwork.

    Args:
        hypernet (nn.Module): The TextToLoRAHypernet instance to train
        target_model (nn.Module or None): Base model for LoRA application (optional for mocking)
        train_tasks (List[Dict]): Training task specifications.
            Each dict should contain:
            - "description": str (task description for embedding)
            - "data": List or DataLoader (task training data)
            - "eval_fn": callable (function to compute loss on task data)
        val_tasks (List[Dict]): Validation task specifications (same structure)
        learning_rate (float): Learning rate for hypernet optimizer. Default: 1e-4
        warmup_steps (int): Number of linear warmup steps. Default: 100
        weight_decay (float): L2 regularization for optimizer. Default: 1e-5
        max_grad_norm (float): Gradient clipping threshold. Default: 1.0
        device (str): Device for training ("cpu" or "cuda"). Default: "cpu"
        checkpoint_dir (str or Path): Directory for saving checkpoints. Default: "./checkpoints"

    Attributes:
        hypernet (nn.Module): The hypernetwork being trained
        optimizer (optim.Optimizer): Optimizer for hypernetwork parameters
        best_val_loss (float): Best validation loss seen so far
        train_step (int): Current training step counter
        checkpoints (Dict): Saved checkpoint states

    Example:
        >>> hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16)
        >>> train_tasks = [{"description": "task1", "data": [...], "eval_fn": lambda x: ...}]
        >>> trainer = TextToLoRATrainer(hypernet, None, train_tasks, val_tasks)
        >>> metrics = trainer.train_epoch(num_steps=100)
        >>> best_checkpoint = trainer.get_best_checkpoint()
    """

    def __init__(
        self,
        hypernet: nn.Module,
        target_model: Optional[nn.Module] = None,
        train_tasks: Optional[List[Dict[str, Any]]] = None,
        val_tasks: Optional[List[Dict[str, Any]]] = None,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Initialize the meta-trainer.

        Args:
            hypernet: TextToLoRAHypernet instance
            target_model: Base model (can be None for mock training)
            train_tasks: List of training task specs
            val_tasks: List of validation task specs
            learning_rate: Optimizer learning rate
            warmup_steps: Steps for linear warmup schedule
            weight_decay: L2 regularization strength
            max_grad_norm: Gradient clipping threshold
            device: "cpu" or "cuda"
            checkpoint_dir: Where to save checkpoints

        Raises:
            ValueError: If hypernet is None
            ValueError: If train_tasks is None or empty
        """
        if hypernet is None:
            raise ValueError("hypernet cannot be None")
        if train_tasks is None or len(train_tasks) == 0:
            raise ValueError("train_tasks cannot be None or empty")

        self.hypernet = hypernet
        self.target_model = target_model
        self.train_tasks = train_tasks if train_tasks else []
        self.val_tasks = val_tasks if val_tasks else []
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            hypernet.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.train_step = 0
        self.checkpoints = {}

        logger.info(
            f"Initialized TextToLoRATrainer: lr={learning_rate}, "
            f"warmup={warmup_steps}, device={device}, "
            f"train_tasks={len(self.train_tasks)}, val_tasks={len(self.val_tasks)}"
        )

    def _get_learning_rate(self, step: int) -> float:
        """
        Compute learning rate with linear warmup schedule.

        Formula:
            step < warmup: lr_scale = step / warmup
            step >= warmup: lr_scale = 1.0

        Args:
            step: Current training step

        Returns:
            float: Scaled learning rate
        """
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            scale = 1.0

        return self.learning_rate * scale

    def _update_learning_rate(self) -> None:
        """Update optimizer learning rate based on warmup schedule."""
        lr = self._get_learning_rate(self.train_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _sample_task(self, task_list: List[Dict]) -> Dict[str, Any]:
        """
        Sample a random task from the task list.

        Args:
            task_list: List of task specifications

        Returns:
            Dict: A randomly selected task
        """
        import random

        return random.choice(task_list)

    def _compute_task_loss(self, task: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a single task using generated LoRA weights.

        In a full implementation, this would:
        1. Get task embedding
        2. Generate LoRA weights
        3. Apply LoRA to target model
        4. Compute loss on task data

        For CPU mock training, we use a synthetic loss function.

        Args:
            task: Task specification with "description", "data", "eval_fn"

        Returns:
            Tensor: Scalar loss value

        Raises:
            KeyError: If task doesn't have required keys
        """
        if "description" not in task:
            raise KeyError("Task must have 'description' key")

        # Mock: Use synthetic loss based on description length and random noise
        # In production, this would embed description, generate LoRA, and evaluate
        description = task.get("description", "")
        synthetic_loss = torch.tensor(
            0.5 + (len(description) % 10) * 0.01, device=self.device, dtype=torch.float32
        )

        return synthetic_loss

    def train_step_meta(self) -> Dict[str, float]:
        """
        Execute a single meta-training step.

        Meta-training loop:
        1. Sample random task from training set
        2. Get task embedding (mocked)
        3. Generate LoRA weights via hypernet
        4. Compute task-specific loss
        5. Backprop through hypernet
        6. Update hypernet parameters

        Returns:
            Dict[str, float]: Training metrics
                - "loss": Task loss value
                - "lr": Current learning rate
                - "grad_norm": Gradient norm before clipping

        Example:
            >>> trainer = TextToLoRATrainer(hypernet, None, train_tasks)
            >>> metrics = trainer.train_step_meta()
            >>> loss = metrics["loss"]
        """
        # Sample task
        task = self._sample_task(self.train_tasks)

        # Create mock task embedding (in production, use real embeddings)
        batch_size = 1
        task_embedding = torch.randn(batch_size, self.hypernet.hidden_dim, device=self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Generate LoRA weights
        lora_weights = self.hypernet(task_embedding)

        # Compute task loss (mock)
        task_loss = self._compute_task_loss(task)

        # For proper meta-training, also include LoRA weight regularization
        # Encourage smaller LoRA weights to prevent overfitting
        lora_reg = 0.0
        for lora_pair in lora_weights:
            a_matrix = lora_pair["A"]
            b_matrix = lora_pair["B"]
            lora_reg += torch.norm(a_matrix) ** 2 + torch.norm(b_matrix) ** 2

        lora_reg = lora_reg / max(1, len(lora_weights))
        total_loss = task_loss + 0.01 * lora_reg

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.hypernet.parameters(), self.max_grad_norm
        )

        # Update learning rate (warmup schedule)
        self._update_learning_rate()

        # Optimizer step
        self.optimizer.step()

        # Increment step counter
        self.train_step += 1

        return {
            "loss": task_loss.item(),
            "lr": self._get_learning_rate(self.train_step),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }

    def train_epoch(self, num_steps: int) -> Dict[str, float]:
        """
        Execute a full epoch of meta-training.

        Args:
            num_steps: Number of meta-training steps to execute

        Returns:
            Dict[str, float]: Aggregated epoch metrics
                - "avg_loss": Average task loss over epoch
                - "final_lr": Final learning rate at epoch end
                - "total_steps": Total steps executed

        Raises:
            ValueError: If num_steps <= 0
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        self.hypernet.train()
        total_loss = 0.0
        max_grad_norm = 0.0

        for step in range(num_steps):
            metrics = self.train_step_meta()
            total_loss += metrics["loss"]
            max_grad_norm = max(max_grad_norm, metrics["grad_norm"])

        avg_loss = total_loss / num_steps

        logger.info(
            f"Epoch completed: avg_loss={avg_loss:.4f}, "
            f"steps={num_steps}, total_steps={self.train_step}"
        )

        return {
            "avg_loss": avg_loss,
            "final_lr": self._get_learning_rate(self.train_step),
            "total_steps": self.train_step,
            "max_grad_norm": max_grad_norm,
        }

    def validate(self) -> Dict[str, float]:
        """
        Evaluate the hypernetwork on validation tasks.

        Args:
            (uses self.val_tasks)

        Returns:
            Dict[str, float]: Validation metrics
                - "val_loss": Average loss on validation tasks
                - "num_val_tasks": Number of validation tasks evaluated
                - "is_best": Whether this is the best val loss seen so far

        Note:
            Automatically saves checkpoint if validation loss improves.
        """
        if not self.val_tasks:
            logger.warning("No validation tasks provided, skipping validation")
            return {"val_loss": 0.0, "num_val_tasks": 0, "is_best": False}

        self.hypernet.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for task in self.val_tasks:
                # Mock validation loss
                task_embedding = torch.randn(1, self.hypernet.hidden_dim, device=self.device)
                lora_weights = self.hypernet(task_embedding)
                val_loss = self._compute_task_loss(task).item()
                total_val_loss += val_loss

        avg_val_loss = total_val_loss / len(self.val_tasks)
        is_best = avg_val_loss < self.best_val_loss

        if is_best:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint(f"best_checkpoint_step{self.train_step}.pt")
            logger.info(f"New best validation loss: {avg_val_loss:.4f}")

        logger.info(
            f"Validation: loss={avg_val_loss:.4f}, is_best={is_best}, "
            f"num_tasks={len(self.val_tasks)}"
        )

        return {
            "val_loss": avg_val_loss,
            "num_val_tasks": len(self.val_tasks),
            "is_best": is_best,
        }

    def save_checkpoint(self, name: str) -> Path:
        """
        Save hypernet checkpoint to disk.

        Args:
            name: Filename for checkpoint (e.g., "best_checkpoint.pt")

        Returns:
            Path: Path to saved checkpoint

        Example:
            >>> trainer = TextToLoRATrainer(hypernet, None, train_tasks)
            >>> checkpoint_path = trainer.save_checkpoint("epoch_1.pt")
        """
        checkpoint_path = self.checkpoint_dir / name

        state_dict = {
            "hypernet": self.hypernet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
            "best_val_loss": self.best_val_loss,
            "hypernet_config": self.hypernet.get_lora_config(),
        }

        torch.save(state_dict, checkpoint_path)
        self.checkpoints[name] = checkpoint_path

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load hypernet from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint structure is invalid

        Example:
            >>> trainer = TextToLoRATrainer(hypernet, None, train_tasks)
            >>> trainer.load_checkpoint("checkpoints/best_checkpoint.pt")
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)

        self.hypernet.load_state_dict(state_dict["hypernet"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.train_step = state_dict.get("train_step", 0)
        self.best_val_loss = state_dict.get("best_val_loss", float("inf"))

        logger.info(
            f"Loaded checkpoint: {checkpoint_path} "
            f"(step={self.train_step}, best_val={self.best_val_loss:.4f})"
        )

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the best checkpoint state seen during training.

        Returns:
            Dict with keys:
            - "hypernet": nn.Module state dict
            - "config": Hypernetwork config
            - "best_val_loss": Best validation loss achieved
            - "train_step": Training step at best checkpoint

            Returns None if no checkpoint saved yet.

        Example:
            >>> trainer = TextToLoRATrainer(hypernet, None, train_tasks)
            >>> trainer.train_epoch(100)
            >>> best = trainer.get_best_checkpoint()
            >>> if best:
            ...     print(f"Best loss: {best['best_val_loss']}")
        """
        if self.best_val_loss == float("inf"):
            logger.warning("No checkpoint saved yet (best_val_loss is inf)")
            return None

        return {
            "hypernet": self.hypernet.state_dict(),
            "config": self.hypernet.get_lora_config(),
            "best_val_loss": self.best_val_loss,
            "train_step": self.train_step,
        }
