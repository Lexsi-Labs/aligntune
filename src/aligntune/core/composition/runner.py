"""
Composition runner for executing multi-stage training pipelines.

This module provides the orchestration logic to execute training stages in sequence,
managing checkpoint threading and error handling across the pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from .stages import Composition, Stage, StageResult, CompositionLoader


logger = logging.getLogger(__name__)


class CompositionRunner:
    """Orchestrates execution of multi-stage training compositions.

    This runner manages:
    - Loading and validating compositions
    - Executing stages in sequence
    - Threading checkpoints from one stage to the next
    - Tracking results and metrics
    - Error handling and reporting
    """

    def __init__(
        self,
        composition: Composition,
        base_output_dir: str,
        device: str = "cpu"
    ):
        """Initialize the composition runner.

        Args:
            composition: Composition object defining the pipeline
            base_output_dir: Base output directory for all stages
            device: Device to use ("cpu", "cuda", "cuda:0", etc.)
        """
        self.composition = composition
        self.base_output_dir = Path(base_output_dir)
        self.device = device
        self.results: List[StageResult] = []
        self.stage_checkpoints: Dict[str, str] = {}  # Maps stage names to checkpoint dirs

        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized CompositionRunner for '{composition.name}'")
        logger.info(f"  Stages: {', '.join(s.name for s in composition.stages)}")
        logger.info(f"  Output dir: {self.base_output_dir}")

    def get_stage_output_dir(self, stage_name: str) -> Path:
        """Get the output directory for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Path to the stage output directory
        """
        return self.base_output_dir / stage_name

    def run_stage(
        self,
        stage: Stage,
        prev_checkpoint_dir: Optional[str] = None
    ) -> StageResult:
        """Execute a single training stage.

        This method:
        1. Creates the stage output directory
        2. Loads the stage config
        3. Initializes from previous checkpoint if specified
        4. Creates a trainer for the algorithm
        5. Runs training
        6. Saves checkpoint
        7. Returns StageResult with metrics

        Args:
            stage: Stage object to execute
            prev_checkpoint_dir: Optional checkpoint from previous stage

        Returns:
            StageResult object with execution status and metrics
        """
        stage_name = stage.name
        logger.info(f"Starting stage: {stage_name}")

        stage_start_time = time.time()
        stage_output_dir = self.get_stage_output_dir(stage_name)
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load stage configuration
            config_path = Path(stage.config_path)
            if not config_path.is_absolute():
                # Try to resolve relative to base output dir
                config_path = self.base_output_dir.parent / stage.config_path

            if not config_path.exists():
                raise FileNotFoundError(f"Stage config not found: {config_path}")

            logger.info(f"  Loading config from: {config_path}")

            # Load configuration (implementation deferred to specific backends)
            # This would normally load YAML and parse it
            config_data = self._load_config_file(config_path)

            # Apply any target parameter overrides
            if stage.target_params:
                logger.info(f"  Applying target parameter overrides: {stage.target_params}")
                config_data = self._merge_configs(config_data, stage.target_params)

            # Handle checkpoint initialization
            init_checkpoint = None
            if stage.init_from:
                # Get checkpoint from a previous stage
                if stage.init_from not in self.stage_checkpoints:
                    raise ValueError(
                        f"Stage '{stage_name}' requested init_from '{stage.init_from}' "
                        f"but it was not found in completed stages"
                    )
                init_checkpoint = self.stage_checkpoints[stage.init_from]
                logger.info(f"  Initializing from previous checkpoint: {init_checkpoint}")
            elif prev_checkpoint_dir:
                # Use checkpoint from immediately previous stage
                init_checkpoint = prev_checkpoint_dir
                logger.info(f"  Using previous stage checkpoint: {init_checkpoint}")

            # Create trainer (implementation deferred to backend factory)
            # This is a placeholder - actual implementation uses BackendFactory
            logger.info(f"  Creating trainer for algorithm: {stage.algo}")
            trainer = self._create_trainer(
                stage=stage,
                config=config_data,
                init_checkpoint=init_checkpoint,
                output_dir=str(stage_output_dir)
            )

            # Run training
            logger.info(f"  Starting training for stage '{stage_name}'")
            metrics = self._run_trainer(trainer)

            # Save checkpoint
            checkpoint_dir = str(stage_output_dir / "checkpoint")
            logger.info(f"  Saving checkpoint to: {checkpoint_dir}")
            self._save_checkpoint(trainer, checkpoint_dir)

            # Record checkpoint for next stage
            self.stage_checkpoints[stage_name] = checkpoint_dir

            duration = time.time() - stage_start_time

            result = StageResult(
                stage_name=stage_name,
                status="success",
                checkpoint_dir=checkpoint_dir,
                metrics=metrics,
                duration_seconds=duration
            )

            logger.info(f"Stage '{stage_name}' completed successfully ({duration:.1f}s)")
            return result

        except Exception as e:
            duration = time.time() - stage_start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Stage '{stage_name}' failed: {error_msg}")

            result = StageResult(
                stage_name=stage_name,
                status="failed",
                error_msg=error_msg,
                duration_seconds=duration
            )
            return result

    def run(self, skip_failed: bool = False, stop_on_failure: bool = True) -> List[StageResult]:
        """Execute all stages in the composition.

        Stages are executed in order, with checkpoints threaded forward. Each stage
        receives the checkpoint from its predecessor (if configured).

        Args:
            skip_failed: If True, continue to next stage even if current fails
            stop_on_failure: If True, stop execution on first failure

        Returns:
            List of StageResult objects, one per stage
        """
        logger.info(f"Starting composition: {self.composition.name}")
        logger.info(f"  Total stages: {len(self.composition.stages)}")

        self.results = []
        prev_checkpoint = None

        for idx, stage in enumerate(self.composition.stages):
            logger.info(f"\n[{idx+1}/{len(self.composition.stages)}] Executing stage: {stage.name}")

            # Execute stage
            result = self.run_stage(stage, prev_checkpoint)
            self.results.append(result)

            # Handle result
            if result.is_failed():
                logger.warning(f"Stage '{stage.name}' failed")
                if stop_on_failure and not skip_failed:
                    logger.error("Stopping composition due to stage failure")
                    break
            elif result.is_success():
                # Thread checkpoint forward if stage succeeded
                prev_checkpoint = result.checkpoint_dir

        return self.results

    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of all stage results.

        Returns:
            Dictionary with execution summary
        """
        total_duration = sum(r.duration_seconds for r in self.results)
        successful = sum(1 for r in self.results if r.is_success())
        failed = sum(1 for r in self.results if r.is_failed())
        skipped = sum(1 for r in self.results if r.is_skipped())

        return {
            'composition_name': self.composition.name,
            'total_stages': len(self.composition.stages),
            'successful_stages': successful,
            'failed_stages': failed,
            'skipped_stages': skipped,
            'total_duration_seconds': total_duration,
            'stage_results': [r.to_dict() for r in self.results]
        }

    # Private helper methods

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Dictionary of configuration data
        """
        import yaml

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Extract 'config' section if present (follows recipe format)
        if isinstance(data, dict) and 'config' in data:
            return data['config']
        return data or {}

    def _merge_configs(
        self,
        base_config: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge override parameters into base config.

        Args:
            base_config: Base configuration dictionary
            overrides: Override parameters to merge

        Returns:
            Merged configuration dictionary
        """
        import copy

        result = copy.deepcopy(base_config)

        def merge_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_dict(target[key], value)
                else:
                    target[key] = value

        merge_dict(result, overrides)
        return result

    def _create_trainer(
        self,
        stage: Stage,
        config: Dict[str, Any],
        init_checkpoint: Optional[str],
        output_dir: str
    ) -> Any:
        """Create a trainer for the stage.

        This is a placeholder implementation. Actual implementation uses BackendFactory
        to create appropriate trainer based on algorithm.

        Args:
            stage: Stage object
            config: Configuration dictionary
            init_checkpoint: Optional checkpoint to initialize from
            output_dir: Output directory for this stage

        Returns:
            Trainer object (actual type depends on backend)
        """
        logger.info(f"[PLACEHOLDER] Creating {stage.algo} trainer")
        # In actual implementation:
        # from ..backend_factory import create_sft_trainer, create_rl_trainer
        # if stage.algo == 'sft':
        #     return create_sft_trainer(...)
        # else:
        #     return create_rl_trainer(...)
        return None

    def _run_trainer(self, trainer: Any) -> Dict[str, Any]:
        """Run training with a trainer object.

        This is a placeholder implementation. Actual implementation calls trainer.train()

        Args:
            trainer: Trainer object

        Returns:
            Dictionary of metrics from training
        """
        logger.info("[PLACEHOLDER] Running trainer")
        # In actual implementation:
        # results = trainer.train()
        # return results
        return {
            'final_loss': 0.5,
            'training_steps': 1000,
            'learning_rate': 2e-4
        }

    def _save_checkpoint(self, trainer: Any, checkpoint_dir: str) -> None:
        """Save checkpoint from trainer.

        This is a placeholder implementation. Actual implementation calls trainer.save()

        Args:
            trainer: Trainer object
            checkpoint_dir: Directory to save checkpoint
        """
        logger.info(f"[PLACEHOLDER] Saving checkpoint to {checkpoint_dir}")
        # In actual implementation:
        # trainer.save_checkpoint(checkpoint_dir)
        # or trainer.model.save_pretrained(checkpoint_dir)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)


class CompositionExecutor:
    """High-level executor for composition files.

    This class provides a simpler interface for end-users to execute compositions.
    """

    @staticmethod
    def execute_composition(
        composition_path: str,
        output_dir: str,
        device: str = "cpu",
        skip_failed: bool = False,
        stop_on_failure: bool = True
    ) -> Dict[str, Any]:
        """Execute a composition from a YAML file.

        Args:
            composition_path: Path to composition YAML file
            output_dir: Output directory for all stages
            device: Device to use
            skip_failed: If True, continue on stage failure
            stop_on_failure: If True, stop on first failure

        Returns:
            Dictionary with execution results
        """
        logger.info(f"Loading composition from: {composition_path}")
        composition = CompositionLoader.load_composition(composition_path)

        logger.info(f"Creating runner with output: {output_dir}")
        runner = CompositionRunner(composition, output_dir, device=device)

        logger.info("Starting execution")
        runner.run(skip_failed=skip_failed, stop_on_failure=stop_on_failure)

        summary = runner.get_results_summary()
        return summary
