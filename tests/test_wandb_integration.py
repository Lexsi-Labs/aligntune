"""
Tests for WandB integration with graceful degradation.

Tests verify that:
1. WandBLogger works correctly when wandb is installed
2. WandBLogger gracefully no-ops when wandb is unavailable
3. All methods silently fail without disrupting training
4. No actual wandb.init calls are made (using mocks)
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional

# Import eagerly, before any test patches `sys.modules` (see below). Several
# tests below use `patch.dict('sys.modules', {'wandb': None})`, which snapshots
# the *entire* sys.modules dict on start() and restores that exact snapshot on
# stop() - wiping out any module (aligntune, torch, datasets, ...) that was
# imported for the first time while the patch was active. Those packages do
# process-global, run-once C-extension registrations (e.g. pyarrow's
# datasets.features.features type registry, torch's docstring registration),
# so a forced re-import later collides with state that was never actually
# reset, raising spurious errors unrelated to wandb. Importing here, before any
# patch.dict on sys.modules starts, ensures they're already present in every
# patch's "original" snapshot and never get evicted.
import torch  # noqa: F401
from aligntune.core.logging import WandBLogger  # noqa: F401


class TestWandBLoggerWithoutWandB(unittest.TestCase):
    """Test WandBLogger graceful degradation when wandb is unavailable."""

    def setUp(self):
        """Set up test fixtures."""
        # Ensure wandb is not available for these tests
        self.wandb_patcher = patch.dict('sys.modules', {'wandb': None})
        self.wandb_patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        self.wandb_patcher.stop()

    def test_logger_init_without_project(self):
        """Test WandBLogger initializes gracefully without project."""
        from aligntune.core.logging import WandBLogger

        # Create logger without project (wandb disabled)
        logger = WandBLogger(project=None)

        # Should not be initialized
        self.assertFalse(logger.is_enabled())
        self.assertFalse(logger.is_initialized)

    def test_logger_init_with_project_no_wandb(self):
        """Test WandBLogger handles missing wandb gracefully."""
        from aligntune.core.logging import WandBLogger

        # Create logger with project but wandb unavailable
        logger = WandBLogger(project="test-project")

        # Should silently disable (WANDB_AVAILABLE is False)
        self.assertFalse(logger.is_enabled())

    def test_log_metrics_no_op(self):
        """Test log_metrics is no-op without wandb."""
        from aligntune.core.logging import WandBLogger

        logger = WandBLogger(project=None)

        # Should not raise error
        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)

    def test_log_artifact_no_op(self):
        """Test log_artifact is no-op without wandb."""
        from aligntune.core.logging import WandBLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")

            logger = WandBLogger(project=None)

            # Should not raise error
            logger.log_artifact(str(test_file), artifact_type="model")

    def test_finalize_no_op(self):
        """Test finalize is no-op without wandb."""
        from aligntune.core.logging import WandBLogger

        logger = WandBLogger(project=None)

        # Should not raise error
        logger.finalize()

    def test_context_manager_no_op(self):
        """Test context manager protocol with no wandb."""
        from aligntune.core.logging import WandBLogger

        with WandBLogger(project=None) as logger:
            logger.log_metrics({"loss": 0.5})

        # Should exit cleanly
        self.assertFalse(logger.is_enabled())


class TestWandBLoggerWithMockWandB(unittest.TestCase):
    """Test WandBLogger with mocked wandb."""

    def setUp(self):
        """Set up mock wandb."""
        self.mock_wandb = MagicMock()
        self.mock_run = MagicMock()
        self.mock_wandb.init.return_value = self.mock_run

        # Patch wandb in the logging module
        self.patcher = patch(
            'aligntune.core.logging.wandb_utils.wandb',
            self.mock_wandb
        )
        self.patcher.start()

        # Also patch WANDB_AVAILABLE
        self.available_patcher = patch(
            'aligntune.core.logging.wandb_utils.WANDB_AVAILABLE',
            True
        )
        self.available_patcher.start()

    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()
        self.available_patcher.stop()

    def test_logger_init_with_wandb(self):
        """Test WandBLogger initializes with wandb."""
        from aligntune.core.logging import WandBLogger

        logger = WandBLogger(
            project="test-project",
            entity="test-entity",
            config={"lr": 1e-5},
            tags=["test"],
        )

        # Should be initialized
        self.assertTrue(logger.is_enabled())
        self.assertTrue(logger.is_initialized)

        # wandb.init should have been called
        self.mock_wandb.init.assert_called_once()

    def test_log_metrics_with_wandb(self):
        """Test log_metrics sends to wandb."""
        from aligntune.core.logging import WandBLogger

        logger = WandBLogger(project="test-project")

        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)

        # Should call wandb run.log
        self.mock_run.log.assert_called_once()

    def test_log_artifact_with_wandb(self):
        """Test log_artifact creates wandb artifact."""
        from aligntune.core.logging import WandBLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")

            logger = WandBLogger(project="test-project")
            logger.log_artifact(str(test_file), artifact_type="model")

            # Should call wandb artifact methods
            self.mock_wandb.Artifact.assert_called_once()
            self.mock_run.log_artifact.assert_called_once()

    def test_log_model_with_wandb(self):
        """Test log_model creates model artifact."""
        from aligntune.core.logging import WandBLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test directory
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")

            logger = WandBLogger(project="test-project")
            logger.log_model(str(model_dir), name="my-model", aliases=["v1"])

            # Should create artifact
            self.mock_wandb.Artifact.assert_called()

    def test_update_config_with_wandb(self):
        """Test update_config updates wandb config."""
        from aligntune.core.logging import WandBLogger

        logger = WandBLogger(project="test-project")
        logger.update_config({"num_params": 7_000_000_000})

        # Config should be updated
        self.assertEqual(self.mock_run.config.__setitem__.call_count, 1)

    def test_watch_model_with_wandb(self):
        """Test watch_model sets up wandb watching."""
        from aligntune.core.logging import WandBLogger
        import torch

        model = torch.nn.Linear(10, 10)

        logger = WandBLogger(project="test-project")
        logger.watch_model(model)

        # Should call wandb watch
        self.mock_run.watch.assert_called_once()

    def test_finalize_with_wandb(self):
        """Test finalize closes wandb run."""
        from aligntune.core.logging import WandBLogger

        logger = WandBLogger(project="test-project")
        logger.finalize()

        # Should call wandb finish
        self.mock_run.finish.assert_called_once()
        self.assertFalse(logger.is_enabled())


class TestWandBLoggerErrorHandling(unittest.TestCase):
    """Test error handling in WandBLogger."""

    def setUp(self):
        """Set up mock wandb with error scenarios."""
        self.mock_wandb = MagicMock()
        self.mock_run = MagicMock()
        self.mock_wandb.init.return_value = self.mock_run

        self.patcher = patch(
            'aligntune.core.logging.wandb_utils.wandb',
            self.mock_wandb
        )
        self.patcher.start()

        self.available_patcher = patch(
            'aligntune.core.logging.wandb_utils.WANDB_AVAILABLE',
            True
        )
        self.available_patcher.start()

    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()
        self.available_patcher.stop()

    def test_log_metrics_error_handling(self):
        """Test log_metrics handles wandb errors gracefully."""
        from aligntune.core.logging import WandBLogger

        # Make log raise an exception
        self.mock_run.log.side_effect = Exception("wandb error")

        logger = WandBLogger(project="test-project")

        # Should not raise, just log warning
        logger.log_metrics({"loss": 0.5})

    def test_log_artifact_nonexistent_path(self):
        """Test log_artifact handles missing files gracefully."""
        from aligntune.core.logging import WandBLogger

        logger = WandBLogger(project="test-project")

        # Should handle missing path gracefully
        logger.log_artifact("/nonexistent/path", artifact_type="model")

        # Should not create artifact
        self.mock_wandb.Artifact.assert_not_called()

    def test_init_failure_graceful_degradation(self):
        """Test wandb.init failure results in no-op logger."""
        from aligntune.core.logging import WandBLogger

        # Make init raise an exception
        self.mock_wandb.init.side_effect = Exception("Network error")

        logger = WandBLogger(project="test-project")

        # Should be disabled after failure
        self.assertFalse(logger.is_enabled())

        # Subsequent calls should be no-ops
        logger.log_metrics({"loss": 0.5})
        self.mock_run.log.assert_not_called()


class TestWandBLoggerIntegration(unittest.TestCase):
    """Integration tests for WandBLogger with other components."""

    def test_config_integration(self):
        """Test wandb config fields in TrainingConfig."""
        from aligntune.core.rl.config import LoggingConfig

        config = LoggingConfig(
            wandb_project="test-project",
            wandb_entity="test-entity",
            wandb_tags=["v1", "test"],
            wandb_notes="Test run",
        )

        self.assertEqual(config.wandb_project, "test-project")
        self.assertEqual(config.wandb_entity, "test-entity")
        self.assertEqual(config.wandb_tags, ["v1", "test"])
        self.assertEqual(config.wandb_notes, "Test run")

    def test_config_defaults(self):
        """Test wandb config defaults."""
        from aligntune.core.rl.config import LoggingConfig

        config = LoggingConfig()

        self.assertIsNone(config.wandb_project)
        self.assertIsNone(config.wandb_entity)
        self.assertEqual(config.wandb_tags, [])
        self.assertIsNone(config.wandb_notes)

    def test_logging_config_validation(self):
        """Test LoggingConfig validation."""
        from aligntune.core.rl.config import LoggingConfig

        # Should validate without errors
        config = LoggingConfig(
            wandb_project="test",
            loggers=["tensorboard"],
        )

        self.assertIsNotNone(config)


if __name__ == '__main__':
    unittest.main()
