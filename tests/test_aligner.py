"""
Tests for AlignTune Aligner (Interactive Training Inspector).

Tests cover:
- AlignerSession lifecycle (start, pause, resume, stop)
- State management and metrics tracking
- Hyperparameter updates (set)
- Checkpoint rollback
- Example tracking and worst_examples
- Model sampling
- Integration with trainer callbacks
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch, call
from dataclasses import dataclass

from aligntune.core.aligner import (
    AlignerSession,
    AlignerState,
    AlignerCallback,
    create_dashboard,
)
from aligntune.core.callbacks import TrainerControl


@dataclass
class MockConfig:
    """Mock training configuration."""
    train: Mock = None
    algo: str = "DPO"


class MockTrainer:
    """Mock trainer for testing."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = MockConfig()
        self.callbacks = []
        self.trained = False

    def train(self):
        """Mock training."""
        self.trained = True


class TestAlignerState:
    """Tests for AlignerState dataclass."""

    def test_state_initialization(self):
        """Test AlignerState initializes with default values."""
        state = AlignerState()
        assert state.step == 0
        assert state.loss == 0.0
        assert state.reward == 0.0
        assert state.kl_divergence == 0.0
        assert state.learning_rate == 0.0
        assert state.batch_size == 0
        assert state.elapsed_time == 0.0
        assert state.recent_examples == []

    def test_state_to_dict(self):
        """Test state conversion to dictionary."""
        state = AlignerState(step=10, loss=0.5, reward=0.8)
        state_dict = state.to_dict()

        assert isinstance(state_dict, dict)
        assert state_dict["step"] == 10
        assert state_dict["loss"] == 0.5
        assert state_dict["reward"] == 0.8


class TestAlignerSessionInitialization:
    """Tests for AlignerSession initialization."""

    def test_session_initialization(self):
        """Test AlignerSession initializes correctly."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        assert session.trainer is trainer
        assert isinstance(session.state, AlignerState)
        assert session.state.step == 0
        assert not session.is_training()
        assert not session.is_paused()

    def test_session_wraps_trainer(self):
        """Test session correctly wraps trainer."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        assert session.get_trainer() is trainer


class TestAlignerSessionLifecycle:
    """Tests for AlignerSession lifecycle (start, pause, resume, stop)."""

    def test_start_creates_thread(self):
        """Test start() creates training thread."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        session.start()
        assert session._training_thread is not None
        # Thread may finish quickly, just verify it was created
        time.sleep(0.2)

        # Cleanup
        session.stop()

    def test_pause_resumes_workflow(self):
        """Test pause and resume."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        # Initially not paused
        assert not session.is_paused()

        # Pause
        session.pause()
        assert session.is_paused()

        # Resume
        session.resume()
        assert not session.is_paused()

    def test_stop_terminates_training(self):
        """Test stop() terminates training."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        session.start()
        time.sleep(0.1)
        session.stop()

        time.sleep(0.2)
        assert not session.is_training()

    def test_multiple_start_calls(self):
        """Test calling start multiple times."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        session.start()
        first_thread = session._training_thread

        # Second start while thread is (may be) running
        # Just verify that start can be called multiple times without crashing
        session.start()

        session.stop()


class TestAlignerSessionStateManagement:
    """Tests for state management."""

    def test_peek_returns_dict(self):
        """Test peek() returns state as dictionary."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        state = session.peek()
        assert isinstance(state, dict)
        assert "step" in state
        assert "loss" in state
        assert "reward" in state
        assert "kl_divergence" in state
        assert "learning_rate" in state

    def test_peek_reflects_current_values(self):
        """Test peek() reflects current state values."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        # Update state
        session.state.step = 42
        session.state.loss = 0.123
        session.state.reward = 0.456

        state = session.peek()
        assert state["step"] == 42
        assert state["loss"] == 0.123
        assert state["reward"] == 0.456

    def test_update_state_from_metrics(self):
        """Test _update_state processes metrics correctly."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        metrics = {
            "loss": 0.5,
            "reward": 0.8,
            "kl": 0.1,
            "learning_rate": 1e-4,
            "batch_size": 16,
        }

        session._update_state(metrics)

        assert session.state.loss == 0.5
        assert session.state.reward == 0.8
        assert session.state.kl_divergence == 0.1
        assert session.state.learning_rate == 1e-4
        assert session.state.batch_size == 16

    def test_metrics_history_tracking(self):
        """Test that metrics history is tracked."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        metrics1 = {"loss": 0.5, "reward": 0.8}
        metrics2 = {"loss": 0.4, "reward": 0.9}

        session._update_state(metrics1)
        session._update_state(metrics2)

        history = session.history()
        assert "loss" in history
        assert "reward" in history
        assert history["loss"] == [0.5, 0.4]
        assert history["reward"] == [0.8, 0.9]


class TestAlignerSessionHyperparameterUpdate:
    """Tests for hyperparameter updates."""

    def test_set_queues_command(self):
        """Test set() queues hyperparameter command."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        session.set(learning_rate=1e-5, beta=0.1)

        # Check command is in queue
        cmd = session.command_queue.get_nowait()
        assert cmd.action == "set_hyperparams"
        assert cmd.kwargs["learning_rate"] == 1e-5
        assert cmd.kwargs["beta"] == 0.1

    def test_set_multiple_hyperparams(self):
        """Test set() with multiple hyperparameters."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        hyperparams = {
            "learning_rate": 5e-5,
            "beta": 0.2,
            "temperature": 0.8,
        }
        session.set(**hyperparams)

        cmd = session.command_queue.get_nowait()
        assert cmd.kwargs == hyperparams


class TestAlignerSessionRollback:
    """Tests for checkpoint rollback."""

    def test_rollback_queues_command(self):
        """Test rollback() queues rollback command."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        session.rollback(100)

        cmd = session.command_queue.get_nowait()
        assert cmd.action == "rollback"
        assert cmd.kwargs["step"] == 100

    def test_rollback_to_different_steps(self):
        """Test rollback to various steps."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        for step in [10, 50, 100, 500]:
            session.rollback(step)

        for step in [10, 50, 100, 500]:
            cmd = session.command_queue.get_nowait()
            assert cmd.kwargs["step"] == step


class TestAlignerSessionExamples:
    """Tests for example tracking."""

    def test_add_example(self):
        """Test _add_example stores examples."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        example = {
            "prompt": "Hello",
            "chosen": "Hi",
            "rejected": "No",
            "reward_delta": 0.5,
        }

        session._add_example(example)
        assert len(session._examples_buffer) == 1

    def test_worst_examples_empty(self):
        """Test worst_examples() with no examples."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        worst = session.worst_examples(5)
        assert worst == []

    def test_worst_examples_sorting(self):
        """Test worst_examples() sorts by reward_delta."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        examples = [
            {"prompt": "p1", "chosen": "c1", "rejected": "r1", "reward_delta": 0.5},
            {"prompt": "p2", "chosen": "c2", "rejected": "r2", "reward_delta": 0.1},
            {"prompt": "p3", "chosen": "c3", "rejected": "r3", "reward_delta": 0.9},
        ]

        for ex in examples:
            session._add_example(ex)

        worst = session.worst_examples(3)
        # Should be sorted by reward_delta (ascending)
        assert worst[0][3] == 0.1
        assert worst[1][3] == 0.5
        assert worst[2][3] == 0.9

    def test_worst_examples_limit(self):
        """Test worst_examples() respects limit."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        for i in range(10):
            example = {
                "prompt": f"p{i}",
                "chosen": f"c{i}",
                "rejected": f"r{i}",
                "reward_delta": float(i),
            }
            session._add_example(example)

        worst = session.worst_examples(5)
        assert len(worst) == 5


class TestAlignerCallbackIntegration:
    """Tests for AlignerCallback integration."""

    def test_callback_initialization(self):
        """Test AlignerCallback initializes correctly."""
        session = AlignerSession(MockTrainer())
        callback = AlignerCallback(session)

        assert callback.aligner_session is session

    def test_callback_on_log_updates_state(self):
        """Test on_log hook updates session state."""
        session = AlignerSession(MockTrainer())
        callback = AlignerCallback(session)

        logs = {
            "loss": 0.5,
            "reward": 0.8,
            "kl": 0.1,
        }

        state = Mock()
        state.start_time = time.time()
        control = TrainerControl()

        callback.on_log(None, state, control, logs=logs)

        assert session.state.loss == 0.5
        assert session.state.reward == 0.8
        assert session.state.kl_divergence == 0.1

    def test_callback_on_step_end_updates_step(self):
        """Test on_step_end updates step count."""
        session = AlignerSession(MockTrainer())
        callback = AlignerCallback(session)

        state = Mock()
        state.step = 42
        control = TrainerControl()

        callback.on_step_end(None, state, control)

        assert session.state.step == 42

    def test_callback_without_session(self):
        """Test callback works without session (graceful fallback)."""
        callback = AlignerCallback(None)

        state = Mock()
        control = TrainerControl()
        logs = {"loss": 0.5}

        # Should not raise
        callback.on_log(None, state, control, logs=logs)
        callback.on_step_end(None, state, control)


class TestAlignerDashboard:
    """Tests for AlignerDashboard."""

    def test_dashboard_creation(self):
        """Test dashboard creation."""
        session = AlignerSession(MockTrainer())
        dashboard = create_dashboard(session)

        if dashboard is not None:
            assert dashboard.aligner_session is session

    def test_dashboard_start_stop(self):
        """Test dashboard start and stop."""
        session = AlignerSession(MockTrainer())
        dashboard = create_dashboard(session)

        if dashboard is not None:
            dashboard.start()
            assert dashboard._running

            dashboard.stop()
            assert not dashboard._running


class TestAlignerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_state_updates(self):
        """Test concurrent state updates are thread-safe."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        def update_state():
            for i in range(100):
                session._update_state({
                    "loss": 0.5 - i * 0.001,
                    "reward": 0.8 + i * 0.001,
                })
                time.sleep(0.001)

        threads = [threading.Thread(target=update_state) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash
        history = session.history()
        assert len(history["loss"]) > 0
        assert len(history["reward"]) > 0

    def test_concurrent_peek_and_update(self):
        """Test concurrent peek() and state update."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        def updater():
            for i in range(100):
                session._update_state({"loss": float(i)})
                time.sleep(0.001)

        def reader():
            for _ in range(100):
                session.peek()
                time.sleep(0.001)

        t1 = threading.Thread(target=updater)
        t2 = threading.Thread(target=reader)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # Should complete without deadlock


class TestAlignerIntegration:
    """Integration tests."""

    def test_full_session_lifecycle(self):
        """Test complete session lifecycle."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        # Start
        session.start()
        time.sleep(0.1)  # Allow thread to start

        # Update state
        session._update_state({"loss": 0.5, "reward": 0.8})

        # Add example
        session._add_example({
            "prompt": "test",
            "chosen": "good",
            "rejected": "bad",
            "reward_delta": 0.5,
        })

        # Set hyperparams
        session.set(learning_rate=1e-5)

        # Peek
        state = session.peek()
        assert state["loss"] == 0.5
        assert state["reward"] == 0.8

        # Get worst examples
        worst = session.worst_examples(1)
        assert len(worst) == 1

        # Get history
        history = session.history()
        assert "loss" in history

        # Stop
        session.stop()
        time.sleep(0.1)  # Allow cleanup
        assert not session.is_training()

    def test_pause_resume_cycle(self):
        """Test pause/resume cycle."""
        trainer = MockTrainer()
        session = AlignerSession(trainer)

        assert not session.is_paused()

        session.pause()
        assert session.is_paused()

        session.resume()
        assert not session.is_paused()

        session.pause()
        assert session.is_paused()

        session.resume()
        assert not session.is_paused()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
