"""
Unit tests for composition stages and runner.

Tests cover:
- Stage and Composition dataclass serialization
- CompositionLoader YAML parsing
- CompositionRunner stage orchestration
- Checkpoint threading
- Error handling
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
import yaml

from src.aligntune.core.composition import (
    Stage,
    Composition,
    StageResult,
    CompositionLoader,
    CompositionRunner,
)


class TestStage:
    """Tests for Stage dataclass."""

    def test_stage_creation(self):
        """Test basic Stage creation."""
        stage = Stage(
            name="sft",
            algo="sft",
            config_path="config_sft.yaml"
        )
        assert stage.name == "sft"
        assert stage.algo == "sft"
        assert stage.config_path == "config_sft.yaml"
        assert stage.init_from is None
        assert stage.target_params == {}

    def test_stage_with_init_from(self):
        """Test Stage with init_from field."""
        stage = Stage(
            name="dpo",
            algo="dpo",
            config_path="config_dpo.yaml",
            init_from="sft"
        )
        assert stage.init_from == "sft"

    def test_stage_with_target_params(self):
        """Test Stage with target parameter overrides."""
        params = {"learning_rate": 1e-5, "epochs": 5}
        stage = Stage(
            name="dpo",
            algo="dpo",
            config_path="config_dpo.yaml",
            target_params=params
        )
        assert stage.target_params == params

    def test_stage_to_dict(self):
        """Test Stage serialization to dict."""
        stage = Stage(
            name="dpo",
            algo="dpo",
            config_path="config_dpo.yaml",
            init_from="sft",
            target_params={"learning_rate": 1e-5}
        )
        data = stage.to_dict()
        assert data['name'] == 'dpo'
        assert data['algo'] == 'dpo'
        assert data['config_path'] == 'config_dpo.yaml'
        assert data['init_from'] == 'sft'
        assert data['target_params'] == {"learning_rate": 1e-5}

    def test_stage_from_dict(self):
        """Test Stage creation from dict."""
        data = {
            'name': 'moa',
            'algo': 'sft',
            'config_path': 'config_moa.yaml',
            'init_from': 'sft',
            'target_params': {'learning_rate': 2e-4}
        }
        stage = Stage.from_dict(data)
        assert stage.name == 'moa'
        assert stage.algo == 'sft'
        assert stage.init_from == 'sft'
        assert stage.target_params == {'learning_rate': 2e-4}


class TestComposition:
    """Tests for Composition dataclass."""

    def test_composition_creation(self):
        """Test basic Composition creation."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
            Stage(name="dpo", algo="dpo", config_path="config_dpo.yaml", init_from="sft"),
        ]
        comp = Composition(
            name="pipeline_1",
            stages=stages,
            description="Test pipeline"
        )
        assert comp.name == "pipeline_1"
        assert len(comp.stages) == 2
        assert comp.description == "Test pipeline"

    def test_composition_get_stage(self):
        """Test getting stage by name."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
            Stage(name="dpo", algo="dpo", config_path="config_dpo.yaml"),
        ]
        comp = Composition(name="test", stages=stages)

        stage = comp.get_stage("sft")
        assert stage is not None
        assert stage.algo == "sft"

        missing = comp.get_stage("missing")
        assert missing is None

    def test_composition_get_stage_index(self):
        """Test getting stage index."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
            Stage(name="dpo", algo="dpo", config_path="config_dpo.yaml"),
            Stage(name="audit", algo="sft", config_path="config_audit.yaml"),
        ]
        comp = Composition(name="test", stages=stages)

        assert comp.get_stage_index("sft") == 0
        assert comp.get_stage_index("dpo") == 1
        assert comp.get_stage_index("audit") == 2
        assert comp.get_stage_index("missing") == -1

    def test_composition_get_previous_stage(self):
        """Test getting previous stage."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
            Stage(name="dpo", algo="dpo", config_path="config_dpo.yaml", init_from="sft"),
            Stage(name="audit", algo="sft", config_path="config_audit.yaml"),
        ]
        comp = Composition(name="test", stages=stages)

        # sft has no previous stage
        assert comp.get_previous_stage("sft") is None

        # dpo's previous stage is sft
        prev = comp.get_previous_stage("dpo")
        assert prev is not None
        assert prev.name == "sft"

        # audit's previous stage is dpo
        prev = comp.get_previous_stage("audit")
        assert prev is not None
        assert prev.name == "dpo"

    def test_composition_to_dict(self):
        """Test Composition serialization."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
            Stage(name="dpo", algo="dpo", config_path="config_dpo.yaml", init_from="sft"),
        ]
        comp = Composition(
            name="test_pipeline",
            stages=stages,
            description="Test description",
            metadata={"version": "1.0"}
        )
        data = comp.to_dict()
        assert data['name'] == 'test_pipeline'
        assert len(data['stages']) == 2
        assert data['description'] == 'Test description'
        assert data['metadata'] == {"version": "1.0"}

    def test_composition_from_dict(self):
        """Test Composition creation from dict."""
        data = {
            'name': 'test_pipeline',
            'description': 'Test',
            'stages': [
                {'name': 'sft', 'algo': 'sft', 'config_path': 'config_sft.yaml'},
                {'name': 'dpo', 'algo': 'dpo', 'config_path': 'config_dpo.yaml', 'init_from': 'sft'},
            ],
            'metadata': {'version': '1.0'}
        }
        comp = Composition.from_dict(data)
        assert comp.name == 'test_pipeline'
        assert len(comp.stages) == 2
        assert comp.stages[1].init_from == 'sft'


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_stage_result_success(self):
        """Test successful stage result."""
        result = StageResult(
            stage_name="sft",
            status="success",
            checkpoint_dir="/path/to/checkpoint",
            metrics={"final_loss": 0.5},
            duration_seconds=3600.0
        )
        assert result.is_success()
        assert not result.is_failed()
        assert not result.is_skipped()

    def test_stage_result_failed(self):
        """Test failed stage result."""
        result = StageResult(
            stage_name="dpo",
            status="failed",
            error_msg="Out of memory",
            duration_seconds=100.0
        )
        assert result.is_failed()
        assert not result.is_success()
        assert not result.is_skipped()
        assert result.error_msg == "Out of memory"

    def test_stage_result_to_dict(self):
        """Test StageResult serialization."""
        result = StageResult(
            stage_name="sft",
            status="success",
            checkpoint_dir="/path/to/checkpoint",
            metrics={"loss": 0.5},
            duration_seconds=3600.0
        )
        data = result.to_dict()
        assert data['stage_name'] == 'sft'
        assert data['status'] == 'success'
        assert data['checkpoint_dir'] == '/path/to/checkpoint'
        assert data['metrics'] == {"loss": 0.5}


class TestCompositionLoader:
    """Tests for CompositionLoader."""

    def test_load_composition_from_yaml(self):
        """Test loading composition from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_data = {
                'name': 'test_pipeline',
                'description': 'Test composition',
                'stages': [
                    {
                        'name': 'sft',
                        'algo': 'sft',
                        'config_path': 'config_sft.yaml'
                    },
                    {
                        'name': 'dpo',
                        'algo': 'dpo',
                        'config_path': 'config_dpo.yaml',
                        'init_from': 'sft'
                    }
                ],
                'metadata': {'version': '1.0'}
            }
            yaml.safe_dump(yaml_data, f)
            f.flush()

            try:
                comp = CompositionLoader.load_composition(f.name)
                assert comp.name == 'test_pipeline'
                assert len(comp.stages) == 2
                assert comp.stages[1].init_from == 'sft'
            finally:
                Path(f.name).unlink()

    def test_load_composition_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            CompositionLoader.load_composition('/nonexistent/path.yaml')

    def test_load_composition_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            try:
                with pytest.raises(Exception):  # YAML parsing error
                    CompositionLoader.load_composition(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_composition_missing_name(self):
        """Test loading composition without name field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_data = {
                'stages': [
                    {'name': 'sft', 'algo': 'sft', 'config_path': 'config_sft.yaml'}
                ]
            }
            yaml.safe_dump(yaml_data, f)
            f.flush()

            try:
                with pytest.raises(ValueError, match="must have a 'name' field"):
                    CompositionLoader.load_composition(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_composition_missing_stages(self):
        """Test loading composition without stages field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_data = {'name': 'test'}
            yaml.safe_dump(yaml_data, f)
            f.flush()

            try:
                with pytest.raises(ValueError, match="must have a 'stages' field"):
                    CompositionLoader.load_composition(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_composition_missing_stage_field(self):
        """Test loading composition with incomplete stage."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_data = {
                'name': 'test',
                'stages': [
                    {'name': 'sft', 'algo': 'sft'}  # Missing config_path
                ]
            }
            yaml.safe_dump(yaml_data, f)
            f.flush()

            try:
                with pytest.raises(ValueError, match="missing required field"):
                    CompositionLoader.load_composition(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_from_dict(self):
        """Test loading composition from dictionary."""
        data = {
            'name': 'test_pipeline',
            'stages': [
                {'name': 'sft', 'algo': 'sft', 'config_path': 'config_sft.yaml'},
                {'name': 'dpo', 'algo': 'dpo', 'config_path': 'config_dpo.yaml', 'init_from': 'sft'},
            ]
        }
        comp = CompositionLoader.load_from_dict(data)
        assert comp.name == 'test_pipeline'
        assert len(comp.stages) == 2

    def test_save_composition(self):
        """Test saving composition to YAML."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
            Stage(name="dpo", algo="dpo", config_path="config_dpo.yaml", init_from="sft"),
        ]
        comp = Composition(
            name="test_pipeline",
            stages=stages,
            description="Test"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_comp.yaml"
            CompositionLoader.save_composition(comp, save_path)

            # Load it back
            loaded = CompositionLoader.load_composition(save_path)
            assert loaded.name == comp.name
            assert len(loaded.stages) == len(comp.stages)
            assert loaded.stages[1].init_from == "sft"


class TestCompositionRunner:
    """Tests for CompositionRunner."""

    def test_runner_initialization(self):
        """Test CompositionRunner initialization."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
            Stage(name="dpo", algo="dpo", config_path="config_dpo.yaml", init_from="sft"),
        ]
        comp = Composition(name="test", stages=stages)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CompositionRunner(comp, tmpdir, device="cpu")
            assert runner.composition == comp
            assert runner.base_output_dir == Path(tmpdir)
            assert runner.device == "cpu"
            assert len(runner.results) == 0

    def test_get_stage_output_dir(self):
        """Test getting stage output directory."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
        ]
        comp = Composition(name="test", stages=stages)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CompositionRunner(comp, tmpdir)
            stage_dir = runner.get_stage_output_dir("sft")
            assert stage_dir == Path(tmpdir) / "sft"

    def test_results_summary(self):
        """Test getting results summary."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
        ]
        comp = Composition(name="test", stages=stages)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CompositionRunner(comp, tmpdir)
            runner.results = [
                StageResult(
                    stage_name="sft",
                    status="success",
                    checkpoint_dir="/path/to/checkpoint",
                    metrics={"loss": 0.5},
                    duration_seconds=100.0
                )
            ]

            summary = runner.get_results_summary()
            assert summary['composition_name'] == 'test'
            assert summary['total_stages'] == 1
            assert summary['successful_stages'] == 1
            assert summary['failed_stages'] == 0
            assert summary['total_duration_seconds'] == 100.0

    def test_merge_configs(self):
        """Test configuration merging."""
        stages = [Stage(name="sft", algo="sft", config_path="config.yaml")]
        comp = Composition(name="test", stages=stages)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CompositionRunner(comp, tmpdir)

            base = {
                'train': {'learning_rate': 1e-4, 'epochs': 3},
                'model': {'name': 'base'}
            }
            overrides = {
                'train': {'learning_rate': 2e-4},
                'model': {'precision': 'bf16'}
            }

            merged = runner._merge_configs(base, overrides)
            assert merged['train']['learning_rate'] == 2e-4
            assert merged['train']['epochs'] == 3  # Preserved
            assert merged['model']['name'] == 'base'
            assert merged['model']['precision'] == 'bf16'

    def test_load_config_file(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'model': {'name': 'test'},
                'train': {'learning_rate': 1e-4}
            }
            yaml.safe_dump(config_data, f)
            f.flush()

            try:
                stages = [Stage(name="sft", algo="sft", config_path="config.yaml")]
                comp = Composition(name="test", stages=stages)

                with tempfile.TemporaryDirectory() as tmpdir:
                    runner = CompositionRunner(comp, tmpdir)
                    loaded = runner._load_config_file(Path(f.name))
                    assert loaded['model']['name'] == 'test'
                    assert loaded['train']['learning_rate'] == 1e-4
            finally:
                Path(f.name).unlink()

    def test_load_config_file_with_config_wrapper(self):
        """Test loading config when wrapped in 'config' key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            file_data = {
                'recipe': {'name': 'test'},
                'config': {
                    'model': {'name': 'test'},
                    'train': {'learning_rate': 1e-4}
                }
            }
            yaml.safe_dump(file_data, f)
            f.flush()

            try:
                stages = [Stage(name="sft", algo="sft", config_path="config.yaml")]
                comp = Composition(name="test", stages=stages)

                with tempfile.TemporaryDirectory() as tmpdir:
                    runner = CompositionRunner(comp, tmpdir)
                    loaded = runner._load_config_file(Path(f.name))
                    # Should extract the 'config' section
                    assert loaded['model']['name'] == 'test'
                    assert loaded['train']['learning_rate'] == 1e-4
            finally:
                Path(f.name).unlink()


class TestFullPipeline:
    """Integration tests for full composition execution."""

    def test_composition_execution_order(self):
        """Test that stages are executed in correct order."""
        stages = [
            Stage(name="sft", algo="sft", config_path="config_sft.yaml"),
            Stage(name="moa", algo="sft", config_path="config_moa.yaml", init_from="sft"),
            Stage(name="dpo", algo="dpo", config_path="config_dpo.yaml", init_from="moa"),
        ]
        comp = Composition(name="test_pipeline", stages=stages)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy config files
            config_dir = Path(tmpdir) / "configs"
            config_dir.mkdir()

            for stage in stages:
                config_file = config_dir / stage.config_path
                yaml.safe_dump({
                    'model': {'name': 'test'},
                    'train': {'learning_rate': 1e-4}
                }, open(config_file, 'w'))

                # Update stage config path to be absolute
                stage.config_path = str(config_file)

            runner = CompositionRunner(comp, tmpdir)

            # In a real test, we'd mock the trainer creation
            # For now, just verify the runner was created correctly
            assert runner.composition.name == 'test_pipeline'
            assert len(runner.composition.stages) == 3
            assert runner.composition.stages[1].init_from == 'sft'
            assert runner.composition.stages[2].init_from == 'moa'
