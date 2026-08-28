"""
Composition stages for multi-stage training pipelines.

This module provides dataclasses and utilities for defining and loading
multi-stage training compositions that support SFT → MoA → ES → DPO → audit workflows.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml
from datetime import datetime


@dataclass
class Stage:
    """Represents a single training stage in a composition pipeline.

    Attributes:
        name: Unique identifier for this stage (e.g., "sft", "moa", "dpo")
        algo: Algorithm to use (e.g., "sft", "dpo", "ppo", "grpo")
        config_path: Path to YAML config file for this stage
        init_from: Optional previous stage name to initialize from its checkpoint
        target_params: Optional dict of parameter overrides for this stage
        description: Optional human-readable description of this stage
    """
    name: str
    algo: str
    config_path: str
    init_from: Optional[str] = None
    target_params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert stage to dictionary representation."""
        data = {
            'name': self.name,
            'algo': self.algo,
            'config_path': self.config_path,
            'init_from': self.init_from,
            'target_params': self.target_params
        }
        if self.description:
            data['description'] = self.description
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stage':
        """Create Stage from dictionary representation."""
        return cls(**data)


@dataclass
class Composition:
    """A multi-stage training composition pipeline.

    Attributes:
        name: Name of this composition
        description: Human-readable description
        stages: List of Stage objects in order
        metadata: Dict of additional metadata (tags, version, etc.)
    """
    name: str
    stages: List[Stage]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert composition to dictionary representation."""
        return {
            'name': self.name,
            'description': self.description,
            'stages': [s.to_dict() for s in self.stages],
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Composition':
        """Create Composition from dictionary representation."""
        stages = [Stage.from_dict(s) for s in data.get('stages', [])]
        return cls(
            name=data['name'],
            stages=stages,
            description=data.get('description', ''),
            metadata=data.get('metadata', {})
        )

    def get_stage(self, stage_name: str) -> Optional[Stage]:
        """Get a stage by name."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None

    def get_stage_index(self, stage_name: str) -> int:
        """Get the index of a stage by name. Returns -1 if not found."""
        for idx, stage in enumerate(self.stages):
            if stage.name == stage_name:
                return idx
        return -1

    def get_previous_stage(self, stage_name: str) -> Optional[Stage]:
        """Get the stage before a given stage."""
        idx = self.get_stage_index(stage_name)
        if idx > 0:
            return self.stages[idx - 1]
        return None


@dataclass
class StageResult:
    """Result from executing a single training stage.

    Attributes:
        stage_name: Name of the stage that was executed
        status: One of "success", "failed", "skipped"
        checkpoint_dir: Path to the checkpoint directory created
        metrics: Dict of metrics from training (loss, accuracy, etc.)
        error_msg: Error message if status is "failed"
        duration_seconds: Time taken to run this stage
    """
    stage_name: str
    status: str  # "success", "failed", "skipped"
    checkpoint_dir: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_msg: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            'stage_name': self.stage_name,
            'status': self.status,
            'checkpoint_dir': self.checkpoint_dir,
            'metrics': self.metrics,
            'error_msg': self.error_msg,
            'duration_seconds': self.duration_seconds
        }

    def is_success(self) -> bool:
        """Check if this stage succeeded."""
        return self.status == "success"

    def is_failed(self) -> bool:
        """Check if this stage failed."""
        return self.status == "failed"

    def is_skipped(self) -> bool:
        """Check if this stage was skipped."""
        return self.status == "skipped"


class CompositionLoader:
    """Load composition specifications from YAML files."""

    @staticmethod
    def load_composition(yaml_path: Union[str, Path]) -> Composition:
        """Load a composition from a YAML file.

        Args:
            yaml_path: Path to the composition YAML file

        Returns:
            Composition object

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML structure is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Composition file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Composition file must contain a dictionary, got {type(data)}")

        # Validate required fields
        if 'name' not in data:
            raise ValueError("Composition must have a 'name' field")
        if 'stages' not in data or not isinstance(data['stages'], list):
            raise ValueError("Composition must have a 'stages' field (list)")

        # Parse stages
        stages = []
        for idx, stage_data in enumerate(data['stages']):
            if not isinstance(stage_data, dict):
                raise ValueError(f"Stage {idx} must be a dictionary")

            required = ['name', 'algo', 'config_path']
            for req in required:
                if req not in stage_data:
                    raise ValueError(f"Stage {idx} missing required field: {req}")

            stages.append(Stage.from_dict(stage_data))

        return Composition(
            name=data['name'],
            stages=stages,
            description=data.get('description', ''),
            metadata=data.get('metadata', {})
        )

    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> Composition:
        """Load a composition from a dictionary.

        Args:
            data: Dictionary containing composition spec

        Returns:
            Composition object

        Raises:
            ValueError: If structure is invalid
        """
        if not isinstance(data, dict):
            raise ValueError(f"Composition data must be a dictionary, got {type(data)}")

        # Validate required fields
        if 'name' not in data:
            raise ValueError("Composition must have a 'name' field")
        if 'stages' not in data or not isinstance(data['stages'], list):
            raise ValueError("Composition must have a 'stages' field (list)")

        # Parse stages
        stages = []
        for idx, stage_data in enumerate(data['stages']):
            if not isinstance(stage_data, dict):
                raise ValueError(f"Stage {idx} must be a dictionary")

            required = ['name', 'algo', 'config_path']
            for req in required:
                if req not in stage_data:
                    raise ValueError(f"Stage {idx} missing required field: {req}")

            stages.append(Stage.from_dict(stage_data))

        return Composition(
            name=data['name'],
            stages=stages,
            description=data.get('description', ''),
            metadata=data.get('metadata', {})
        )

    @staticmethod
    def save_composition(composition: Composition, yaml_path: Union[str, Path]) -> None:
        """Save a composition to a YAML file.

        Args:
            composition: Composition object to save
            yaml_path: Path where to save the YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data = composition.to_dict()
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
