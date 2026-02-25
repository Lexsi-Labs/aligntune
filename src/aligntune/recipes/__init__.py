"""
Recipe system for AlignTune - pre-configured training recipes.

This module provides a registry of ready-to-run training configurations
for popular models and tasks, leveraging authenticated HF access.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..core.sft.config import SFTConfig
from ..core.rl.config import UnifiedConfig

class RecipeType(Enum):
    """Types of training recipes."""
    SFT = "sft"
    DPO = "dpo"
    PPO = "ppo"
    GRPO = "grpo"
    GSPO = "gspo"

class ModelFamily(Enum):
    """Model families supported by recipes."""
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    PHI = "phi"
    OTHER = "other"

@dataclass
class RecipeMetadata:
    """Metadata for a recipe."""
    name: str
    description: str
    tags: List[str]
    requires_auth: bool = False
    estimated_time: Optional[str] = None
    estimated_memory: Optional[str] = None
    created_at: Optional[datetime] = None

@dataclass
class Recipe:
    """A training recipe with metadata."""
    name: str
    description: str
    model: str
    dataset: str
    task: str
    algorithm: str
    backend: str
    config: Union[SFTConfig, UnifiedConfig]
    tags: List[str] = None
    requires_auth: bool = False
    estimated_time: Optional[str] = None
    estimated_memory: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()

class RecipeRegistry:
    """Registry for training recipes."""

    def __init__(self):
        self._recipes: Dict[str, Recipe] = {}
        self._load_builtin_recipes()

    def _load_builtin_recipes(self):
        """Load built-in recipes for popular models."""
        recipes_dir = Path(__file__).parent / "configs"
        recipes_dir.mkdir(exist_ok=True)

        # Load recipes from YAML files
        for yaml_file in recipes_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    recipe_data = yaml.safe_load(f)

                if 'recipe' in recipe_data:
                    recipe_info = recipe_data['recipe']
                    config_data = recipe_data.get('config', {})

                    # Determine config type and create config object
                    if recipe_info.get('algorithm', '').lower() in ['dpo', 'ppo', 'grpo', 'gspo']:
                        config = UnifiedConfig.from_dict(config_data)
                    else:
                        config = SFTConfig.from_dict(config_data)

                    recipe = Recipe(
                        name=recipe_info['name'],
                        description=recipe_info['description'],
                        model=recipe_info['model'],
                        dataset=recipe_info['dataset'],
                        task=recipe_info['task'],
                        algorithm=recipe_info['algorithm'],
                        backend=recipe_info.get('backend', 'auto'),
                        config=config,
                        tags=recipe_info.get('tags', []),
                        requires_auth=recipe_info.get('requires_auth', False),
                        estimated_time=recipe_info.get('estimated_time'),
                        estimated_memory=recipe_info.get('estimated_memory')
                    )

                    self._recipes[recipe.name] = recipe

            except Exception as e:
                print(f"Failed to load recipe from {yaml_file}: {e}")

    def register_recipe(self, recipe: Recipe):
        """Register a new recipe."""
        if recipe.name in self._recipes:
            print(f"Recipe '{recipe.name}' already exists, overwriting")
        self._recipes[recipe.name] = recipe

    def get_recipe(self, name: str) -> Optional[Recipe]:
        """Get a recipe by name."""
        return self._recipes.get(name)

    def list_recipes(self, tags: Optional[List[str]] = None, algorithm: Optional[str] = None) -> List[Recipe]:
        """List available recipes, optionally filtered by tags or algorithm."""
        recipes = list(self._recipes.values())

        if tags:
            recipes = [r for r in recipes if any(tag in r.tags for tag in tags)]

        if algorithm:
            recipes = [r for r in recipes if r.algorithm.lower() == algorithm.lower()]

        return recipes

    def search_recipes(self, query: str) -> List[Recipe]:
        """Search recipes by name, description, or model."""
        query_lower = query.lower()
        return [
            r for r in self._recipes.values()
            if (query_lower in r.name.lower() or
                query_lower in r.description.lower() or
                query_lower in r.model.lower() or
                query_lower in r.dataset.lower())
        ]

    def create_recipe_from_config(
        self,
        name: str,
        description: str,
        config: Union[SFTConfig, UnifiedConfig],
        tags: Optional[List[str]] = None
    ) -> Recipe:
        """Create a recipe from a configuration object."""
        # Extract metadata from config
        if isinstance(config, UnifiedConfig):
            model = config.model.name_or_path
            dataset = config.datasets[0].name if config.datasets else "unknown"
            task = "rl"
            algorithm = config.algo.value if hasattr(config.algo, 'value') else str(config.algo)
            backend = "trl"  # Default for RL
        else:
            model = config.model.name_or_path
            dataset = config.dataset.name
            task = config.dataset.task_type.value if hasattr(config.dataset.task_type, 'value') else str(config.dataset.task_type)
            algorithm = "sft"
            backend = "auto"

        recipe = Recipe(
            name=name,
            description=description,
            model=model,
            dataset=dataset,
            task=task,
            algorithm=algorithm,
            backend=backend,
            config=config,
            tags=tags or []
        )

        self.register_recipe(recipe)
        return recipe

    def save_recipe(self, recipe: Recipe, filepath: Union[str, Path]):
        """Save a recipe to a YAML file."""
        filepath = Path(filepath)

        # Convert config to dict
        config_dict = recipe.config.to_dict() if hasattr(recipe.config, 'to_dict') else {}

        recipe_data = {
            'recipe': {
                'name': recipe.name,
                'description': recipe.description,
                'model': recipe.model,
                'dataset': recipe.dataset,
                'task': recipe.task,
                'algorithm': recipe.algorithm,
                'backend': recipe.backend,
                'tags': recipe.tags,
                'requires_auth': recipe.requires_auth,
                'estimated_time': recipe.estimated_time,
                'estimated_memory': recipe.estimated_memory
            },
            'config': config_dict
        }

        with open(filepath, 'w') as f:
            yaml.dump(recipe_data, f, default_flow_style=False, indent=2, sort_keys=False)

        print(f"Saved recipe '{recipe.name}' to {filepath}")


# Global registry instance
registry = RecipeRegistry()


def list_recipes(tags: Optional[List[str]] = None, algorithm: Optional[str] = None) -> List[Recipe]:
    """List available recipes."""
    return registry.list_recipes(tags=tags, algorithm=algorithm)


def get_recipe(name: str) -> Optional[Recipe]:
    """Get a recipe by name."""
    return registry.get_recipe(name)


def search_recipes(query: str) -> List[Recipe]:
    """Search recipes by query."""
    return registry.search_recipes(query)


def create_recipe_from_config(
    name: str,
    description: str,
    config: Union[SFTConfig, UnifiedConfig],
    tags: Optional[List[str]] = None
) -> Recipe:
    """Create and register a recipe from a configuration."""
    return registry.create_recipe_from_config(name, description, config, tags)


def save_recipe(recipe: Recipe, filepath: Union[str, Path]):
    """Save a recipe to file."""
    registry.save_recipe(recipe, filepath)


# Built-in recipes
def _create_builtin_recipes():
    """Create and register built-in recipes for popular models."""

    from .config import SFTModelConfig, SFTDatasetConfig, SFTTrainingConfig, SFTLoggingConfig
    from ..core.rl.config import ModelConfig as RLModelConfig, DatasetConfig as RLDatasetConfig, TrainingConfig as RLTrainingConfig, LoggingConfig as RLLoggingConfig

    # LLaMA 3 SFT Recipe
    llama3_sft_config = SFTConfig(
        model=SFTModelConfig(
            name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
            precision="bf16",
            use_unsloth=True,
            peft_enabled=True,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            max_seq_length=4096
        ),
        dataset=SFTDatasetConfig(
            name="mlabonne/FineTome-100k",
            split="train",
            task_type="instruction_following",
            chat_template="llama3"
        ),
        train=SFTTrainingConfig(
            epochs=1,
            per_device_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=10,
            max_grad_norm=0.3,
            optimizer="adamw_8bit",
            lr_scheduler="cosine",
            packing=True,
            padding_free=True
        ),
        logging=SFTLoggingConfig(
            output_dir="./output/llama3-sft",
            run_name="llama3-instruction-tuning"
        )
    )

    llama3_recipe = Recipe(
        name="llama3-instruction-tuning",
        description="Fine-tune LLaMA 3 8B Instruct on instruction following with memory-efficient settings",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        dataset="mlabonne/FineTome-100k",
        task="instruction_following",
        algorithm="sft",
        backend="unsloth",
        config=llama3_sft_config,
        tags=["llama", "instruction-tuning", "memory-efficient", "popular"],
        requires_auth=True,
        estimated_time="2-4 hours",
        estimated_memory="24GB"
    )
    registry.register_recipe(llama3_recipe)

    # Qwen 3 DPO Recipe
    qwen3_dpo_config = UnifiedConfig(
        algo="dpo",
        model=RLModelConfig(
            name_or_path="Qwen/Qwen2.5-3B-Instruct",
            precision="bf16",
            use_peft=True,
            lora_r=8,
            lora_alpha=16,
            max_seq_length=2048
        ),
        datasets=[
            RLDatasetConfig(
                name="argilla/ultrafeedback-binarized-preferences-cleaned",
                split="train",
                field_mappings={
                    "prompt": "instruction",
                    "chosen": "chosen_response",
                    "rejected": "rejected_response"
                }
            )
        ],
        train=RLTrainingConfig(
            epochs=1,
            per_device_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            max_grad_norm=0.3,
            beta=0.1,
            max_length=2048,
            max_prompt_length=512,
            optimizer="adamw_8bit",
            lr_scheduler="cosine"
        ),
        logging=RLLoggingConfig(
            output_dir="./output/qwen3-dpo",
            run_name="qwen3-preference-tuning"
        )
    )

    qwen3_recipe = Recipe(
        name="qwen3-preference-tuning",
        description="Train Qwen 3 3B on preference data using DPO for alignment",
        model="Qwen/Qwen2.5-3B-Instruct",
        dataset="argilla/ultrafeedback-binarized-preferences-cleaned",
        task="preference_optimization",
        algorithm="dpo",
        backend="trl",
        config=qwen3_dpo_config,
        tags=["qwen", "dpo", "alignment", "popular"],
        requires_auth=True,
        estimated_time="1-2 hours",
        estimated_memory="16GB"
    )
    registry.register_recipe(qwen3_recipe)

    # LLaMA 3 GRPO Recipe
    llama3_grpo_config = UnifiedConfig(
        algo="grpo",
        model=RLModelConfig(
            name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
            precision="bf16",
            use_peft=True,
            lora_r=16,
            lora_alpha=32,
            max_seq_length=4096,
            reward_value_model="meta-llama/Llama-3.2-1B-Instruct"
        ),
        datasets=[
            RLDatasetConfig(
                name="hendrydong/MATH",
                split="train",
                format_type="math"
            )
        ],
        train=RLTrainingConfig(
            max_steps=500,
            per_device_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            max_grad_norm=0.3,
            grpo_alpha=0.1,
            grpo_beta=0.1,
            temperature=0.7,
            optimizer="adamw_8bit",
            lr_scheduler="cosine"
        ),
        logging=RLLoggingConfig(
            output_dir="./output/llama3-grpo",
            run_name="llama3-math-reasoning"
        )
    )

    llama3_grpo_recipe = Recipe(
        name="llama3-math-reasoning",
        description="Train LLaMA 3 for mathematical reasoning using GRPO reinforcement learning",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        dataset="hendrydong/MATH",
        task="math_reasoning",
        algorithm="grpo",
        backend="trl",
        config=llama3_grpo_config,
        tags=["llama", "grpo", "math", "reasoning", "advanced"],
        requires_auth=True,
        estimated_time="4-8 hours",
        estimated_memory="32GB"
    )
    registry.register_recipe(llama3_grpo_recipe)


# Initialize built-in recipes
_create_builtin_recipes()


# CLI convenience functions
def show_recipe_info(name: str):
    """Display detailed information about a recipe."""
    recipe = get_recipe(name)
    if not recipe:
        print(f"Recipe '{name}' not found.")
        return

    print(f"\n🍳 Recipe: {recipe.name}")
    print(f"📝 Description: {recipe.description}")
    print(f"🤖 Model: {recipe.model}")
    print(f"📊 Dataset: {recipe.dataset}")
    print(f"🎯 Task: {recipe.task}")
    print(f"⚙️  Algorithm: {recipe.algorithm}")
    print(f"🔧 Backend: {recipe.backend}")

    if recipe.tags:
        print(f"🏷️  Tags: {', '.join(recipe.tags)}")

    if recipe.requires_auth:
        print("🔐 Requires authentication: Yes")

    if recipe.estimated_time:
        print(f"⏱️  Estimated time: {recipe.estimated_time}")

    if recipe.estimated_memory:
        print(f"💾 Estimated memory: {recipe.estimated_memory}")

    print()


def load_recipe_from_yaml(filepath: Union[str, Path]) -> Recipe:
    """Load a recipe from a YAML file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Recipe file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # Parse the config based on algorithm
    algorithm = data.get('algorithm', 'sft')
    if algorithm.lower() in ['dpo', 'ppo', 'grpo', 'gspo']:
        # RL config
        config = UnifiedConfig(
            model=data.get('model', {}),
            dataset=data.get('dataset', {}),
            training=data.get('training', {}),
            logging=data.get('logging', {})
        )
    else:
        # SFT config
        config = SFTConfig(
            model=data.get('model', {}),
            dataset=data.get('dataset', {}),
            training=data.get('training', {}),
            logging=data.get('logging', {})
        )

    return Recipe(
        name=data['name'],
        description=data.get('description', ''),
        model=data['model']['name_or_path'],
        dataset=data['dataset']['name'],
        task=data.get('task', 'instruction-tuning'),
        algorithm=algorithm,
        backend=data.get('backend', 'trl'),
        config=config,
        tags=data.get('tags', []),
        requires_auth=data.get('requires_auth', False),
        estimated_time=data.get('estimated_time'),
        estimated_memory=data.get('estimated_memory')
    )

def load_builtin_recipes() -> List[Recipe]:
    """Load all built-in recipes."""
    return list_recipes()

def list_available_recipes(tags: Optional[List[str]] = None, algorithm: Optional[str] = None):
    """List all available recipes with filtering."""
    recipes = list_recipes(tags=tags, algorithm=algorithm)

    if not recipes:
        print("No recipes found matching criteria.")
        return

    print(f"\n🍳 Available Recipes ({len(recipes)} found):")
    print("-" * 80)

    for recipe in recipes:
        auth_indicator = "🔐" if recipe.requires_auth else "  "
        print(f"{auth_indicator} {recipe.name:<25} | {recipe.description[:50]}...")
        print(f"{'':<27} | Model: {recipe.model} | Algorithm: {recipe.algorithm}")

    print()


__all__ = [
    'RecipeType',
    'ModelFamily',
    'RecipeMetadata',
    'Recipe',
    'RecipeRegistry',
    'registry',
    'list_recipes',
    'get_recipe',
    'search_recipes',
    'create_recipe_from_config',
    'save_recipe',
    'load_recipe_from_yaml',
    'load_builtin_recipes',
    'show_recipe_info',
    'list_available_recipes'
]