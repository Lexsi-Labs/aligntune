import logging
import shutil
import json
import os
from pathlib import Path
import numpy as np

# Disable WandB for automated testing
os.environ["WANDB_MODE"] = "disabled"

# 1. Core Imports
from aligntune.core.rl.config_loader import ConfigLoader
from aligntune.core.rl.trainer_factory import TrainerFactory
from aligntune.core.rl.models import ModelManager
from aligntune.core.rl.registries import DatasetRegistry

# 2. Evaluation Imports
from aligntune.eval import BaseEvaluator, RLEvaluator, Metric
from aligntune.eval.metrics import RougeMetric, BleuMetric

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Unified_Test")

# ==============================================================================
# Custom Metric Implementation
# ==============================================================================
class CustomDiversityMetric(Metric):
    """
    A custom metric to measure vocabulary diversity (Unique words / Total words).
    Demonstrates support for custom evaluation logic.
    """
    def __init__(self):
        super().__init__("custom_diversity")

    def compute(self, predictions, references, **kwargs):
        if not predictions:
            return {"custom_diversity": 0.0}
        
        scores = []
        for text in predictions:
            words = str(text).lower().split()
            if not words:
                scores.append(0.0)
                continue
            unique_ratio = len(set(words)) / len(words)
            scores.append(unique_ratio)
            
        return {"custom_diversity": float(np.mean(scores))}

# ==============================================================================
# Helper Functions
# ==============================================================================
def create_dummy_dataset(path: str):
    """Create a lightweight dummy dataset for testing."""
    data = [
        {
            "prompt": "Question: What is 2+2?\n\nAnswer:", 
            "target": "The answer is 4." 
        },
        {
            "prompt": "Question: Capital of France?\n\nAnswer:", 
            "target": "Paris" 
        },
        {
            "prompt": "Translate 'Hello' to Spanish.",
            "target": "Hola"
        }
    ] * 5  # Repeat to make a small batch
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)
    logger.info(f"Created dummy dataset at {path}")

def run_test():
    # Setup paths
    output_dir = "./output/test_unified"
    data_path = "./output/dummy_data/unified_data.json"
    
    # Create data
    create_dummy_dataset(data_path)

    # ==============================================================================
    # STEP 1: Configuration
    # ==============================================================================
    logger.info("Step 1: Setting up Configuration")
    
    config_dict = {
        "algo": "grpo",
        "model": {
            "name_or_path": "Qwen/Qwen2.5-0.5B",
            "precision": "bf16",
            "use_peft": True,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "datasets": [
            {
                "name": data_path,
                "split": "train",
                "max_samples": 10,
            }
        ],
        "rewards": [{"type": "length", "weight": 0.1}],
        "train": {
            "epochs": 1,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "save_interval": 10,
            "num_generations": 8,
        },
        "logging": {
            "output_dir": output_dir,
            "loggers": ["tensorboard"]
        }
    }

    config = ConfigLoader.load_from_dict(config_dict)

    # ==============================================================================
    # STEP 2: SFT / Generic Evaluation (Pre-training Baseline)
    # ==============================================================================
    logger.info("Step 2: Running SFT Evaluation (Generic Metrics + Custom)")
    
    # Load Model & Data
    manager = ModelManager(config.model)
    model = manager.load_policy_model().model
    tokenizer = manager.tokenizer
    
    dataset = DatasetRegistry.load_dataset("local", path=data_path, split="train")
    
    # Initialize BaseEvaluator with:
    # 1. Generic Metrics (Perplexity - default)
    # 2. Text Gen Metrics (ROUGE, BLEU)
    # 3. Custom Metric (Diversity)
    sft_evaluator = BaseEvaluator(
        metrics=[RougeMetric(), BleuMetric(), CustomDiversityMetric()],
        batch_size=4,
        use_cache=False
    )
    
    sft_results = sft_evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        task_name="sft_baseline",
        max_samples=5
    )
    
    print("\n--- SFT Baseline Results ---")
    for k, v in sft_results.items():
        print(f"{k}: {v:.4f}")
    print("----------------------------\n")

    # ==============================================================================
    # STEP 3: Training (GRPO)
    # ==============================================================================
    logger.info("Step 3: Running GRPO Training")
    
    trainer = TrainerFactory.create_trainer(config)
    train_result = trainer.train()
    
    model_save_path = output_dir
    if isinstance(train_result, dict) and 'model_path' in train_result:
        model_save_path = train_result['model_path']
        
    logger.info(f"Training finished. Models saved to: {model_save_path}")

    # ==============================================================================
    # STEP 4: RL Evaluation (Post-training)
    # ==============================================================================
    logger.info("Step 4: Running RL Evaluation (KL, Reward Acc)")

    # Update config to load trained weights
    eval_config = config
    eval_config.model.name_or_path = model_save_path
    
    # Load Policy (Trained)
    manager = ModelManager(eval_config.model)
    policy_model = manager.load_policy_model().model
    
    # Load Reference (Original Base)
    eval_config.model.name_or_path = "Qwen/Qwen2.5-0.5B" 
    ref_manager = ModelManager(eval_config.model)
    reference_model = ref_manager.load_reference_model().model

    # Define a dummy reward function for testing
    def dummy_reward_func(text):
        return float(len(str(text)))

    # Initialize RLEvaluator
    # Note: RLEvaluator inherits from BaseEvaluator, so it technically supports SFT metrics too
    rl_evaluator = RLEvaluator(batch_size=4, use_cache=False)

    rl_results = rl_evaluator.evaluate_rl(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_model=dummy_reward_func,
        max_samples=5 
    )

    # ==============================================================================
    # STEP 5: Final Report
    # ==============================================================================
    print("\n" + "="*50)
    print("FINAL UNIFIED TEST REPORT")
    print("="*50)
    print("PHASE 1: SFT/Generic Metrics")
    for k, v in sft_results.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nPHASE 2: RL Specific Metrics")
    for k, v in rl_results.items():
        print(f"  {k}: {v:.4f}")
    print("="*50)

    # Cleanup
    logger.info("Cleaning up...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    if os.path.exists("./output/dummy_data"):
        shutil.rmtree("./output/dummy_data", ignore_errors=True)

if __name__ == "__main__":
    run_test()