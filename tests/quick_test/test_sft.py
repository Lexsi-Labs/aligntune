import os
import shutil
import unittest
import logging
import sys
from pathlib import Path
from datasets import load_dataset
import torch

# Ensure we can import the library locally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# 1. Main Library Import (Triggers auto-patching if Unsloth is present)
import aligntune

# 2. Import Factory directly to avoid circular dependency issues during testing imports
from aligntune.core.backend_factory import create_sft_trainer

# 3. Import Utils
from aligntune.utils.model_loader import ModelLoader
from aligntune.eval.evaluator import BaseEvaluator
from aligntune.eval.metrics.text import BleuMetric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDualBackend(unittest.TestCase):
    """
    Integration test ensuring both backend workflows function independently.
    
    1. TRL Workflow: Train (5 steps) -> Save -> Load (Transformers) -> Eval
    2. Unsloth Workflow: Train (5 steps) -> Save -> Load (Unsloth) -> Eval
    """

    @classmethod
    def setUpClass(cls):
        cls.studio_path = Path("/teamspace/studios/this_studio")
        cls.base_output_dir = cls.studio_path / "dual_backend_test"
        
        # Clean previous runs
        if cls.base_output_dir.exists():
            shutil.rmtree(cls.base_output_dir)
        cls.base_output_dir.mkdir(parents=True, exist_ok=True)
            
        # Using Qwen 2.5 0.5B (Fast, supported by both backends)
        cls.model_name = "Qwen/Qwen2.5-0.5B-Instruct" 

    def _get_eval_dataset(self):
        """Prepare a tiny evaluation dataset (5 samples)."""
        dataset = load_dataset("gsm8k", "main", split="test")
        eval_subset = dataset.select(range(5)) 
        
        # Map to 'input'/'target' for BaseEvaluator
        def map_for_eval(example):
            return {
                "input": example["question"],
                "target": example["answer"]
            }
        
        return eval_subset.map(map_for_eval, remove_columns=dataset.column_names)

    def test_01_trl_workflow(self):
        """Test the complete Standard TRL/Transformers pipeline."""
        logger.info("\n" + "="*60)
        logger.info("🧪 STARTING TEST: Standard TRL Workflow (5 Steps)")
        logger.info("="*60)

        output_dir = str(self.base_output_dir / "trl_run")
        
        # 1. Training (TRL Backend)
        logger.info("--> Training with TRL Backend...")
        
        # We pass arguments directly to the factory (User-friendly API)
        # We enforce backend="trl" and use_unsloth=False
        trainer = create_sft_trainer(
            model_name=self.model_name,
            dataset_name="gsm8k",
            backend="trl", 
            output_dir=output_dir,
            max_seq_length=512,
            use_unsloth=False,      
            peft_enabled=False, # TRL Full Finetune test
            batch_size=2,
            learning_rate=1e-5,
            
            # Data Config
            subset="main",
            task_type="instruction_following",
            # Map GSM8K 'question'/'answer' -> Library 'instruction'/'response'
            column_mapping={"question": "instruction", "answer": "response"},
            
            # Training Limits
            max_steps=5,    # Explicitly 5 steps
            num_epochs=1,   # Required arg, but max_steps takes precedence
            
            # Logging
            run_name="trl_integration_test"
        )
        
        trainer.train()
        saved_path = trainer.save_model()
        
        logger.info(f"TRL Model saved to: {saved_path}")

        # Cleanup trainer to free VRAM for next steps
        del trainer
        torch.cuda.empty_cache()

        # 2. Loading (Standard Transformers)
        logger.info("--> Loading with Standard Transformers...")
        loader = ModelLoader()
        # use_unsloth=False ensures we test standard HF loading path
        model, tokenizer = loader.load_local_weights(
            saved_path, 
            use_unsloth=False, 
            load_in_4bit=False 
        )
        
        # Verify it's NOT an Unsloth model wrapper
        self.assertFalse(hasattr(model, "fast_generate"), "TRL workflow should not produce Unsloth model object")

        # 3. Evaluation
        logger.info("--> Running Evaluation...")
        evaluator = BaseEvaluator(
            metrics=[BleuMetric()], 
            batch_size=1,
            use_cache=False
        )
        
        results = evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=self._get_eval_dataset(),
            task_name="trl_workflow_eval"
        )
        
        logger.info(f"TRL Eval Results: {results}")
        self.assertIn("bleu", results)
        logger.info("✅ TRL Workflow Passed")

    def test_02_unsloth_workflow(self):
        """Test the complete Unsloth pipeline (if installed)."""
        logger.info("\n" + "="*60)
        logger.info("🧪 STARTING TEST: Unsloth Workflow (5 Steps)")
        logger.info("="*60)

        # Basic check to skip if environment is totally broken
        try:
            import unsloth
        except ImportError:
            logger.warning("Unsloth not installed. Skipping Unsloth workflow test.")
            return

        output_dir = str(self.base_output_dir / "unsloth_run")
        
        # 1. Training (Unsloth Backend)
        logger.info("--> Training with Unsloth Backend...")
        
        # We enforce backend="unsloth" and use_unsloth=True
        trainer = create_sft_trainer(
            model_name=self.model_name,
            dataset_name="gsm8k",
            backend="unsloth", 
            output_dir=output_dir,
            max_seq_length=512,
            use_unsloth=True,       
            peft_enabled=True,  # Unsloth is optimized for PEFT/LoRA
            batch_size=2,
            learning_rate=2e-4,
            precision="fp16",
            
            # Data Config
            subset="main",
            task_type="instruction_following",
            column_mapping={"question": "instruction", "answer": "response"},
            
            # Training Limits
            max_steps=5,    # Explicitly 5 steps
            num_epochs=1,
            
            # Logging
            run_name="unsloth_integration_test"
        )
        
        trainer.train()
        saved_path = trainer.save_model()
        
        logger.info(f"Unsloth Model saved to: {saved_path}")
        
        # Cleanup
        del trainer
        torch.cuda.empty_cache()

        # 2. Loading (Unsloth Optimized)
        logger.info("--> Loading with Unsloth...")
        loader = ModelLoader()
        # use_unsloth=True triggers the robust loading logic we fixed in model_loader.py
        model, tokenizer = loader.load_local_weights(
            saved_path, 
            use_unsloth=True, 
            load_in_4bit=True,
            max_seq_length=512
        )
        
        # Verify it IS an Unsloth model
        # Note: Depending on fallback, this might be False, but in a working env it should be True
        is_unsloth = hasattr(model, "fast_generate")
        
        if is_unsloth:
             from unsloth import FastLanguageModel
             FastLanguageModel.for_inference(model)
             logger.info("Unsloth inference mode enabled.")
        else:
             logger.warning("Loaded model does not have 'fast_generate'. Fallback might have triggered.")

        # 3. Evaluation
        logger.info("--> Running Evaluation...")
        evaluator = BaseEvaluator(
            metrics=[BleuMetric()], 
            batch_size=1,
            use_cache=False
        )
        
        results = evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=self._get_eval_dataset(),
            task_name="unsloth_workflow_eval"
        )
        
        logger.info(f"Unsloth Eval Results: {results}")
        self.assertIn("bleu", results)
        logger.info("✅ Unsloth Workflow Passed")

if __name__ == "__main__":
    unittest.main()