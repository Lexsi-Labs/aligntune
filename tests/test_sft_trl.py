"""
SFT dual-backend integration test — TRL backend only.

Split out from the former combined test_sft.py (which mixed "trl" and
"unsloth" backend workflows in one file/process) because Unsloth
monkey-patches transformers/trl globally on import, which was silently
breaking later, TRL-only tests run in the same process. See
test_sft_unsloth.py for the Unsloth-backend workflow.

1. TRL Workflow: Train (5 steps) -> Save -> Load (Transformers) -> Eval
"""

import os
import shutil
import tempfile
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


class TestSFTTRLWorkflow(unittest.TestCase):
    """
    Integration test for the TRL SFT workflow: Train (5 steps) -> Save ->
    Load (Transformers) -> Eval.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_output_dir = Path(tempfile.mkdtemp(prefix="sft_trl_test_"))

        # Clean previous runs
        if cls.base_output_dir.exists():
            shutil.rmtree(cls.base_output_dir)
        cls.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Model is overridable via --model CLI option (see conftest.py).
        cls.model_name = os.environ.get("ALIGNTUNE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    def _get_eval_dataset(self):
        """Prepare a tiny evaluation dataset (5 samples)."""
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        eval_subset = dataset.select(range(5))

        # Map to 'input'/'target' for BaseEvaluator
        def map_for_eval(example):
            return {
                "input": example["question"],
                "target": example["answer"]
            }

        return eval_subset.map(map_for_eval, remove_columns=dataset.column_names)

    def test_trl_workflow(self):
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
            dataset_name="openai/gsm8k",
            backend="trl",
            output_dir=output_dir,
            max_seq_length=512,
            use_unsloth=False,
            peft_enabled=False, # TRL Full Finetune test
            batch_size=2,
            learning_rate=1e-5,

            # Data Config — gsm8k has 'question'/'answer' columns; use question directly
            subset="main",
            dataset_text_field="question",

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


if __name__ == "__main__":
    unittest.main()
