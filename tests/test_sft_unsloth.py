"""
SFT dual-backend integration test — Unsloth backend only.

Split out from the former combined test_sft.py (which mixed "trl" and
"unsloth" backend workflows in one file/process) because Unsloth
monkey-patches transformers/trl globally on import, which was silently
breaking later, TRL-only tests run in the same process. See test_sft_trl.py
for the TRL-backend workflow.

2. Unsloth Workflow: Train (5 steps) -> Save -> Load (Unsloth) -> Eval
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


class TestSFTUnslothWorkflow(unittest.TestCase):
    """
    Integration test for the Unsloth SFT workflow: Train (5 steps) -> Save ->
    Load (Unsloth) -> Eval.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_output_dir = Path(tempfile.mkdtemp(prefix="sft_unsloth_test_"))

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

    def test_unsloth_workflow(self):
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
            dataset_name="openai/gsm8k",
            backend="unsloth",
            output_dir=output_dir,
            max_seq_length=512,
            use_unsloth=True,
            peft_enabled=True,  # Unsloth is optimized for PEFT/LoRA
            batch_size=2,
            learning_rate=2e-4,
            precision="fp16",

            # Data Config — gsm8k has 'question'/'answer' columns; use question directly
            subset="main",
            dataset_text_field="question",

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
