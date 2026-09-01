"""
Tests for Indic evaluation integration.

Tests the lm-eval integration with Indic language benchmarks without
executing actual model evaluations (CPU-only, no model loading).
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Minimal imports to avoid heavy dependencies during testing
try:
    from aligntune.eval.lm_eval_integration import (
        INDIC_TASKS,
        LMEvalTask,
        get_available_indic_tasks,
        get_available_indic_tasks_by_language,
    )
except ImportError as e:
    pytest.skip(f"Skipping tests due to import error: {e}", allow_module_level=True)


class TestIndicTaskRegistry:
    """Test Indic task registration and retrieval."""

    def test_indic_tasks_registered(self):
        """Test that Indic tasks are registered."""
        assert len(INDIC_TASKS) > 0
        # Check specific categories
        assert any("hi" in task for task in INDIC_TASKS.keys())
        assert any("ta" in task for task in INDIC_TASKS.keys())
        assert any("bn" in task for task in INDIC_TASKS.keys())

    def test_milu_tasks_present(self):
        """Test MILU (Indic MMLU) tasks are registered."""
        milu_tasks = [t for t in INDIC_TASKS.keys() if t.startswith("milu")]
        assert len(milu_tasks) >= 3, "Should have MILU tasks for at least 3 languages"
        assert "milu_hi" in INDIC_TASKS
        assert "milu_ta" in INDIC_TASKS
        assert "milu_bn" in INDIC_TASKS

    def test_indicxtreme_tasks_present(self):
        """Test IndicXTREME tasks are registered."""
        indicxtreme_keywords = ["indicopa", "indicsentiment", "indicxnli", "indicqa"]
        indicxtreme_tasks = [
            t for t in INDIC_TASKS.keys()
            if any(k in t.lower() for k in indicxtreme_keywords)
        ]
        assert len(indicxtreme_tasks) > 0, "Should have IndicXTREME tasks"

    def test_genbench_tasks_present(self):
        """Test IndicGenBench tasks are registered."""
        genbench_keywords = ["floresin", "crosssumin", "xquadin"]
        genbench_tasks = [
            t for t in INDIC_TASKS.keys()
            if any(k in t.lower() for k in genbench_keywords)
        ]
        assert len(genbench_tasks) > 0, "Should have GenBench tasks"

    def test_sarvam_tasks_present(self):
        """Test Sarvam Indic evaluation tasks are registered."""
        sarvam_keywords = ["mmlu_in", "gsm8k_in", "triviaqa_in"]
        sarvam_tasks = [
            t for t in INDIC_TASKS.keys()
            if any(k in t.lower() for k in sarvam_keywords)
        ]
        assert len(sarvam_tasks) > 0, "Should have Sarvam tasks"

    def test_task_properties(self):
        """Test that task definitions have required properties."""
        for task_name, task_def in INDIC_TASKS.items():
            assert isinstance(task_def, LMEvalTask)
            assert task_def.name, f"Task {task_name} has no name"
            assert task_def.category, f"Task {task_name} has no category"
            assert task_def.description, f"Task {task_name} has no description"
            assert task_def.lm_eval_task_name, f"Task {task_name} has no lm_eval_task_name"
            assert task_def.metrics, f"Task {task_name} has no metrics"


class TestIndicTaskRetrieval:
    """Test retrieval functions for Indic tasks."""

    def test_get_available_indic_tasks(self):
        """Test retrieval of all Indic tasks."""
        tasks = get_available_indic_tasks()
        assert len(tasks) > 0
        assert all(isinstance(t, str) for t in tasks)

    def test_get_tasks_by_language_hindi(self):
        """Test retrieval of Hindi tasks."""
        tasks = get_available_indic_tasks_by_language("hi")
        assert len(tasks) > 0
        assert all("_hi" in t for t in tasks)

    def test_get_tasks_by_language_tamil(self):
        """Test retrieval of Tamil tasks."""
        tasks = get_available_indic_tasks_by_language("ta")
        assert len(tasks) > 0
        assert all("_ta" in t for t in tasks)

    def test_get_tasks_by_language_bengali(self):
        """Test retrieval of Bengali tasks."""
        tasks = get_available_indic_tasks_by_language("bn")
        assert len(tasks) > 0
        assert all("_bn" in t for t in tasks)

    def test_get_tasks_by_language_aliases(self):
        """Test that language aliases work."""
        hi_tasks_short = get_available_indic_tasks_by_language("hi")
        hi_tasks_long = get_available_indic_tasks_by_language("hindi")
        assert set(hi_tasks_short) == set(hi_tasks_long)

        ta_tasks_short = get_available_indic_tasks_by_language("ta")
        ta_tasks_long = get_available_indic_tasks_by_language("tamil")
        assert set(ta_tasks_short) == set(ta_tasks_long)

    def test_get_tasks_by_language_invalid(self):
        """Test error handling for invalid languages."""
        with pytest.raises(ValueError):
            get_available_indic_tasks_by_language("xx")


class TestIndicBenchmarkConfiguration:
    """Test configuration of Indic benchmark evaluation."""

    def test_language_filtering_basic(self):
        """Test that language filtering mechanism works correctly."""
        # Verify we can filter by language
        hi_tasks = get_available_indic_tasks_by_language("hi")
        ta_tasks = get_available_indic_tasks_by_language("ta")

        # Should be different sets
        assert set(hi_tasks) != set(ta_tasks)
        # All Hindi tasks should have hi in name
        assert all("_hi" in t for t in hi_tasks)
        # All Tamil tasks should have ta in name
        assert all("_ta" in t for t in ta_tasks)


class TestTaskMetrics:
    """Test task metrics configuration."""

    def test_milu_uses_exact_match(self):
        """Test that MILU tasks use exact_match metric."""
        milu_tasks = {k: v for k, v in INDIC_TASKS.items() if k.startswith("milu")}
        for task in milu_tasks.values():
            assert "exact_match" in task.metrics

    def test_indicqa_uses_f1(self):
        """Test that IndicQA tasks use F1 metric."""
        qa_tasks = {k: v for k, v in INDIC_TASKS.items() if "indicqa" in k}
        for task in qa_tasks.values():
            assert "f1" in task.metrics

    def test_floresin_uses_bleu(self):
        """Test that FloresIN (MT) tasks use BLEU metric."""
        mt_tasks = {k: v for k, v in INDIC_TASKS.items() if "floresin" in k}
        for task in mt_tasks.values():
            assert "bleu" in task.metrics


class TestTaskCategories:
    """Test task categorization."""

    def test_qa_tasks_categorized(self):
        """Test QA tasks are properly categorized."""
        qa_keywords = ["indicqa", "xquadin", "triviaqa"]
        for task_name, task in INDIC_TASKS.items():
            if any(k in task_name for k in qa_keywords):
                assert task.category in ["question_answering", "qa"]

    def test_translation_tasks_categorized(self):
        """Test translation tasks are properly categorized."""
        for task_name, task in INDIC_TASKS.items():
            if "floresin" in task_name:
                assert task.category == "machine_translation"

    def test_reasoning_tasks_categorized(self):
        """Test reasoning tasks are properly categorized."""
        for task_name, task in INDIC_TASKS.items():
            if "xnli" in task_name:
                assert task.category == "reasoning"


class TestTaskNaming:
    """Test task naming conventions."""

    def test_language_code_in_task_name(self):
        """Test that task names contain language codes."""
        for task_name in INDIC_TASKS.keys():
            # Most tasks should have explicit language code
            has_lang = "_hi" in task_name or "_ta" in task_name or "_bn" in task_name
            assert has_lang, f"Task {task_name} doesn't have language code"

    def test_consistent_naming(self):
        """Test naming follows consistent patterns."""
        for task_name, task in INDIC_TASKS.items():
            # Task name should use underscores and be lowercase
            assert task_name.islower()
            assert "_" in task_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
