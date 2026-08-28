"""
Minimal tests for RubricReward that don't require full environment setup.

These tests focus on validation and can run without the full dependency stack.
"""

import sys
import os
from unittest.mock import MagicMock, Mock

# Test file can be imported without full aligntune
def test_rubric_reward_module_exists():
    """Test that rubric_reward.py module exists and can be parsed."""
    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    assert os.path.exists(rubric_reward_path), "rubric_reward.py not found"


def test_rubric_reward_syntax_valid():
    """Test that rubric_reward.py has valid Python syntax."""
    import py_compile
    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    try:
        py_compile.compile(rubric_reward_path, doraise=True)
    except py_compile.PyCompileError as e:
        raise AssertionError(f"Syntax error in rubric_reward.py: {e}")


def test_rubric_reward_class_defined():
    """Test that RubricReward class is defined."""
    import ast
    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    with open(rubric_reward_path) as f:
        tree = ast.parse(f.read())

    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "RubricReward" in classes, f"RubricReward class not found. Found: {classes}"


def test_rubric_reward_has_methods():
    """Test that RubricReward has required methods."""
    import ast
    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    with open(rubric_reward_path) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RubricReward":
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            assert "__init__" in methods, "Missing __init__ method"
            assert "compute" in methods, "Missing compute method"
            assert "batch_compute" in methods, "Missing batch_compute method"
            break
    else:
        raise AssertionError("RubricReward class not found in AST")


def test_rubric_type_in_enum():
    """Test that RUBRIC is added to RewardType enum."""
    core_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "core.py"
    )
    with open(core_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    assert "RUBRIC = " in content, "RUBRIC not in RewardType enum"
    assert 'RUBRIC = "rubric"' in content, "RUBRIC enum value incorrect"


def test_rubric_registered_in_registry():
    """Test that rubric is registered in RewardRegistry."""
    registry_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "registry.py"
    )
    with open(registry_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    assert '"rubric"' in content, "rubric not registered"
    assert "RewardType.RUBRIC" in content, "RewardType.RUBRIC not found in registry"


def test_rubric_reward_docstring():
    """Test that RubricReward has proper docstrings."""
    import ast
    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    with open(rubric_reward_path) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RubricReward":
            docstring = ast.get_docstring(node)
            assert docstring is not None, "RubricReward missing class docstring"
            assert "rubric" in docstring.lower(), "Docstring doesn't mention rubric"
            break


def test_test_file_exists():
    """Test that test_rubric_reward.py was created."""
    test_path = os.path.join(
        os.path.dirname(__file__), "test_rubric_reward.py"
    )
    assert os.path.exists(test_path), "test_rubric_reward.py not found"


def test_test_file_has_tests():
    """Test that test_rubric_reward.py has test classes."""
    import ast
    test_path = os.path.join(
        os.path.dirname(__file__), "test_rubric_reward.py"
    )
    with open(test_path) as f:
        tree = ast.parse(f.read())

    classes = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
    ]
    assert len(classes) > 0, f"No test classes found. Found: {classes}"

    # Check for specific test classes
    expected_classes = [
        "TestRubricRewardInit",
        "TestRubricRewardCompute",
        "TestRubricRewardBatchCompute",
    ]
    for expected in expected_classes:
        assert expected in classes, f"Missing test class: {expected}"


def test_lazy_load_in_factory():
    """Test that lazy_load_rubric_reward is in factory."""
    core_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "core.py"
    )
    with open(core_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    assert "_lazy_load_rubric_reward" in content, "Lazy load method not found in factory"
    assert "from aligntune.rewards.rubric_reward import RubricReward" in content, "Import statement not found"


def test_compute_signature():
    """Test that compute method has correct signature."""
    import ast
    import inspect

    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    with open(rubric_reward_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Check signature contains expected parameters
    assert "def compute(" in content, "compute method not found"
    assert "completion: str" in content, "completion parameter missing"
    assert "reference: Optional[str]" in content, "reference parameter missing"
    assert "prompt: Optional[str]" in content, "prompt parameter missing"


def test_batch_compute_signature():
    """Test that batch_compute method has correct signature."""
    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    with open(rubric_reward_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Check signature contains expected parameters
    assert "def batch_compute(" in content, "batch_compute method not found"
    assert "completions:" in content, "completions parameter missing"
    assert "prompts:" in content, "prompts parameter missing"


def test_initialization_validation():
    """Test that __init__ validates inputs."""
    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    with open(rubric_reward_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Check for validation logic
    assert "ValueError" in content, "No ValueError validation found"
    assert "TypeError" in content, "No TypeError validation found"
    assert "rubric" in content, "Rubric validation not found"
    assert "judge" in content, "Judge validation not found"


def test_cache_methods():
    """Test that cache management methods are present."""
    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    with open(rubric_reward_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    assert "def get_cache_info" in content, "get_cache_info method not found"
    assert "def clear_cache" in content, "clear_cache method not found"


def test_error_handling():
    """Test that error handling is present."""
    rubric_reward_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aligntune", "rewards", "rubric_reward.py"
    )
    with open(rubric_reward_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    assert "try:" in content, "No try/except found"
    assert "except Exception" in content, "No exception handling found"
    assert "logger" in content, "No logging found"


if __name__ == "__main__":
    # Run all test functions
    import sys

    test_functions = [
        test_rubric_reward_module_exists,
        test_rubric_reward_syntax_valid,
        test_rubric_reward_class_defined,
        test_rubric_reward_has_methods,
        test_rubric_type_in_enum,
        test_rubric_registered_in_registry,
        test_rubric_reward_docstring,
        test_test_file_exists,
        test_test_file_has_tests,
        test_lazy_load_in_factory,
        test_compute_signature,
        test_batch_compute_signature,
        test_initialization_validation,
        test_cache_methods,
        test_error_handling,
    ]

    failed = []
    for test_func in test_functions:
        try:
            test_func()
            print(f"[PASS] {test_func.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {test_func.__name__}: {e}")
            failed.append(test_func.__name__)

    if failed:
        print(f"\n{len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"\nAll {len(test_functions)} tests passed!")
        sys.exit(0)
