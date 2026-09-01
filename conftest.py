"""
Root conftest.py — ensures the src/ layout is on sys.path for pytest
so that `import aligntune` works without a package installation step.
"""
import sys
from pathlib import Path

# Add src/ to sys.path so pytest can find the aligntune package
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
