#!/usr/bin/env python3
"""
Utility script to convert Python training scripts to Jupyter notebooks.
"""

import json
import sys
from pathlib import Path


def create_notebook_from_script(script_path: Path, notebook_path: Path, title: str, description: str):
    """Convert a Python training script to a Jupyter notebook."""
    
    # Read the Python script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Split script into logical sections
    cells = []
    
    # Cell 1: Title and description (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            f"{description}\n",
            "\n",
            "This notebook demonstrates fine-tuning using AlignTune framework.\n",
            "\n",
            "## Setup\n",
            "\n",
            "First, we need to install AlignTune and its dependencies."
        ]
    })
    
    # Cell 2: Installation (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "colab": {
                "base_uri": "https://localhost:8080/",
                "height": 1000
            },
            "id": "install_cell"
        },
        "outputs": [],
        "source": [
            "# Clone and install AlignTune\n",
            "!git clone https://github.com/zeralyngkhoi19/FinetuneHub_Internal.git\n",
            "%cd FinetuneHub_Internal\n",
            "!pip install -e ."
        ]
    })
    
    # Cell 3: Imports section (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Imports and Configuration\n",
            "\n",
            "Import necessary libraries and set up configuration."
        ]
    })
    
    # Cell 4: Extract imports and setup from script
    lines = script_content.split('\n')
    import_lines = []
    config_lines = []
    function_lines = []
    main_lines = []
    
    in_imports = True
    in_config = False
    in_functions = False
    in_main = False
    
    for line in lines:
        # Skip shebang and docstring
        if line.startswith('#!/') or (line.startswith('"""') and in_imports):
            continue
        if line.strip().startswith('"""') and not line.strip().endswith('"""'):
            in_imports = False
            continue
        if line.strip().endswith('"""') and not line.strip().startswith('"""'):
            in_imports = True
            continue
        
        # Collect imports
        if in_imports and (line.startswith('import ') or line.startswith('from ')):
            import_lines.append(line)
        elif line.startswith('# =') and 'CONFIGURATION' in line:
            in_config = True
            in_imports = False
            config_lines.append(line)
        elif line.startswith('# =') and 'HELPER' in line:
            in_config = False
            in_functions = True
            function_lines.append(line)
        elif line.startswith('def main'):
            in_functions = False
            in_main = True
            main_lines.append(line)
        elif in_imports and line.strip() and not line.strip().startswith('#'):
            in_imports = False
        elif in_config:
            config_lines.append(line)
        elif in_functions:
            function_lines.append(line)
        elif in_main:
            main_lines.append(line)
    
    # Cell 4: Imports and setup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "imports_cell"},
        "outputs": [],
        "source": import_lines + ["", "sys.path.insert(0, str(Path(__file__).parent.parent.parent))"]
    })
    
    # Cell 5: Configuration (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Configuration\n",
            "\n",
            "Set up model, dataset, and training hyperparameters."
        ]
    })
    
    # Cell 6: Configuration code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "config_cell"},
        "outputs": [],
        "source": config_lines
    })
    
    # Cell 7: Helper functions (markdown)
    if function_lines:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Helper Functions\n",
                "\n",
                "Define utility functions for evaluation and memory management."
            ]
        })
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "functions_cell"},
            "outputs": [],
            "source": function_lines
        })
    
    # Cell 8: Training (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Training Pipeline\n",
            "\n",
            "Run pre-training evaluation, training, and post-training evaluation."
        ]
    })
    
    # Cell 9: Main execution
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "main_cell"},
        "outputs": [],
        "source": main_lines + ["", "if __name__ == '__main__':", "    main()"]
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    # Write notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Created notebook: {notebook_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_notebook_from_script.py <script_path> <notebook_path> [title] [description]")
        sys.exit(1)
    
    script_path = Path(sys.argv[1])
    notebook_path = Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else script_path.stem.replace('_', ' ').title()
    description = sys.argv[4] if len(sys.argv) > 4 else f"Training script: {script_path.name}"
    
    create_notebook_from_script(script_path, notebook_path, title, description)
