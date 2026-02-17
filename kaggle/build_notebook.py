import json
import os
import sys
from typing import List, Dict

# Ensure we can find files relative to lema-demo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_code_cell(source_lines: List[str]) -> Dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }

def create_markdown_cell(text: str) -> Dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split('\n')]
    }

def read_file_content(rel_path: str) -> str:
    path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")
    with open(path, "r") as f:
        return f.read()

def build_kaggle_notebook(output_path: str = "kaggle/notebook.ipynb"):
    """Build a Kaggle notebook from training/train.py using %writefile magic."""
    
    cells = []
    
    # 1. Header
    cells.append(create_markdown_cell(
        "# LEMA Fine-Tuning Demonstration\n"
        "This notebook demonstrates fine-tuning Llama-2-7B using LEMA.\n"
        "It sets up the environment, generates data, and runs the training loop."
    ))
    
    # 2. Install dependencies
    cells.append(create_code_cell([
        "!pip install -q transformers safetensors accelerate",
        "# Clone LEMA repository (using main branch for demo)",
        "!git clone https://github.com/Pomilon/LEMA.git",
        "!pip install -q -e LEMA/"
    ]))
    
    # 3. Directory Setup
    setup_dirs = [
        "import os",
        "",
        "# Create directories",
        "os.makedirs('training', exist_ok=True)",
        "os.makedirs('data', exist_ok=True)",
        "os.makedirs('checkpoints', exist_ok=True)",
        "os.makedirs('utils', exist_ok=True)",
        "",
        "# Create __init__.py for packages",
        "with open('training/__init__.py', 'w') as f: pass",
        "with open('data/__init__.py', 'w') as f: pass",
        "with open('utils/__init__.py', 'w') as f: pass",
    ]
    cells.append(create_code_cell(setup_dirs))

    # 4. Write files using %%writefile
    # List of (source_rel_path, target_rel_path_in_notebook)
    files_to_write = [
        ("training/lema_integration.py", "training/lema_integration.py"),
        ("training/checkpoint_manager.py", "training/checkpoint_manager.py"),
        ("utils/seed_utils.py", "utils/seed_utils.py"),
        ("utils/logging_utils.py", "utils/logging_utils.py"),
        ("data/build_dataset.py", "data/build_dataset.py"),
        ("training/train.py", "training/train.py"),
    ]

    for source_path, dest_path in files_to_write:
        content = read_file_content(source_path)
        # Prepend magic command
        lines = [f"%%writefile {dest_path}"] + content.split('\n')
        cells.append(create_code_cell(lines))
    
    # 5. Generate Dataset
    cells.append(create_code_cell([
        "# Generate Dataset",
        "!python data/build_dataset.py"
    ]))
    
    # 6. Run Training
    cells.append(create_code_cell([
        "# Run Training",
        "!python training/train.py"
    ]))
    
    # Notebook Structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Save Notebook
    abs_output_path = os.path.join(BASE_DIR, output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    
    with open(abs_output_path, "w") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Notebook created: {abs_output_path}")

if __name__ == "__main__":
    build_kaggle_notebook("kaggle/notebook.ipynb")
