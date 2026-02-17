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

def load_config():
    config_path = os.path.join(BASE_DIR, "config.json")
    if not os.path.exists(config_path):
        return {"hf_repo_id": "YOUR-USERNAME/LEMA-llama-2-7b"}
    
    with open(config_path, "r") as f:
        return json.load(f)

def build_inference_notebook(output_path: str = "kaggle/inference_notebook.ipynb"):
    """Build a Kaggle notebook for INFERENCE using the fine-tuned model."""
    
    config = load_config()
    hf_repo_id = config.get("hf_repo_id", "YOUR-USERNAME/LEMA-llama-2-7b")
    
    cells = []
    
    # 1. Header
    cells.append(create_markdown_cell(
        "# LEMA Inference & Merging Demonstration\n"
        "This notebook loads the fine-tuned LEMA model (adapter weights) and runs inference to verify the custom chat format.\n"
        "It also demonstrates how to merge the adapter into the base model if needed."
    ))
    
    # 2. Install dependencies
    cells.append(create_code_cell([
        "!pip install -q transformers safetensors accelerate",
        "# Clone LEMA repository",
        "!git clone https://github.com/Pomilon/LEMA.git",
        "!pip install -q -e LEMA/"
    ]))
    
    # 3. Directory Setup & Imports
    setup_dirs = [
        "import os",
        "",
        "os.makedirs('inference/framework', exist_ok=True)",
        "os.makedirs('inference/engines', exist_ok=True)",
        "os.makedirs('inference', exist_ok=True)",
        "os.makedirs('checkpoints/final', exist_ok=True)",
        "",
        "# Create __init__.py",
        "with open('inference/__init__.py', 'w') as f: pass",
        "with open('inference/framework/__init__.py', 'w') as f: pass",
        "with open('inference/engines/__init__.py', 'w') as f: pass",
    ]
    cells.append(create_code_cell(setup_dirs))

    # 4. Write Framework Files using %%writefile
    files_to_write = [
        ("inference/framework/model_handler.py", "inference/framework/model_handler.py"),
        ("inference/framework/chat_parser.py", "inference/framework/chat_parser.py"),
        ("inference/framework/conversation.py", "inference/framework/conversation.py"),
        ("inference/engines/cli_engine.py", "inference/engines/cli_engine.py"),
        ("tools/merge_adapter.py", "tools/merge_adapter.py"),
    ]

    for source_path, dest_path in files_to_write:
        content = read_file_content(source_path)
        # Create directory for the file if needed
        dir_name = os.path.dirname(dest_path)
        if dir_name:
            cells.append(create_code_cell([f"import os; os.makedirs('{dir_name}', exist_ok=True)"]))
            
        lines = [f"%%writefile {dest_path}"] + content.split('\n')
        cells.append(create_code_cell(lines))
        
    # 5. Instructions for Uploading Weights
    cells.append(create_markdown_cell(
        "## Upload Your Weights\n"
        "1. Create a Kaggle Dataset containing your `adapter_model.bin` and `lema_config.json`.\n"
        "2. Add the dataset to this notebook.\n"
        "3. Copy the files to `checkpoints/final/` below.\n"
        "\n"
        "For this demo, we assume the dataset is mounted at `/kaggle/input/lema-finetuned-weights/`."
    ))
    
    cells.append(create_code_cell([
        "# Example copy command (adjust path to your dataset)",
        "# !cp /kaggle/input/lema-finetuned-weights/* checkpoints/final/",
        "",
        "# Verify files",
        "!ls -l checkpoints/final/"
    ]))

    # 6. Prepare Base Model (GBI)
    cells.append(create_markdown_cell(
        "## Prepare Base Model\n"
        "We need the monolithic `.safetensors` file for LEMA to function."
    ))
    cells.append(create_code_cell([
        "import sys",
        "import os",
        "sys.path.append(os.path.abspath('LEMA/src'))",
        "",
        "from lema.utils.model_utils import prepare_monolithic_safetensors",
        "",
        "MODEL_NAME = 'NousResearch/Llama-2-7b-hf'",
        "MODEL_PATH = 'llama2_7b.safetensors'",
        "",
        "if not os.path.exists(MODEL_PATH):",
        "    print(f'Preparing {MODEL_PATH}...')",
        "    prepare_monolithic_safetensors(MODEL_NAME, MODEL_PATH, device='auto')",
    ]))

    # 7. Run Inference Script
    cells.append(create_markdown_cell("## Run Inference"))
    
    inference_script = """
import sys
import torch
from inference.framework.model_handler import LemaModelHandler
from inference.framework.chat_parser import ChatParser

# Setup
checkpoint_path = "checkpoints/final"
handler = LemaModelHandler(checkpoint_path, device="cuda")
parser = ChatParser()

# Test Prompts
questions = [
    "What is LEMA?",
    "Who invented the telephone?",
    "What is photosynthesis?"
]

print("-" * 60)
for q in questions:
    print(f"\nUser: {q}")
    prompt = parser.format_prompt(q)
    
    # Generate
    response_text = handler.generate(prompt, max_new_tokens=128)
    
    print(f"\nRaw Output:\n{response_text}")
    
    # Parse
    parsed = parser.parse_response(response_text)
    if parsed.is_valid:
        print(f"\n✅ Valid LEMA Format!")
        print(f"Answer: {parsed.answer}")
        print(f"Confidence: {parsed.confidence}")
    else:
        print(f"\n❌ Invalid Format")

print("-" * 60)
"""
    cells.append(create_code_cell(inference_script.split('\n')))

    # 8. Merge Adapter (Optional)
    cells.append(create_markdown_cell(
        "## Merge Adapter (Export)\n"
        "Convert the LEMA adapter + Base Model into a standard HuggingFace model."
    ))
    
    # NOTE: LEMA doesn't have a built-in merge utility exposed in public API yet for export.
    # We would need to implement a simple one: load base, apply lora, save.
    # For now, we can leave this as a placeholder or implement a basic script.
    
    merge_script = """
# Merge Adapter and Stream Upload to Hugging Face
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# Authentication
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("✅ Logged in via Kaggle Secrets")
except:
    hf_token = input("Enter HF Token (Write):")
    login(token=hf_token)

REPO_ID = "{hf_repo_id}" # Change this to your repo

!python tools/merge_adapter.py \\
    --checkpoint checkpoints/final \\
    --output merged_model \\
    --base_model llama2_7b.safetensors \\
    --repo_id {{REPO_ID}} \\
    --token {{hf_token}}

print("\\n✅ Streaming Merge & Upload Complete!")
""".format(hf_repo_id=hf_repo_id)
    cells.append(create_code_cell(merge_script.split('\n')))

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
    
    print(f"✅ Inference Notebook created: {abs_output_path}")

if __name__ == "__main__":
    build_inference_notebook("kaggle/inference_notebook.ipynb")
