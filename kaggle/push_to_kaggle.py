import subprocess
import json
import os
import sys

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    config_path = os.path.join(BASE_DIR, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please run the setup script or create config.json manually.")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if config["kaggle_username"] == "YOUR-KAGGLE-USERNAME":
        print("❌ Please update 'kaggle_username' in config.json")
        sys.exit(1)
        
    return config

def push_to_kaggle(notebook_path: str = "kaggle/notebook.ipynb"):
    """Push notebook to Kaggle using CLI."""
    
    config = load_config()
    username = config["kaggle_username"]
    
    abs_notebook_path = os.path.join(BASE_DIR, notebook_path)
    if not os.path.exists(abs_notebook_path):
        print(f"❌ Notebook not found at: {abs_notebook_path}")
        return False
        
    kaggle_dir = os.path.dirname(abs_notebook_path)
    metadata_path = os.path.join(kaggle_dir, "kernel-metadata.json")
    
    # Create kernel metadata
    kernel_slug = "lema-finetuning-demo"
    
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": "LEMA Fine-Tuning Demo",
        "code_file": os.path.basename(abs_notebook_path),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f" pushing kernel {username}/{kernel_slug}...")
    
    # Push using Kaggle CLI
    # We must run this from the directory containing kernel-metadata.json usually, or specify path
    # kaggle kernels push -p <path>
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", kaggle_dir],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Notebook pushed successfully!")
        print(result.stdout)
    else:
        print("❌ Push failed:")
        print(result.stderr)
        return False
    
    return True

if __name__ == "__main__":
    push_to_kaggle("kaggle/notebook.ipynb")
