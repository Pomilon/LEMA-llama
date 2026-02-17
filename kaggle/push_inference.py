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
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if config["kaggle_username"] == "YOUR-KAGGLE-USERNAME":
        print("❌ Please update 'kaggle_username' in config.json")
        sys.exit(1)
        
    return config

def push_inference_notebook():
    """Push inference notebook to Kaggle using CLI."""
    
    config = load_config()
    username = config["kaggle_username"]
    
    kaggle_dir = os.path.join(BASE_DIR, "kaggle")
    metadata_src = os.path.join(kaggle_dir, "inference-metadata.json")
    metadata_dest = os.path.join(kaggle_dir, "kernel-metadata.json")
    
    # Check if files exist
    if not os.path.exists(os.path.join(kaggle_dir, "inference_notebook.ipynb")):
        print("❌ inference_notebook.ipynb not found. Run build_inference_notebook.py first.")
        return False
        
    # Create kernel metadata
    kernel_slug = "lema-inference-demo"
    
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": "LEMA Inference Demo",
        "code_file": "inference_notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    with open(metadata_dest, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"🚀 Pushing {username}/{kernel_slug}...")
    
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", kaggle_dir],
        capture_output=True,
        text=True
    )
    
    # Restore original metadata (optional, but good for cleanliness)
    os.remove(metadata_dest)
    
    if result.returncode == 0:
        print("✅ Inference Notebook pushed successfully!")
        print(result.stdout)
    else:
        print("❌ Push failed:")
        print(result.stderr)
        return False
    
    return True

if __name__ == "__main__":
    push_inference_notebook()
