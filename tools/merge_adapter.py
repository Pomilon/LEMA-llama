import os
import torch
import argparse
import gc
import psutil
import json
import shutil
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import HfApi

# Adjust path to import lema
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lema import LemaConfig, LemaModel

def get_ram_usage():
    return psutil.virtual_memory().used / 1e9

def get_disk_usage(path="."):
    total, used, free = shutil.disk_usage(path)
    return free / 1e9

def merge_adapter(checkpoint_dir: str, output_dir: str, base_model_path: str, repo_id: str = None, token: str = None):
    """
    Merges LEMA LoRA adapter into base model.
    If repo_id is provided, performs STREAMING UPLOAD:
    - Saves a shard
    - Uploads to HF
    - Deletes local shard
    This bypasses local disk limits.
    """
    print(f"[{get_ram_usage():.2f}GB RAM | {get_disk_usage():.2f}GB Disk] Loading LEMA config...")
    
    api = None
    if repo_id:
        if not token:
            print("❌ Repo ID provided but no token found.")
            return
        api = HfApi(token=token)
        print(f"🚀 Streaming Upload Enabled: Target -> {repo_id}")
    
    try:
        config = LemaConfig.from_pretrained(checkpoint_dir)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Force STREAMING strategy for merge tool, regardless of checkpoint config
    config.strategy = MemoryStrategy.STREAMING

    if not os.path.exists(config.gbi_path) and not os.path.exists(base_model_path):
        if os.path.exists(base_model_path):
            config.gbi_path = base_model_path
        else:
            print("Base model not found.")
            return

    print(f"[{get_ram_usage():.2f}GB] Initializing LEMA model...")
    config.device = "cpu"
    model = LemaModel(config)
    model.adapter._max_pool_size = 1
    
    print(f"[{get_ram_usage():.2f}GB] Loading adapter weights...")
    model.lora_manager.load_pretrained(checkpoint_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Metadata for index.json
    weight_map = {}
    
    # Define Shards (4 layers per shard)
    layers = model.adapter.get_layer_metadata()
    block_layers = [l for l in layers if l['type'] == 'block']
    shard_size = 4
    
    # Calculate Total Shards for Naming
    # Embeddings (1) + Layers (32) + Head/Norm (1) = 34 "units"
    # Embeddings is processed alone -> Shard 1
    # Layers (32) / 4 = 8 Shards
    # Head/Norm -> Final Shard
    # Total ~10 shards? Let's keep it dynamic but we need total count for proper naming "00001-of-XXXXX"
    # Actually, safetensors naming convention "model-00001-of-00005.safetensors" assumes we know total at start.
    # Let's pre-calculate.
    # 1 (Emb) + 8 (Layers) + 1 (Head) = 10 shards.
    
    total_shards = 1 + (len(block_layers) // shard_size) + 1
    if len(block_layers) % shard_size != 0: total_shards += 1 # Remainder
    
    current_shard_idx = 1
    current_shard_weights = {}
    
    def save_and_upload_shard():
        nonlocal current_shard_idx, current_shard_weights
        if not current_shard_weights: return
        
        # Proper naming from the start
        filename = f"model-{current_shard_idx:05d}-of-{total_shards:05d}.safetensors"
        filepath = os.path.join(output_dir, filename)
        
        print(f"[{get_ram_usage():.2f}GB RAM | {get_disk_usage():.2f}GB Disk] Saving {filename}...")
        save_file(current_shard_weights, filepath)
        
        # Update map
        for k in current_shard_weights.keys():
            weight_map[k] = filename
            
        # Clear memory
        current_shard_weights.clear()
        current_shard_idx += 1
        gc.collect()
        
        # UPLOAD AND DELETE
        if api:
            print(f"⬆️ Uploading {filename}...")
            try:
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload shard {current_shard_idx-1}/{total_shards}"
                )
                print(f"✅ Uploaded. Deleting local file to save space.")
                os.remove(filepath)
            except Exception as e:
                print(f"❌ Upload failed for {filename}: {e}")
                # Don't delete if upload failed, so user can manually recover if space allows
        else:
            print(f"💾 Saved locally.")

    # --- 1. Embeddings ---
    print(f"Processing Embeddings (Shard {current_shard_idx})...")
    emb_name = model.adapter.get_param_names_for_layer(0)[0]
    current_shard_weights["model.embed_tokens.weight"] = model.memory.gbi.handle.get_tensor(emb_name).clone().to(dtype=torch.float16)
    save_and_upload_shard() # Save embeddings as shard 1

    # --- 2. Transformer Layers ---
    for i, layer_meta in enumerate(block_layers):
        idx = layer_meta['block_index']
        
        if idx % 5 == 0:
            print(f"[{get_ram_usage():.2f}GB] Merging Layer {idx}...")
        
        # Load & Merge
        model.memory.prefetch_to_ram(layer_meta['id'], 0)
        flat_buffer = model.memory.ram_buffers[0]
        module = model.adapter.construct_layer_module(layer_meta['id'], flat_buffer, model.lora_manager)
        
        for _, child in module.named_modules():
            if hasattr(child, "lora_A") and hasattr(child, "base_layer"):
                scale = child.scaling
                delta = (child.lora_B.data @ child.lora_A.data) * scale
                child.base_layer.weight.data += delta.to(child.base_layer.weight.dtype)
        
        # Extract
        prefix = f"model.layers.{idx}."
        names = model.adapter.get_param_names_for_layer(layer_meta['id'])
        module_params = dict(module.named_parameters())
        
        for full_name in names:
            clean_k = full_name[len(prefix):]
            if clean_k not in module_params:
                clean_k = clean_k.replace(".weight", ".base_layer.weight")
            
            # Store in state dict (clone to detach from LEMA's reusable buffer)
            # CAST TO FP16 to save space (Standard Llama is FP16/BF16)
            current_shard_weights[full_name] = module_params[clean_k].data.clone().to(dtype=torch.float16).cpu()
            
        del module
        model.adapter.layer_pool.clear()
        gc.collect()
        
        # Check if shard is full
        if (i + 1) % shard_size == 0:
            save_and_upload_shard()

    # Save any remaining layers in buffer
    if current_shard_weights:
        save_and_upload_shard()

    # --- 3. Head / Norm ---
    print(f"Processing Head & Norm (Shard {current_shard_idx})...")
    last_layer_id = layers[-1]['id']
    model.memory.prefetch_to_ram(last_layer_id, 0)
    head_buffer = model.memory.ram_buffers[0]
    head_module = model.adapter.construct_layer_module(last_layer_id, head_buffer, model.lora_manager)
    
    current_shard_weights["model.norm.weight"] = head_module.norm.weight.data.clone().to(dtype=torch.float16)
    current_shard_weights["lm_head.weight"] = head_module.lm_head.weight.data.clone().to(dtype=torch.float16)
    
    del head_module
    gc.collect()
    
    # Save final shard
    save_and_upload_shard()
    
    # --- 4. Save Index & Configs ---
    print("Saving index and configs...")
    
    index_data = {"metadata": {}, "weight_map": weight_map}
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    
    # Auxiliary files
    config_path = os.path.join(output_dir, "config.json")
    AutoConfig.from_pretrained(config.model_name_or_path).save_pretrained(output_dir)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: {e}")

    if api:
        print("⬆️ Uploading index and configs...")
        files_to_upload = [
            "model.safetensors.index.json", "config.json", "generation_config.json",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"
        ]
        
        for fname in files_to_upload:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                try:
                    api.upload_file(
                        path_or_fileobj=fpath,
                        path_in_repo=fname,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message="Upload config/index"
                    )
                except Exception as e:
                    print(f"Failed to upload {fname}: {e}")
        print("✅ Streaming Upload Complete!")
    else:
        print("✅ Local Merge Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="llama2_7b.safetensors")
    parser.add_argument("--repo_id", type=str, default=None, help="HF Repo ID for streaming upload")
    parser.add_argument("--token", type=str, default=None, help="HF Token")
    
    args = parser.parse_args()
    merge_adapter(args.checkpoint, args.output, args.base_model, args.repo_id, args.token)
