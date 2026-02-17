"""
Handle loading and interaction with LEMA fine-tuned models.
"""

from lema import LemaModel, MemoryStrategy
from transformers import AutoTokenizer
import torch
import threading
from typing import Optional, List
import time

class LemaModelHandler:
    """Manages LEMA model loading and inference."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Load a fine-tuned LEMA model.
        
        Args:
            checkpoint_path: Path to LEMA checkpoint
            device: Device to run on (cuda/cpu)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        print(f"Loading LEMA model from {checkpoint_path}...")
        # Load model using LEMA's API
        self.model = LemaModel.from_pretrained(checkpoint_path, device=device)
        # Ensure model components are on correct device
        self.model.to(device)
        self.model.initialize_lora()
        
        # Access internal components for manual forward pass
        self.memory = self.model.memory
        self.adapter = self.model.adapter
        self.layers = self.adapter.get_layer_metadata()
        self.lora_manager = self.model.lora_manager
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.model_name_or_path
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Debug: Check LoRA weights
        lora_params = self.model.get_trainable_parameters()
        if lora_params:
            total_norm = sum(p.norm().item() for p in lora_params)
            print(f"Debug: Total LoRA weight norm: {total_norm:.4f}")
            if total_norm == 0:
                print("⚠️ Warning: LoRA weights are all zeros!")
        else:
            print("⚠️ Warning: No LoRA parameters found in model!")

        print("Model loaded successfully.")
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        stop_sequences: List[str] = ["[/LEMA_REPLY]"]
    ) -> str:
        """
        Generate text from prompt using LEMA's streaming architecture.
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=512 # limit input context
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate loop
        current_input_ids = input_ids
        
        print("Generating...", end="", flush=True)
        
        for i in range(max_new_tokens):
            with torch.no_grad():
                logits = self._forward_pass(current_input_ids)
            
            # Get last token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Sample or greedy
            if do_sample and temperature > 0:
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append
            current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
            print(".", end="", flush=True)

            # Check stop sequences
            if stop_sequences:
                decoded_so_far = self.tokenizer.decode(current_input_ids[0, -20:], skip_special_tokens=False)
                if any(stop in decoded_so_far for stop in stop_sequences):
                    break
        
        print(" Done!")
        
        # Decode
        output_text = self.tokenizer.decode(current_input_ids[0], skip_special_tokens=False)
        return output_text
    
    def _forward_pass(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass logic for LEMA models.
        Replicates LemaTrainer.train_step logic but inference only.
        """
        is_streaming = (self.model.config.strategy == MemoryStrategy.STREAMING)
        hidden_states = input_ids # Start with embeddings usually handled by first layer or similar?
        # Wait, LemaModelAdapter usually handles embeddings in the first layer or separately.
        # In LemaTrainer: hidden_states = inputs (which are input_ids)
        # So the adapter handles input_ids -> embeddings.
        
        # Prefetch first layer
        if is_streaming:
            self.memory.prefetch_to_ram(self.layers[0]['id'], 0)
            self.memory.async_transfer_to_vram(self.layers[0]['id'], 0, ram_slot=0)
            if len(self.layers) > 1:
                self.memory.prefetch_to_ram(self.layers[1]['id'], 1)
        else:
            self.memory.async_transfer_to_vram(self.layers[0]['id'], 0)
            
        for i, layer_meta in enumerate(self.layers):
            slot = i % 2
            next_slot = (i + 1) % 2
            
            flat_vram = self.memory.get_vram_flat_buffer(slot)
            
            # Prefetch next layers
            disk_thread = None
            if i + 1 < len(self.layers):
                if is_streaming:
                    self.memory.async_transfer_to_vram(self.layers[i+1]['id'], next_slot, ram_slot=next_slot)
                    if i + 2 < len(self.layers):
                        disk_thread = threading.Thread(target=self.memory.prefetch_to_ram, args=(self.layers[i+2]['id'], slot))
                        disk_thread.start()
                else:
                    self.memory.async_transfer_to_vram(self.layers[i+1]['id'], next_slot)
            
            # Construct layer
            layer_module = self.adapter.construct_layer_module(layer_meta['id'], flat_vram, self.lora_manager)
            
            # Forward
            # Note: We disable gradient checkpointing for inference
            hidden_states = self.adapter.forward_layer(layer_module, hidden_states, gradient_checkpointing=False)
            
            if disk_thread: disk_thread.join()
            
            # Release layer (move to CPU/Disk if needed, or just free VRAM pointer)
            # LemaModelAdapter.release_layer_module handles cleanup
            if hasattr(self.adapter, "release_layer_module"):
                self.adapter.release_layer_module(layer_module)
            del layer_module
            
        return hidden_states
