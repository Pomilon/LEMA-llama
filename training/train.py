import os
import sys
import torch
import json
import psutil
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Add project root to python path to allow imports from lema-demo modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lema import LemaConfig, LemaModel, MemoryStrategy
from lema.utils.model_utils import prepare_monolithic_safetensors
from training.lema_integration import LemaTrainingManager
from training.checkpoint_manager import CheckpointManager
from utils.seed_utils import seed_everything
from utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger(__name__)

# Constants
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
MODEL_FILENAME = "llama2_7b.safetensors"
DATASET_PATH = "data/training_data.jsonl"
OUTPUT_DIR = "checkpoints"
MAX_LENGTH = 512
BATCH_SIZE = 8 # Increased to 8. Higher BS = more compute per layer = better latency hiding for LEMA streaming.
LEARNING_RATE = 1e-4

class ChatDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.examples = []
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.examples.append(data['text'])
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Squeeze to remove batch dimension added by tokenizer
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        # For Causal LM, labels are usually the input_ids
        # We want to predict the next token.
        # The model handles shifting internally.
        labels = input_ids.clone()
        
        # Mask padding tokens in labels so we don't train on them
        # (Assuming pad_token_id is set correctly)
        if self.tokenizer.pad_token_id is not None:
             labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def main():
    seed_everything(42)
    logger.info("Starting LEMA training pipeline")

    # 1. Model Preparation
    if not os.path.exists(MODEL_FILENAME):
        logger.info(f"Preparing monolithic safetensors for {MODEL_NAME}...")
        # Check if we have GPU available for faster conversion if possible, or just CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # The user guide says 'auto' to save RAM
        prepare_monolithic_safetensors(MODEL_NAME, MODEL_FILENAME, device="auto")
        logger.info("Model preparation complete.")
    else:
        logger.info(f"Found existing model file: {MODEL_FILENAME}")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Dataset
    dataset = ChatDataset(DATASET_PATH, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Checkpoint Manager
    checkpoint_manager = CheckpointManager(OUTPUT_DIR)
    
    # 5. Initialize LEMA
    # Check if resuming
    start_step = 0
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    
    if latest_checkpoint:
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
        # When resuming, we load the model from the checkpoint
        # However, LEMA currently loads config + adapters.
        # The base model path (gbi_path) is in the config.
        # So we can just use from_pretrained.
        model = LemaModel.from_pretrained(latest_checkpoint)
        # We need to re-attach the optimizer if we want to resume optimizer state
        # LEMA's save_checkpoint saves optimizer.pt
        # LEMA's load logic for optimizer is usually handled manually or by trainer
        # Let's see if LemaModel.from_pretrained handles optimizer.
        # Looking at API docs: "Loads a LEMA model from a directory containing lema_config.json and adapter_model.bin."
        # It doesn't mention optimizer.
        # We might need to load optimizer state manually if critical, but for fine-tuning it's often okay to restart optimizer or
        # check if LemaTrainer handles it. 
        # For this demo, let's assume we create a new trainer and maybe load optimizer if possible, 
        # or just proceed with loaded weights.
        
        # Create config for reference (it's loaded in model.config)
        config = model.config
        
        # Try to parse step from checkpoint name if possible
        try:
            # Assuming format checkpoint-XXX
            start_step = int(os.path.basename(latest_checkpoint).split('-')[-1])
        except ValueError:
            start_step = 0
            
    else:
        logger.info("Initializing new LEMA model")
        config = LemaConfig(
            model_name_or_path=MODEL_NAME,
            gbi_path=MODEL_FILENAME,
            device="cuda" if torch.cuda.is_available() else "cpu",
            strategy=MemoryStrategy.STREAMING,
            lora_rank=16,
            lora_alpha=32,
            gradient_checkpointing=True, # Important for memory
            save_steps=500,
            output_dir=OUTPUT_DIR
        )
        model = LemaModel(config)
        model.initialize_lora()

    # 6. Trainer Setup
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=LEARNING_RATE)
    trainer = model.get_trainer(optimizer)
    
    # If we resumed, we might want to load optimizer state if it exists
    if latest_checkpoint:
         optimizer_path = os.path.join(latest_checkpoint, "optimizer.pt")
         if os.path.exists(optimizer_path):
             logger.info(f"Loading optimizer state from {optimizer_path}")
             optimizer.load_state_dict(torch.load(optimizer_path))

    # 7. Training Loop
    logger.info("Starting training loop...")
    total_steps = len(dataloader)
    
    # We might need to skip steps if resuming within an epoch, 
    # but since we are doing 1 epoch and dataloader is shuffled, exact state resumption is hard without saving dataloader state.
    # We will just continue training for the remaining number of steps if we can, or just run for an epoch.
    # For simplicity in this demo, we run 1 full epoch over the data.
    # If resuming, we could just run for (total_steps - start_step) if we want to be precise about epoch count,
    # but re-running data is fine for robustness in this demo.
    
    current_step = start_step
    
    for i, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch['input_ids'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        # Forward & Backward
        # Trainer handles gradient accumulation if implemented, or we do it here if needed.
        # LemaTrainer.train_step does one step.
        logits, loss = trainer.train_step(input_ids, labels=labels)
        
        current_step += 1
        
        if current_step % 10 == 0:
            vram_gb = torch.cuda.memory_allocated() / 1e9
            ram_gb = psutil.virtual_memory().used / 1e9
            logger.info(f"Step {current_step}/{total_steps} | Loss: {loss:.4f} | VRAM: {vram_gb:.2f}GB | RAM: {ram_gb:.2f}GB")
            checkpoint_manager.save_metadata(current_step, loss, vram=vram_gb, ram=ram_gb)
            
        # LEMA handles automatic checkpointing based on save_steps in config.
        # But we can also force save at the end.

    logger.info("Training complete.")
    
    # Save final model
    final_save_path = os.path.join(OUTPUT_DIR, "final")
    logger.info(f"Saving final model to {final_save_path}")
    trainer.save_checkpoint(final_save_path)
    logger.info("Done!")

if __name__ == "__main__":
    main()
