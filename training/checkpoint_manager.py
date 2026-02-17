import os
import glob
import json
from typing import Optional

class CheckpointManager:
    def __init__(self, output_dir: str = "./checkpoints"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint directory."""
        # LEMA saves checkpoints as subdirectories in output_dir
        # Assuming format or just any subdirectory that looks like a checkpoint
        # Usually checkpoint-500, checkpoint-1000 etc.
        checkpoints = sorted(glob.glob(f"{self.output_dir}/*"))
        # Filter for directories
        checkpoints = [d for d in checkpoints if os.path.isdir(d)]
        
        if not checkpoints:
            return None
            
        # Sort by modification time to get the truly latest one
        checkpoints.sort(key=os.path.getmtime)
        return checkpoints[-1]
    
    def should_resume(self) -> bool:
        """Check if we should resume from a checkpoint."""
        return self.get_latest_checkpoint() is not None
    
    def save_metadata(self, step: int, loss: float, **kwargs):
        """Save training metadata."""
        metadata = {
            "step": step,
            "loss": float(loss) if loss is not None else 0.0,
            **kwargs
        }
        # Save to a separate metadata file or directory to avoid cluttering checkpoints if needed
        # But instructions say "Include metadata (step number, loss, timestamp)"
        # LEMA's automatic checkpointing might not include this custom metadata file inside the checkpoint dir
        # So we save it to the main output dir with a step suffix
        with open(f"{self.output_dir}/metadata-{step}.json", "w") as f:
            json.dump(metadata, f, indent=2)
