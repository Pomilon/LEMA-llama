import torch
from typing import Optional
from lema import LemaConfig, LemaModel

class LemaTrainingManager:
    """Clean interface to LEMA for training."""
    
    def __init__(self, config: LemaConfig):
        self.config = config
        self.model = LemaModel(config)
        self.model.initialize_lora()
        
    def get_trainer(self, optimizer: torch.optim.Optimizer):
        return self.model.get_trainer(optimizer)
    
    def save_checkpoint(self, path: str):
        self.model.save_pretrained(path)
    
    @classmethod
    def load_checkpoint(cls, path: str):
        # When loading from pretrained, we load the model first
        # LemaModel.from_pretrained returns a LemaModel instance
        model = LemaModel.from_pretrained(path)
        # Create a manager instance and attach the loaded model
        # We can extract the config from the loaded model
        manager = cls.__new__(cls)
        manager.config = model.config
        manager.model = model
        return manager
