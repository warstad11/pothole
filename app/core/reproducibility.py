
"""
Reproducibility utilities for ensuring deterministic results.
"""
import random
import numpy as np
import torch
import os

def set_global_seed(seed: int = 42):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # Make PyTorch operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed) # MPS uses same seed logic often, but good to be explicit if needed

    # Set PYTHONHASHSEED for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Global random seed set to {seed} for reproducibility")

def get_torch_generator(seed: int = 42) -> torch.Generator:
    """
    Get a PyTorch random number generator with fixed seed.
    
    Args:
        seed: Random seed value
        
    Returns:
        Configured torch.Generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
