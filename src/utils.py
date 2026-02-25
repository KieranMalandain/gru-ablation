# src/utils.py
import torch
import random
import numpy as np

def set_seed(seed: int = 14):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module, verbose: bool = True) -> int:
    """
    Verifies the total trainable parameter count.
    Standrard GRU: Input X, hidden H, output classes C:
    GRU Params = 3H(X+H+2)
    Linear Params = C(H+1)
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print("="*30)
        print(f"{'Layer':<20} | {'Shape':<20} | {'Parameters':<10}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name:<20} | {str(list(param.shape)):<20} | {param.numel():<10}")
        print('-'*30)
        print(f"Total Trainable Params: {total_params:,}")
        print("="*30)
    return total_params
