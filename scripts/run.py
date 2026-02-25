import torch
import json
import os
import sys

from src.utils import set_seed, count_parameters
from src.data import get_scifar10_dataloaders
from src.model import sCIFAR10_GRU
from src.train import train_model

def main():
    set_seed(14)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Running on {device}.")

    # hyperparams
    EPOCHS = 40
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-4

    os.makedirs('./docs/report', exist_ok=True)

    print("\n=====| Loading s-CIFAR-10 Data |=====\n")
    train_loader, test_loader = get_scifar10_dataloaders(batch_size=BATCH_SIZE)
    results = {}

    print("="*50)
    print("Experiment 1: Baseline GRU (w/ Learnable Biases)")
    print("="*50 + "\n")

    baseline_model = sCIFAR10_GRU(ablate_biases=False).to(device)
    count_parameters(baseline_model)

    baseline_history = train_model(
        model=baseline_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        device=device,
        learning_rate=LEARNING_RATE
    )
    results['baseline'] = baseline_history

    print("=" * 50)
    print("Experiment 2: Ablated GRU (Frozen Gate Biases)")
    print("=" * 50 + "\n")

    set_seed(14)
    ablated_model = sCIFAR10_GRU(ablate_biases=True).to(device)
    count_parameters(ablated_model)

    ablated_history = train_model(
        model=ablated_model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        epochs=EPOCHS, 
        device=device,
        learning_rate=LEARNING_RATE
    )
    results['ablated'] = ablated_history

    results_path = './docs/report/results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nExperiments complete; results saved to {results_path}.")
    print("Now please run `scripts/plot_results.py` to generate figures.")

if __name__ == "__main__":
    main()