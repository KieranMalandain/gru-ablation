# scripts/plot.py
import json
import matplotlib.pyplot as plt
import os

def load_results(filepath='./docs/report/results.json'):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_academic_results(results, save_dir='./docs/report/figures'):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(results['baseline']['train_loss']) + 1)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'figure.dpi': 300
    })

    # Test accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(epochs, results['baseline']['test_acc'], label='Baseline (Learnable Biases)', 
            color='#1f77b4', marker='o', markersize=6)
    ax.plot(epochs, results['ablated']['test_acc'], label='Ablated (Frozen Biases)', 
            color='#d62728', marker='s', markersize=6, linestyle='--')
    
    ax.set_title('s-CIFAR-10 (T=1024) Test Accuracy vs. Epochs', pad=15, fontweight='bold')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Classification Accuracy')
    ax.set_ylim([0.05, 0.45])
    ax.set_xticks(epochs)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'))
    plt.close()

    # Training loss
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(epochs, results['baseline']['train_loss'], label='Baseline (Learnable Biases)', 
            color='#1f77b4', marker='o', markersize=6)
    ax.plot(epochs, results['ablated']['train_loss'], label='Ablated (Frozen Biases)', 
            color='#d62728', marker='s', markersize=6, linestyle='--')
    
    ax.set_title('s-CIFAR-10 Training Loss Landscape', pad=15, fontweight='bold')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Cross Entropy Loss')
    ax.set_xticks(epochs)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss_comparison.png'))
    plt.close()
    
    print(f"Figures successfully generated in {save_dir}")

if __name__ == "__main__":
    results = load_results()
    plot_academic_results(results)