import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path

OUTPUT_DIR = 'outputs'
FIGURES_DIR = 'outputs/figures_analysis'
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def analyze_baseline():
    path = os.path.join(OUTPUT_DIR, 'central_baseline_metrics.json')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    data = load_json(path)
    
    train_loss = data.get('train_loss', [])
    train_acc = data.get('train_acc', [])
    val_loss = data.get('val_loss', [])
    val_acc = data.get('val_acc', [])
    
    print("\n--- Central Baseline Analysis ---")
    print(f"Training epochs: {len(train_loss)}")
    
    if val_acc:
        print(f"Final Val Acc: {val_acc[-1]:.2f}%")
        print(f"Best Val Acc: {max(val_acc):.2f}%")
    
    if train_acc:
        print(f"Final Train Acc: {train_acc[-1]:.2f}%")
    
    if train_loss:
        print(f"Final Train Loss: {train_loss[-1]:.4f}")
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    epochs_train = list(range(1, len(train_loss) + 1))
    plt.plot(epochs_train, train_loss, label='Train Loss')
    
    if val_loss:
        # Val loss is evaluated less frequently, calculate corresponding epochs
        eval_interval = len(train_loss) // len(val_loss) if val_loss else 1
        epochs_val = [eval_interval * (i + 1) for i in range(len(val_loss))]
        plt.plot(epochs_val, val_loss, label='Val Loss', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Central Baseline: Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'baseline_loss.png'))
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, train_acc, label='Train Acc')
    
    if val_acc:
        eval_interval = len(train_acc) // len(val_acc) if val_acc else 1
        epochs_val = [eval_interval * (i + 1) for i in range(len(val_acc))]
        plt.plot(epochs_val, val_acc, label='Val Acc', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Central Baseline: Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'baseline_accuracy.png'))
    plt.close()
    
    print(f"Baseline plots saved to {FIGURES_DIR}")

def analyze_iid():
    path = os.path.join(OUTPUT_DIR, 'fedavg_iid_metrics.json')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    data = load_json(path)
    df = pd.DataFrame(data)
    
    # Filter out NaNs for stats
    valid_test = df['test_acc'].dropna()
    valid_val = df['val_acc'].dropna()

    print("\n--- IID Analysis ---")
    if not valid_test.empty:
        print(f"Max Test Acc: {valid_test.max():.2f}%")
        print(f"Final Test Acc: {valid_test.iloc[-1]:.2f}%")
    else:
        print("No valid Test Acc data found.")

    if not valid_val.empty:
        print(f"Max Val Acc: {valid_val.max():.2f}%")
    else:
        print("No valid Val Acc data found.")

    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    
    # Plot Train Loss (usually non-sparse)
    train_loss = df.dropna(subset=['train_loss'])
    plt.plot(train_loss['round'], train_loss['train_loss'], label='Train Loss')
    
    # Plot Val Loss (sparse, drop NaNs to connect lines)
    val_loss = df.dropna(subset=['val_loss'])
    if not val_loss.empty:
        plt.plot(val_loss['round'], val_loss['val_loss'], label='Val Loss', marker='o') # Added marker
        
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('FedAvg IID: Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'iid_loss.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    
    # Plot Test Acc (sparse)
    test_acc = df.dropna(subset=['test_acc'])
    if not test_acc.empty:
        plt.plot(test_acc['round'], test_acc['test_acc'], label='Test Acc', marker='o') # Added marker
        
    # Plot Val Acc (sparse)
    val_acc = df.dropna(subset=['val_acc'])
    if not val_acc.empty:
        plt.plot(val_acc['round'], val_acc['val_acc'], label='Val Acc', marker='x') # Added marker

    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('FedAvg IID: Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'iid_accuracy.png'))
    plt.close()
    print(f"IID plots saved to {FIGURES_DIR}")

def analyze_non_iid_sweep():
    # Find all non-iid files
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('noniid_nc') and f.endswith('.json')]
    
    results = []
    
    for f in files:
        # Parse filename: noniid_nc10_j16.json
        parts = f.replace('noniid_', '').replace('.json', '').split('_')
        nc = int(parts[0].replace('nc', ''))
        j = int(parts[1].replace('j', ''))
        
        path = os.path.join(OUTPUT_DIR, f)
        data = load_json(path)
        
        # Get last test accuracy (or max) - FILTER NaNs
        raw_test_accs = data.get('test_acc', [])
        # Filter None or NaN
        test_accs = [x for x in raw_test_accs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        
        final_acc = test_accs[-1] if test_accs else 0
        best_acc = max(test_accs) if test_accs else 0
        
        results.append({
            'Nc': nc,
            'J': j,
            'Final Test Acc': final_acc,
            'Best Test Acc': best_acc,
            'Filename': f
        })
        
    if not results:
        print("No Non-IID results found.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by=['Nc', 'J'])
    
    print("\n--- Non-IID Sweep Results ---")
    print(df.to_string(index=False))
    
    # Pivot for Heatmap (Final Acc)
    pivot_table = df.pivot(index='Nc', columns='J', values='Final Test Acc')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title('Non-IID Sweep: Final Test Accuracy')
    plt.savefig(os.path.join(FIGURES_DIR, 'noniid_heatmap.png'))
    plt.close()
    print(f"Non-IID heatmap saved to {FIGURES_DIR}")

if __name__ == "__main__":
    analyze_baseline()
    analyze_iid()
    analyze_non_iid_sweep()
