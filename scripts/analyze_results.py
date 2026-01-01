import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = 'outputs'
FIGURES_DIR = 'outputs/figures_analysis'
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def analyze_iid():
    path = os.path.join(OUTPUT_DIR, 'fedavg_iid_metrics.json')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    data = load_json(path)
    df = pd.DataFrame(data)
    
    print("\n--- IID Analysis ---")
    print(f"Max Test Acc: {df['test_acc'].max():.2f}%")
    print(f"Max Val Acc: {df['val_acc'].max():.2f}%")
    print(f"Final Test Acc: {df['test_acc'].iloc[-1]:.2f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['round'], df['train_loss'], label='Train Loss')
    plt.plot(df['round'], df['val_loss'], label='Val Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('FedAvg IID: Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'iid_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df['round'], df['test_acc'], label='Test Acc')
    plt.plot(df['round'], df['val_acc'], label='Val Acc')
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
        
        # Get last test accuracy (or max)
        test_accs = data.get('test_acc', [])
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
    analyze_iid()
    analyze_non_iid_sweep()
