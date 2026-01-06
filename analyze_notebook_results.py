import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = 'outputMasks'
PLOT_PATH = os.path.join(OUTPUT_DIR, 'analysis_comparison.png')

def analyze_results():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Directory '{OUTPUT_DIR}' not found.")
        return

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*_metrics.json')))
    if not files:
        print(f"No metric files found in '{OUTPUT_DIR}'")
        return

    print(f"{'Experiment':<30} | {'Max Test Acc':<12} | {'Round':<5} | {'Final Acc':<10} | {'Avg Last 5':<10}")
    print("-" * 85)

    plt.figure(figsize=(10, 6))

    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            name = os.path.basename(fpath).replace('_metrics.json', '')
            
            if 'round' not in data or 'test_acc' not in data:
                print(f"Skipping {name}: missing 'round' or 'test_acc' keys.")
                continue

            rounds = data['round']
            test_acc = data['test_acc']
            
            if not rounds or not test_acc:
                print(f"Skipping {name}: empty data.")
                continue

            max_acc = max(test_acc)
            max_round = rounds[test_acc.index(max_acc)]
            final_acc = test_acc[-1]
            avg_last_5 = np.mean(test_acc[-5:]) if len(test_acc) >= 5 else np.mean(test_acc)
            
            print(f"{name:<30} | {max_acc:12.2f} | {max_round:5d} | {final_acc:10.2f} | {avg_last_5:10.2f}")
            
            plt.plot(rounds, test_acc, marker='o', markersize=3, label=f"{name} (Best: {max_acc:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    plt.title('Test Accuracy vs Communication Round')
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    print("-" * 85)
    print(f"Comparison plot saved to: {PLOT_PATH}")

if __name__ == "__main__":
    analyze_results()
