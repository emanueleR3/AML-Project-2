import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Configuration
OUTPUT_DIRS = {
    "100_epochs": "outputs100epochs",
    "100_last": "outputs100last",
    "300_epochs": "outputs300epochs"
}
ANALYSIS_DIR = "analysis_outputs"

def load_json(path):
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def run_analysis():
    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR)

    metrics = {}
    
    # 1. Load Data
    for name, dir_path in OUTPUT_DIRS.items():
        metrics[name] = {}
        
        # Central Baseline
        baseline_path = os.path.join(dir_path, "central_baseline_metrics.json")
        metrics[name]["baseline"] = load_json(baseline_path)
        
        # FedAvg IID
        iid_path = os.path.join(dir_path, "fedavg_iid_metrics.json")
        metrics[name]["iid"] = load_json(iid_path)
        
        # Non-IID (Nc=1, J=4) - Check if exists
        noniid_path = os.path.join(dir_path, "noniid_nc1_j4.json")
        metrics[name]["noniid"] = load_json(noniid_path)

    # 2. Plot Central Baseline Comparison
    plt.figure(figsize=(10, 6))
    for name, data in metrics.items():
        if data["baseline"]:
            # Handle different key names if present (some files use val_acc, some train_acc if val not present)
            # Based on file inspection, keys are 'val_acc'
            y_data = data["baseline"].get('val_acc', [])
            if not y_data:
                y_data = data["baseline"].get('train_acc', []) # Fallback
            
            x_data = range(1, len(y_data) + 1)
            plt.plot(x_data, y_data, marker='o', label=f"{name} (Max: {max(y_data):.2f}%)")
    
    plt.title("Central Baseline Training: Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(ANALYSIS_DIR, "baseline_comparison.png"))
    plt.close()

    # 3. Plot FedAvg IID Comparison
    plt.figure(figsize=(10, 6))
    for name, data in metrics.items():
        if data["iid"]:
            # FedAvg metrics usually have 'round' and 'val_acc'
            x_data = data["iid"]['round']
            y_data = data["iid"]['val_acc']
            
            # Filter out NaNs if any (eval might be every few rounds)
            points = [(x, y) for x, y in zip(x_data, y_data) if y is not None and not (isinstance(y, float) and pd.isna(y))]
            if points:
                xs, ys = zip(*points)
                plt.plot(xs, ys, marker='x', label=f"{name} (Max: {max(ys):.2f}%)")

    plt.title("FedAvg IID: Validation Accuracy vs Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(ANALYSIS_DIR, "fedavg_iid_comparison.png"))
    plt.close()
    
    # 4. Plot Non-IID Comparison
    plt.figure(figsize=(10, 6))
    for name, data in metrics.items():
        if data["noniid"]:
            x_data = data["noniid"]['round']
            y_data = data["noniid"]['val_acc']
            
            points = [(x, y) for x, y in zip(x_data, y_data) if y is not None and not (isinstance(y, float) and pd.isna(y))]
            if points:
                xs, ys = zip(*points)
                plt.plot(xs, ys, marker='s', linestyle='--', label=f"{name} (Max: {max(ys):.2f}%)")

    plt.title("Non-IID (Nc=1, J=4): Validation Accuracy vs Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(ANALYSIS_DIR, "noniid_comparison.png"))
    plt.close()

    # 5. Summary Table
    summary = []
    for name, data in metrics.items():
        row = {"Experiment": name}
        
        # Baseline
        if data["baseline"]:
             vals = data["baseline"].get('val_acc', [])
             row["Baseline Best Acc"] = max(vals) if vals else "N/A"
             row["Baseline Epochs"] = len(vals)
        else:
            row["Baseline Best Acc"] = "N/A"
            row["Baseline Epochs"] = 0
            
        # IID
        if data["iid"]:
            vals = [v for v in data["iid"]['val_acc'] if v is not None and not pd.isna(v)]
            row["IID Best Val Acc"] = max(vals) if vals else "N/A"
            row["IID Rounds"] = data["iid"]['round'][-1] if data["iid"]['round'] else 0
        else:
             row["IID Best Val Acc"] = "N/A"
             row["IID Rounds"] = 0
             
        # Non-IID
        if data["noniid"]:
            vals = [v for v in data["noniid"]['val_acc'] if v is not None and not pd.isna(v)]
            row["Non-IID Best Val Acc"] = max(vals) if vals else "N/A"
        else:
             row["Non-IID Best Val Acc"] = "N/A"

        summary.append(row)

    df = pd.DataFrame(summary)
    print("\n=== Experiment Summary ===")
    print(df.to_string(index=False))
    
    df.to_csv(os.path.join(ANALYSIS_DIR, "summary_metrics.csv"), index=False)
    print(f"\nAnalysis saved to {ANALYSIS_DIR}")

if __name__ == "__main__":
    run_analysis()
