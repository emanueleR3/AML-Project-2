import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (8, 5)
})

from pathlib import Path
OUTPUT_DIR = str(Path(__file__).parent.resolve())
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_clean_metric(data, metric_name):
    """Extracts metric list, removing None/NaN values and returning (indices, values)"""
    if metric_name not in data:
        return np.array([]), np.array([])
    
    values = data[metric_name]
    indices = []
    clean_values = []
    
    x_axis = data.get('epoch') or data.get('round')
    
    for i, v in enumerate(values):
        if v is not None and not np.isnan(v):
            if x_axis and i < len(x_axis):
                indices.append(x_axis[i])
            else:
                indices.append(i + 1)
            clean_values.append(v)
            
    return np.array(indices), np.array(clean_values)

def generate_table_1_and_plots():
    print("Generating Table 1 and Plots 1a/1b (Scheduler Sweep)...")
    source_dir = os.path.join(OUTPUT_DIR, "scheduler_sweep")
    if not os.path.exists(source_dir):
        print(f"Skipping Table 1: {source_dir} not found")
        return

    results = []
    loss_data = {}
    acc_data = {}

    for filename in os.listdir(source_dir):
        if not filename.endswith(".json"): continue
        
        filepath = os.path.join(source_dir, filename)
        data = load_json(filepath)
        if not data: continue

        scheduler = filename.replace("scheduler_", "").replace(".json", "").capitalize()
        
        epochs, test_acc = get_clean_metric(data, 'test_acc')
        _, test_loss = get_clean_metric(data, 'test_loss')
        
        final_acc = data.get('final_test_acc')
        if final_acc is None and len(test_acc) > 0:
            final_acc = test_acc[-1]
            
        best_acc = max(test_acc) if len(test_acc) > 0 else 0
        best_epoch = epochs[np.argmax(test_acc)] if len(test_acc) > 0 else 0
        
        results.append({
            "Scheduler": scheduler,
            "Final Accuracy (%)": final_acc,
            "Best Accuracy (%)": best_acc,
            "Best Epoch": best_epoch
        })
        
        acc_data[scheduler] = (epochs, test_acc)
        loss_data[scheduler] = (get_clean_metric(data, 'test_loss')[0], test_loss)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Best Accuracy (%)", ascending=False)
        latex_table = df.to_latex(index=False, float_format="%.2f", caption="Centralized Baseline Scheduler Comparison", label="tab:centralized_schedulers")
        with open(os.path.join(TABLES_DIR, "table1_centralized.tex"), "w") as f:
            f.write(latex_table)
        df.to_csv(os.path.join(TABLES_DIR, "table1_centralized.csv"), index=False)

    plt.figure()
    for name, (x, y) in loss_data.items():
        if len(x) > 0:
            plt.plot(x, y, label=name)
    plt.xlabel("Epochs")
    plt.ylabel("Test Loss")
    plt.title("Centralized Test Loss per Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "centralized_loss.pdf"))
    plt.savefig(os.path.join(PLOTS_DIR, "centralized_loss.png"))
    plt.close()

    plt.figure()
    for name, (x, y) in acc_data.items():
        if len(x) > 0:
            plt.plot(x, y, label=name)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Centralized Test Accuracy per Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "centralized_accuracy.pdf"))
    plt.savefig(os.path.join(PLOTS_DIR, "centralized_accuracy.png"))
    plt.close()

def generate_table_2_and_3_plots():
    print("Generating Tables 2, 3 and Plots 2, 3 (Heterogeneity & Local Steps)...")
    scaled_dir = os.path.join(OUTPUT_DIR, "scaled")
    main_dir = os.path.join(OUTPUT_DIR, "main")
    
    if not os.path.exists(scaled_dir):
        print("Skipping Table 2/3: scaled directory not found")
        return

    data_records = []
    
    def process_file(fpath, label, nc, j):
        d = load_json(fpath)
        if not d: return
        rounds, acc = get_clean_metric(d, 'test_acc')
        best_acc = max(acc) if len(acc) > 0 else 0
        final_acc = acc[-1] if len(acc) > 0 else 0
        
        r_80 = next((r for r, a in zip(rounds, acc) if a >= 80.0), None)
        
        data_records.append({
            "Label": label,
            "Nc": nc,
            "J": j,
            "Final Accuracy (%)": final_acc,
            "Best Accuracy (%)": best_acc,
            "Rounds to 80%": r_80 if r_80 else "> Max",
            "rounds": rounds,
            "acc": acc
        })

    # 1. IID Baseline (Nc=IID)
    iid_path = os.path.join(main_dir, "fedavg_iid_metrics.json")
    if os.path.exists(iid_path):
        process_file(iid_path, "IID", "IID", 4)

    # 2. Non-IID Scaled
    pattern = re.compile(r"noniid_nc(\d+)_j(\d+)\.json")
    for filename in os.listdir(scaled_dir):
        match = pattern.match(filename)
        if match:
            nc = int(match.group(1))
            j = int(match.group(2))
            process_file(os.path.join(scaled_dir, filename), f"Nc={nc}, J={j}", nc, j)

    df_all = pd.DataFrame(data_records)
    if df_all.empty: return

    df_nc = df_all[df_all['J'] == 4].copy()
 
    df_nc['Nc_Sort'] = df_nc['Nc'].apply(lambda x: 999 if x == 'IID' else int(x))
    df_nc = df_nc.sort_values("Nc_Sort")
    
    table2_cols = ["Nc", "Final Accuracy (%)", "Best Accuracy (%)", "Rounds to 80%"]
    latex_table2 = df_nc[table2_cols].to_latex(index=False, float_format="%.2f", caption="Impact of Statistical Heterogeneity (Nc)", label="tab:heterogeneity")
    with open(os.path.join(TABLES_DIR, "table2_heterogeneity.tex"), "w") as f:
        f.write(latex_table2)
    df_nc[table2_cols].to_csv(os.path.join(TABLES_DIR, "table2_heterogeneity.csv"), index=False)

    plt.figure()
    for _, row in df_nc.iterrows():
        label = f"Nc={row['Nc']}" if row['Nc'] != "IID" else "IID"
        plt.plot(row['rounds'], row['acc'], label=label)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Impact of Heterogeneity (Nc) on Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "heterogeneity.pdf"))
    plt.savefig(os.path.join(PLOTS_DIR, "heterogeneity.png"))
    plt.close()

    target_nc = 10
    df_j = df_all[df_all['Nc'] == target_nc].copy()
    
    if df_j.empty:
        target_nc = 1
        df_j = df_all[df_all['Nc'] == target_nc].copy()
        
    if not df_j.empty:
        df_j = df_j.sort_values("J")
        table3_cols = ["J", "Final Accuracy (%)", "Best Accuracy (%)", "Rounds to 80%"]
        latex_table3 = df_j[table3_cols].to_latex(index=False, float_format="%.2f", caption=f"Impact of Local Steps (J) with Nc={target_nc}", label="tab:local_steps")
        with open(os.path.join(TABLES_DIR, "table3_local_steps.tex"), "w") as f:
            f.write(latex_table3)
        df_j[table3_cols].to_csv(os.path.join(TABLES_DIR, "table3_local_steps.csv"), index=False)

        plt.figure()
        for _, row in df_j.iterrows():
            plt.plot(row['rounds'], row['acc'], label=f"J={row['J']}")
        plt.xlabel("Communication Rounds")
        plt.ylabel("Test Accuracy (%)")
        plt.title(f"Impact of Local Steps (J) with Nc={target_nc}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOTS_DIR, "local_steps.pdf"))
        plt.savefig(os.path.join(PLOTS_DIR, "local_steps.png"))
        plt.close()

def generate_table_4_and_plot():
    print("Generating Table 4 and Plot 4 (Sparse Ablation)...")
    source_dir = os.path.join(OUTPUT_DIR, "sparse_ablation")
    if not os.path.exists(source_dir):
        print("Skipping Table 4: sparse_ablation not found")
        return

    records = []
    
    
    for filename in os.listdir(source_dir):
        if not filename.endswith(".json"): continue
        d = load_json(os.path.join(source_dir, filename))
        if not d: continue
        
        rounds, acc = get_clean_metric(d, 'test_acc')
        best_acc = max(acc) if len(acc) > 0 else 0
        
        param_type = "Unknown"
        param_val = 0
        
        if "calib" in filename:
            param_type = "Calibration Rounds"
            param_val = int(filename.replace("ablation_calib", "").replace(".json", ""))
        elif "sparsity" in filename:
            param_type = "Sparsity (%)"
            param_val = int(filename.replace("ablation_sparsity", "").replace(".json", ""))
            
        records.append({
            "Type": param_type,
            "Value": param_val,
            "Best Accuracy (%)": best_acc,
            "rounds": rounds,
            "acc": acc
        })
        
    df = pd.DataFrame(records)
    if df.empty: return


    df_sorted = df.sort_values(["Type", "Value"])
    cols = ["Type", "Value", "Best Accuracy (%)"]
    latex_table4 = df_sorted[cols].to_latex(index=False, float_format="%.2f", caption="Sparse Ablation: Sparsity Levels and Calibration Rounds", label="tab:sparse_ablation")
    with open(os.path.join(TABLES_DIR, "table4_sparse_ablation.tex"), "w") as f:
        f.write(latex_table4)
    df_sorted[cols].to_csv(os.path.join(TABLES_DIR, "table4_sparse_ablation.csv"), index=False)

    df_sparsity = df[df['Type'] == "Sparsity (%)"].sort_values("Value")
    
    plt.figure()
    
    
    if not df_sparsity.empty:
        plt.plot(df_sparsity['Value'], df_sparsity['Best Accuracy (%)'], marker='o', label='Sparsity Impact')
        
    plt.xlabel("Sparsity Ratio (%)")
    plt.ylabel("Best Test Accuracy (%)")
    plt.title("Impact of Sparsity Ratio on Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "sparsity_ablation.pdf"))
    plt.savefig(os.path.join(PLOTS_DIR, "sparsity_ablation.png"))
    plt.close()


def generate_table_5_and_plot():
    print("Generating Table 5 and Plot 5 (Masking Rules)...")
    source_dir = os.path.join(OUTPUT_DIR, "sparse_noniid")
    if not os.path.exists(source_dir):
        print("Skipping Table 5: sparse_noniid not found")
        return

    records = []

    pattern = re.compile(r"sparse_nc(\d+)_j(\d+)_(.+)\.json")
    
    for filename in os.listdir(source_dir):
        match = pattern.match(filename)
        if match:
            nc = int(match.group(1))
            j = int(match.group(2))
            rule = match.group(3).replace("_", " ").title()
            
            d = load_json(os.path.join(source_dir, filename))
            if not d: continue
            
            rounds, acc = get_clean_metric(d, 'test_acc')
            best_acc = max(acc) if len(acc) > 0 else 0
            
            records.append({
                "Nc": nc,
                "J": j,
                "Masking Rule": rule,
                "Best Accuracy (%)": best_acc
            })
            
    df = pd.DataFrame(records)
    if df.empty: return
    
 
    df_sorted = df.sort_values(["Nc", "J", "Masking Rule"])
    latex_table5 = df_sorted.to_latex(index=False, float_format="%.2f", caption="Comparison of Masking Rules in Non-IID", label="tab:masking_rules")
    with open(os.path.join(TABLES_DIR, "table5_masking_rules.tex"), "w") as f:
        f.write(latex_table5)
    df_sorted.to_csv(os.path.join(TABLES_DIR, "table5_masking_rules.csv"), index=False)
    
    target_nc = 5
    target_j = 4
    df_plot = df[(df['Nc'] == target_nc) & (df['J'] == target_j)].copy()
    
    if df_plot.empty:

        if not df.empty:
            df_plot = df.iloc[[0]] 
            
    if not df_plot.empty:
        plt.figure()
        sns.barplot(data=df_plot, x="Masking Rule", y="Best Accuracy (%)")
        plt.xticks(rotation=45)
        plt.ylim(bottom=0) 
        plt.title(f"Masking Rule Comparison (Nc={target_nc}, J={target_j})")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "masking_comparison.pdf"))
        plt.savefig(os.path.join(PLOTS_DIR, "masking_comparison.png"))
        plt.close()

if __name__ == "__main__":
    generate_table_1_and_plots()
    generate_table_2_and_3_plots()
    generate_table_4_and_plot()
    generate_table_5_and_plot()
    print("Generation complete. Check 'output/tables' and 'output/plots'.")
