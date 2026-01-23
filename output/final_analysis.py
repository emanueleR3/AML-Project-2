import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'central': '#2ecc71',
    'fedavg_iid': '#3498db',
    'noniid': '#e74c3c',
    'sparse_ls': '#9b59b6',
    'sparse_rnd': '#95a5a6',
    'ablation': '#f39c12',
}

SCRIPT_DIR = Path(__file__).parent
MAIN_DIR = SCRIPT_DIR / "main"
SCALED_DIR = SCRIPT_DIR / "scaled"          
SPARSE_DIR = SCRIPT_DIR / "sparse_ablation"   
SPARSE_IID_DIR = SCRIPT_DIR / "sparse_iid"    
SPARSE_NONIID_DIR = SCRIPT_DIR / "sparse_noniid"  
SCHEDULER_DIR = SCRIPT_DIR / "scheduler-sweep"  
FIGURES_DIR = SCRIPT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_central_baseline(directory: Path) -> Optional[dict]:
    """Load central baseline metrics."""
    filepath = directory / "central_baseline_metrics.json"
    return load_json(filepath) if filepath.exists() else None


def load_fedavg_iid(directory: Path) -> Optional[dict]:
    """Load FedAvg IID metrics."""
    filepath = directory / "fedavg_iid_metrics.json"
    return load_json(filepath) if filepath.exists() else None


def load_scaled_noniid_experiments(directory: Path) -> Dict[Tuple[int, int], dict]:
    """
    Load Non-IID experiments with scaled rounds from output/scaled/.
    Files are named: noniid_scaled_nc{nc}_j{j}.json
    """
    results = {}
    if not directory.exists():
        return results
    
    for filepath in directory.glob("noniid_nc*.json"):
        filename = filepath.stem  # noniid_nc{nc}_j{j}
        try:
            parts = filename.split('_')
            nc = int(parts[1].replace('nc', ''))
            j = int(parts[2].replace('j', ''))
            results[(nc, j)] = load_json(filepath)
        except (IndexError, ValueError) as e:
            print(f"  Warning: Could not parse {filename}: {e}")
    
    return results


def load_sparse_experiments(directory: Path, iid_directory: Path = None) -> Dict[str, dict]:
    results = {}
    if not directory.exists():
        return results
    
    for filepath in directory.glob("final_*.json"):
        name = filepath.stem
        results[name] = load_json(filepath)
    
    if iid_directory and iid_directory.exists():
        metrics_file = iid_directory / "sparse_iid_metrics.json"
        if metrics_file.exists():
            results["sparse_iid_metrics"] = load_json(metrics_file)

    return results


def load_ablation_results(directory: Path) -> Dict[str, dict]:
    """Load ablation study results from output/output:sparse/."""
    results = {
        'calibration': {},
        'sparsity': {},
        'summary': None
    }
    
    if not directory.exists():
        return results
    
    for filepath in directory.glob("ablation_calib*.json"):
        num = int(filepath.stem.replace('ablation_calib', ''))
        results['calibration'][num] = load_json(filepath)
    
    for filepath in directory.glob("ablation_sparsity*.json"):
        pct = int(filepath.stem.replace('ablation_sparsity', ''))
        results['sparsity'][pct] = load_json(filepath)
    
    summary_path = directory / "complete_summary.json"
    if summary_path.exists():
        results['summary'] = load_json(summary_path)
    
    return results


def load_scheduler_sweep(directory: Path) -> Optional[dict]:
    """Load scheduler sweep results from output/scheduler-sweep/."""
    summary_path = directory / "summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    return None


def extract_final_test_acc(metrics: dict) -> Optional[float]:
    """Extract final test accuracy from metrics."""
    if 'test_acc' in metrics:
        accs = [a for a in metrics['test_acc'] if a is not None and not (isinstance(a, float) and np.isnan(a))]
        if accs:
            return accs[-1]
    if 'val_acc' in metrics:
        accs = [a for a in metrics['val_acc'] if a is not None and not (isinstance(a, float) and np.isnan(a))]
        if accs:
            return accs[-1]
    return None

def plot_central_baseline(data: dict):
    # Combined Plot: Training Loss and Training Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Subplot 1: Training Loss
    ax1 = axes[0]
    epochs = range(1, len(data['train_loss']) + 1)
    ax1.plot(epochs, data['train_loss'], color=COLORS['central'], linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Central Baseline - Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Training Accuracy
    ax2 = axes[1]
    # Ensure epochs match training accuracy data length
    acc_epochs = range(1, len(data['train_acc']) + 1)
    ax2.plot(acc_epochs, data['train_acc'], color=COLORS['central'], linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy (%)')
    ax2.set_title('Central Baseline - Training Accuracy')
    
    best_acc = max(data['train_acc'])
    ax2.axhline(y=best_acc, color=COLORS['central'], linestyle='--', alpha=0.5)
    ax2.annotate(f'Best: {best_acc:.2f}%', xy=(acc_epochs[-1], best_acc), 
                 xytext=(-50, 5), textcoords='offset points', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'central_baseline_curves.png')
    plt.savefig(FIGURES_DIR / 'central_baseline_curves.pdf')
    plt.close()
    
    print("✓ Saved: central_baseline_curves.png/pdf (Training Loss + Training Accuracy)")


def plot_fedavg_iid(data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    rounds = data['round']
    
    ax1 = axes[0]
    ax1.plot(rounds, data['train_loss'], color=COLORS['fedavg_iid'], linewidth=2)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('FedAvg IID - Training Loss')
    
    ax2 = axes[1]
    acc_key = 'test_acc' if 'test_acc' in data else 'val_acc'
    acc = np.array(data[acc_key])
    rounds_arr = np.array(rounds)
    valid = ~np.isnan(acc)
    
    ax2.plot(rounds_arr[valid], acc[valid], color=COLORS['fedavg_iid'], linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('FedAvg IID - Test Accuracy')
    
    final_acc = acc[valid][-1]
    ax2.annotate(f'Final: {final_acc:.1f}%', xy=(rounds_arr[valid][-1], final_acc),
                 xytext=(-50, 5), textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fedavg_iid_convergence.png')
    plt.savefig(FIGURES_DIR / 'fedavg_iid_convergence.pdf')
    plt.close()
    print("✓ Saved: fedavg_iid_convergence.png/pdf")


def plot_noniid_scaled_heatmap(noniid_results: Dict[Tuple[int, int], dict]):
    if not noniid_results:
        print("⚠ No scaled non-IID results found, skipping heatmap")
        return None
    
    ncs = sorted(set(k[0] for k in noniid_results.keys()))
    js = sorted(set(k[1] for k in noniid_results.keys()))
    
    final_acc_matrix = np.full((len(ncs), len(js)), np.nan)
    
    for (nc, j), data in noniid_results.items():
        nc_idx = ncs.index(nc)
        j_idx = js.index(j)
        final_acc = extract_final_test_acc(data)
        if final_acc is not None:
            final_acc_matrix[nc_idx, j_idx] = final_acc
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    j_labels = [f'J={j}\n(R={400//j})' for j in js]
    
    sns.heatmap(final_acc_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=j_labels,
                yticklabels=[f'Nc={nc}' for nc in ncs],
                ax=ax, cbar_kws={'label': 'Test Accuracy (%)'})
    ax.set_title('Non-IID Results with Scaled Rounds\n(J × Rounds = 400)')
    ax.set_xlabel('Local Steps (J) with Scaled Rounds (R)')
    ax.set_ylabel('Classes per Client (Nc)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'noniid_scaled_heatmap.png')
    plt.savefig(FIGURES_DIR / 'noniid_scaled_heatmap.pdf')
    plt.close()
    print("✓ Saved: noniid_scaled_heatmap.png/pdf")
    
    rows = []
    for nc in ncs:
        for j in js:
            if (nc, j) in noniid_results:
                data = noniid_results[(nc, j)]
                final_acc = extract_final_test_acc(data)
                scaled_rounds = 400 // j
                rows.append({
                    'Nc': nc, 'J': j, 'Rounds': scaled_rounds, 
                    'Test Acc (%)': round(final_acc, 2) if final_acc else None
                })
    return pd.DataFrame(rows)


def plot_ablation_studies(ablation_results: Dict[str, dict]):
    has_calib = bool(ablation_results.get('calibration'))
    has_sparsity = bool(ablation_results.get('sparsity'))
    summary = ablation_results.get('summary')
    
    if not has_calib and not has_sparsity and not summary:
        print("⚠ No ablation results found, skipping")
        return
    
    if summary:
        calib_data = summary.get('ablation_calibration', {})
        sparsity_data = summary.get('ablation_sparsity', {})
    else:
        calib_data = {}
        sparsity_data = {}
        
        for num, data in ablation_results.get('calibration', {}).items():
            calib_data[str(num)] = extract_final_test_acc(data)
        
        for pct, data in ablation_results.get('sparsity', {}).items():
            sparsity_data[str(pct/100)] = extract_final_test_acc(data)
    
    n_plots = int(bool(calib_data)) + int(bool(sparsity_data))
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    if calib_data:
        ax = axes[plot_idx]
        rounds = sorted([int(k) for k in calib_data.keys()])
        accs = [calib_data[str(r)] for r in rounds]
        
        bars = ax.bar(range(len(rounds)), accs, color='steelblue')
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds)
        ax.set_xlabel('Number of Calibration Rounds')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Effect of Calibration Rounds')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{acc:.2f}', ha='center', fontsize=9)
        
        plot_idx += 1
    
    if sparsity_data:
        ax = axes[plot_idx]
        ratios = sorted([float(k) for k in sparsity_data.keys()])
        accs = [sparsity_data[str(r)] for r in ratios]
        
        ax.plot(ratios, accs, 'o-', color='darkorange', linewidth=2, markersize=10)
        ax.set_xlabel('Sparsity Ratio')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Effect of Sparsity Ratio')
        ax.grid(True, alpha=0.3)
        
        for r, acc in zip(ratios, accs):
            ax.annotate(f'{acc:.2f}', (r, acc), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ablation_studies.png')
    plt.savefig(FIGURES_DIR / 'ablation_studies.pdf')
    plt.close()
    print("✓ Saved: ablation_studies.png/pdf")


def plot_sparse_comparison(sparse_results: Dict[str, dict], fedavg_iid: dict = None):
    if not sparse_results:
        print("⚠ No sparse results found, skipping comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = {
        'final_iid_ls': 'IID + Least Sensitive',
        'final_noniid_ls': 'Non-IID + Least Sensitive',
        'final_noniid_random': 'Non-IID + Random',
        'sparse_iid_metrics': 'Sparse IID (50% Sparsity)',
    }
    
    colors = {
        'final_iid_ls': COLORS['sparse_ls'],
        'final_noniid_ls': COLORS['noniid'],
        'final_noniid_random': COLORS['sparse_rnd'],
        'sparse_iid_metrics': COLORS['sparse_ls'],
    }
    
    ax1 = axes[0]
    
    if fedavg_iid:
        acc_key = 'test_acc' if 'test_acc' in fedavg_iid else 'val_acc'
        acc = np.array(fedavg_iid[acc_key])
        rounds = np.array(fedavg_iid['round'])
        valid = ~np.isnan(acc)
        if valid.any():
            ax1.plot(rounds[valid], acc[valid], color=COLORS['fedavg_iid'], 
                     linewidth=2, linestyle='--', label='Dense FedAvg IID', alpha=0.8)
    
    for name, data in sparse_results.items():
        if name == 'sparse_iid_metrics':
             continue
        if 'round' in data and 'test_acc' in data:
            label = labels.get(name, name)
            color = colors.get(name, 'gray')
            ax1.plot(data['round'], data['test_acc'], color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Sparse FedAvg - Convergence')
    ax1.legend(loc='lower right', fontsize=9)
    
    ax2 = axes[1]
    
    names = list(sparse_results.keys())
    accs = [extract_final_test_acc(sparse_results[n]) or 0 for n in names]
    bar_labels = [labels.get(n, n).replace(' + ', '\n') for n in names]
    bar_colors = [colors.get(n, 'gray') for n in names]
    
    bars = ax2.bar(range(len(names)), accs, color=bar_colors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(bar_labels, fontsize=9)
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Sparse FedAvg - Final Test Accuracy')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{acc:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    if fedavg_iid:
        dense_acc = extract_final_test_acc(fedavg_iid)
        if dense_acc:
            ax2.axhline(y=dense_acc, color=COLORS['fedavg_iid'], linestyle='--', 
                        alpha=0.8, label=f'Dense IID ({dense_acc:.1f}%)')
            ax2.legend()
    
    plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'sparse_fedavg_comparison.png')
    # plt.savefig(FIGURES_DIR / 'sparse_fedavg_comparison.pdf')
    plt.close()
    # print("✓ Saved: sparse_fedavg_comparison.png/pdf")


def load_sparse_noniid_results(directory: Path) -> Dict[str, dict]:
    results = {}
    if not directory.exists():
        return results
    
    for filepath in directory.glob("sparse_nc*.json"):
        name = filepath.stem
        results[name] = load_json(filepath)
    
    return results


def plot_masking_rule_comparison(sparse_noniid_results: Dict[str, dict], noniid_results: Dict):
    if not sparse_noniid_results:
        print("⚠ No sparse non-IID results found, skipping masking comparison")
        return
    
    configs = {}
    for name, data in sparse_noniid_results.items():
        parts = name.replace('sparse_', '').split('_')
        try:
            nc = int(parts[0].replace('nc', ''))
            j = int(parts[1].replace('j', ''))
            rule = '_'.join(parts[2:])
            
            key = (nc, j)
            if key not in configs:
                configs[key] = {}
            
            final_acc = extract_final_test_acc(data)
            if final_acc:
                configs[key][rule] = final_acc
        except (IndexError, ValueError):
            print(f"  Warning: Could not parse {name}")
            continue
    
    if not configs:
        print("⚠ No valid sparse non-IID results parsed")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    config_labels = [f'Nc={nc}, J={j}' for (nc, j) in sorted(configs.keys())]
    all_rules = set()
    for rule_accs in configs.values():
        all_rules.update(rule_accs.keys())
    all_rules = sorted(all_rules)
    
    rule_colors = {
        'least_sensitive': '#3498db',
        'most_sensitive': '#e74c3c',
        'lowest_magnitude': '#2ecc71',
        'highest_magnitude': '#9b59b6',
        'random': '#95a5a6'
    }
    
    x = np.arange(len(config_labels))
    width = 0.15
    n_rules = len(all_rules)
    
    for i, rule in enumerate(all_rules):
        offsets = x + (i - n_rules/2 + 0.5) * width
        accs = []
        for key in sorted(configs.keys()):
            accs.append(configs[key].get(rule, 0))
        
        color = rule_colors.get(rule, 'gray')
        label = rule.replace('_', ' ').title()
        bars = ax.bar(offsets, accs, width, label=label, color=color)
        
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{acc:.1f}', ha='center', fontsize=8, rotation=45)
    
    for idx, (nc, j) in enumerate(sorted(configs.keys())):
        if (nc, j) in noniid_results:
            dense_acc = extract_final_test_acc(noniid_results[(nc, j)])
            if dense_acc:
                ax.hlines(dense_acc, idx - 0.4, idx + 0.4, colors='black', 
                         linestyles='--', linewidth=2, alpha=0.7)
                ax.text(idx + 0.45, dense_acc, f'Dense: {dense_acc:.1f}%', 
                       fontsize=8, va='center')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Sparse Non-IID: Masking Rule Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'masking_rule_comparison.png')
    plt.savefig(FIGURES_DIR / 'masking_rule_comparison.pdf')
    plt.close()
    print("✓ Saved: masking_rule_comparison.png/pdf")


def generate_summary(central, fedavg_iid, noniid_results, sparse_results, ablation) -> pd.DataFrame:
    rows = []
    
    if central:
        best_val = max(central['val_acc'])
        rows.append({
            'Experiment': 'Central Baseline',
            'Final Acc (%)': round(central['val_acc'][-1], 2),
            'Best Acc (%)': round(best_val, 2),
            'Category': 'Baseline'
        })
    
    if fedavg_iid:
        final_acc = extract_final_test_acc(fedavg_iid)
        rows.append({
            'Experiment': 'FedAvg IID',
            'Final Acc (%)': round(final_acc, 2) if final_acc else None,
            'Best Acc (%)': round(max(fedavg_iid.get('test_acc', fedavg_iid.get('val_acc', [0]))), 2),
            'Category': 'Dense FL'
        })
    
    for (nc, j), data in sorted(noniid_results.items()):
        final_acc = extract_final_test_acc(data)
        if final_acc:
            rows.append({
                'Experiment': f'Non-IID (Nc={nc}, J={j}, R={400//j})',
                'Final Acc (%)': round(final_acc, 2),
                'Best Acc (%)': round(data.get('best_val_acc', final_acc), 2),
                'Category': 'Non-IID (Scaled)'
            })
    
    for name, data in sparse_results.items():
        final_acc = extract_final_test_acc(data)
        if final_acc:
            label = name.replace('final_', '').replace('_', ' ').title()
            rows.append({
                'Experiment': f'Sparse {label}',
                'Final Acc (%)': round(final_acc, 2),
                'Best Acc (%)': round(max([x for x in data.get('test_acc', [final_acc]) if x is not None]), 2),
                'Category': 'Sparse FL'
            })
    
    return pd.DataFrame(rows)


def generate_latex_tables(summary_df: pd.DataFrame, noniid_df: Optional[pd.DataFrame]):
    summary_df.to_csv(SCRIPT_DIR / 'summary_results.csv', index=False)
    print("✓ Saved: summary_results.csv")
    
    latex = summary_df.to_latex(index=False, escape=False, column_format='lccc')
    with open(SCRIPT_DIR / 'summary_table.tex', 'w') as f:
        f.write("% Auto-generated\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Experiment Results Summary}\n\\label{tab:summary}\n")
        f.write(latex)
        f.write("\\end{table}\n")
    print("✓ Saved: summary_table.tex")
    
    if noniid_df is not None and not noniid_df.empty:
        latex_noniid = noniid_df.to_latex(index=False, escape=False, column_format='cccc')
        with open(SCRIPT_DIR / 'noniid_scaled_table.tex', 'w') as f:
            f.write("% Auto-generated - Non-IID with Scaled Rounds\n")
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\caption{Non-IID Results (J × Rounds = 400)}\n\\label{tab:noniid_scaled}\n")
            f.write(latex_noniid)
            f.write("\\end{table}\n")
        print("✓ Saved: noniid_scaled_table.tex")


def main():
    print("=" * 60)
    print("Federated Learning Experiment Analysis")
    print("Using: output/scaled/ and output/output:sparse/")
    print("=" * 60)
    print()
    
    print("Loading data...")
    
    central = load_central_baseline(MAIN_DIR)
    fedavg_iid = load_fedavg_iid(MAIN_DIR)
    noniid_results = load_scaled_noniid_experiments(SCALED_DIR)
    sparse_results = load_sparse_experiments(SPARSE_DIR, SPARSE_IID_DIR)
    ablation = load_ablation_results(SPARSE_DIR)
    scheduler = load_scheduler_sweep(SCHEDULER_DIR)
    
    print(f"  ✓ Central baseline: {'Found' if central else 'Not found'}")
    print(f"  ✓ FedAvg IID: {'Found' if fedavg_iid else 'Not found'}")
    print(f"  ✓ Scaled Non-IID: {len(noniid_results)} experiments")
    print(f"  ✓ Sparse experiments: {len(sparse_results)} results")
    print(f"  ✓ Ablation (calibration): {len(ablation.get('calibration', {}))} configs")
    print(f"  ✓ Ablation (sparsity): {len(ablation.get('sparsity', {}))} configs")
    print(f"  ✓ Scheduler sweep: {'Found' if scheduler else 'Not found'}")
    
    sparse_noniid = load_sparse_noniid_results(SPARSE_NONIID_DIR)
    print(f"  ✓ Sparse Non-IID (masking): {len(sparse_noniid)} experiments")
    if ablation.get('summary'):
        print(f"  ✓ Complete summary: Found")
    print()
    
    print("Generating visualizations...")
    print("-" * 40)
    
    if central:
        plot_central_baseline(central)
    
    if fedavg_iid:
        plot_fedavg_iid(fedavg_iid)
    
    noniid_df = plot_noniid_scaled_heatmap(noniid_results)
    
    plot_ablation_studies(ablation)
    
    plot_sparse_comparison(sparse_results, fedavg_iid)
    
    plot_masking_rule_comparison(sparse_noniid, noniid_results)
    
    print()
    
    print("Generating summary...")
    print("-" * 40)
    
    summary_df = generate_summary(central, fedavg_iid, noniid_results, sparse_results, ablation)
    # generate_latex_tables(summary_df, noniid_df)
    
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    
    if ablation.get('summary'):
        summary = ablation['summary']
        print("=" * 60)
        print("KEY FINDINGS")
        print("=" * 60)
        
        opt = summary.get('optimal_params', {})
        print(f"\n• Optimal Parameters:")
        print(f"  → Calibration Rounds: {opt.get('calibration_rounds', 'N/A')}")
        print(f"  → Sparsity Ratio: {opt.get('sparsity_ratio', 'N/A')}")
        
        final = summary.get('final_experiments', {})
        if final:
            print(f"\n• Final Sparse Experiments:")
            for exp, acc in final.items():
                print(f"  → {exp}: {acc:.2f}%")
            
            if 'noniid_ls' in final and 'noniid_random' in final:
                diff = final['noniid_ls'] - final['noniid_random']
                print(f"\n  ★ Least Sensitive vs Random: +{diff:.2f} pp")
    
    if scheduler:
        print(f"\n• Scheduler Comparison:")
        print(f"  → Best Scheduler: {scheduler['best_scheduler'].upper()}")
        print(f"  → Best Val Acc: {scheduler['best_val_acc']:.2f}%")
        print(f"  → Best Test Acc: {scheduler['best_test_acc']:.2f}%")
        print(f"\n  All schedulers:")
        for name, accs in scheduler.get('all_results', {}).items():
            print(f"    {name}: val={accs['val']:.2f}%, test={accs['test']:.2f}%")
    
    print()
    print("✓ Analysis complete! Check 'figures/' directory.")


if __name__ == "__main__":
    main()
