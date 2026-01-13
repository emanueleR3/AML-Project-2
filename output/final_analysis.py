#!/usr/bin/env python3
"""
Federated Learning Experiment Analysis Script
=============================================
This script analyzes FL experiment results and generates publication-ready
figures and tables for the final report.

Outputs:
- Central baseline training curves
- FedAvg IID convergence (100 vs 300 rounds comparison)
- Non-IID Nc×J heatmap (with scaled rounds support)
- Sparse FedAvg comparison across ALL mask rules
- Ablation studies (calibration rounds, sparsity ratio)
- Summary tables (CSV + LaTeX)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================
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

# Color palette for consistent styling
COLORS = {
    'central': '#2ecc71',       # Green
    'fedavg_iid': '#3498db',    # Blue
    'noniid': '#e74c3c',        # Red
    'sparse_ls': '#9b59b6',     # Purple (least sensitive)
    'sparse_ms': '#e67e22',     # Orange (most sensitive)
    'sparse_lm': '#1abc9c',     # Teal (lowest magnitude)
    'sparse_hm': '#f1c40f',     # Yellow (highest magnitude)
    'sparse_rnd': '#95a5a6',    # Gray (random)
    '100_rounds': '#3498db',    # Blue
    '300_rounds': '#e74c3c',    # Red
}

# Paths
SCRIPT_DIR = Path(__file__).parent
MAIN_DIR = SCRIPT_DIR / "main"
EXTENDED_DIR = SCRIPT_DIR / "extended"
SPARSE_DIR = SCRIPT_DIR / "sparse"
FIGURES_DIR = SCRIPT_DIR / "figures"

# Ensure figures directory exists
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================================
# Data Loading Functions
# ============================================================================
def load_json(filepath: Path) -> dict:
    """Load JSON file and return data dict."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_central_baseline(directory: Path) -> dict:
    """Load central baseline metrics."""
    filepath = directory / "central_baseline_metrics.json"
    if filepath.exists():
        return load_json(filepath)
    return None


def load_fedavg_iid(directory: Path) -> dict:
    """Load FedAvg IID metrics."""
    filepath = directory / "fedavg_iid_metrics.json"
    if filepath.exists():
        return load_json(filepath)
    return None


def load_noniid_experiments(directory: Path) -> Dict[Tuple[int, int], dict]:
    """
    Load all non-IID experiment results.
    Returns dict with (nc, j) tuples as keys.
    """
    results = {}
    pattern = str(directory / "noniid_nc*_j*.json")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        # Parse nc and j from filename: noniid_nc{nc}_j{j}.json
        parts = filename.replace('.json', '').split('_')
        nc = int(parts[1].replace('nc', ''))
        j = int(parts[2].replace('j', ''))
        results[(nc, j)] = load_json(Path(filepath))
    return results


def load_sparse_experiments(directory: Path) -> Dict[str, dict]:
    """
    Load sparse FedAvg experiment results.
    Returns dict with experiment name as key.
    """
    results = {}
    
    # Check both sparse directory and legacy masks directory
    for check_dir in [directory, SCRIPT_DIR / "masks"]:
        if not check_dir.exists():
            continue
            
        # Load exp_*.json files
        for filepath in check_dir.glob("exp_*.json"):
            name = filepath.stem
            results[name] = load_json(filepath)
    
    return results


def load_ablation_results(directory: Path) -> Dict[str, dict]:
    """Load ablation study results (calibration rounds, sparsity)."""
    results = {
        'calibration': {},
        'sparsity': {},
        'summary': None
    }
    
    if not directory.exists():
        return results
    
    # Load calibration ablations
    for filepath in directory.glob("ablation_calib*.json"):
        num = int(filepath.stem.replace('ablation_calib', ''))
        results['calibration'][num] = load_json(filepath)
    
    # Load sparsity ablations
    for filepath in directory.glob("ablation_sparsity*.json"):
        pct = int(filepath.stem.replace('ablation_sparsity', ''))
        results['sparsity'][pct] = load_json(filepath)
    
    # Load summary if exists
    summary_path = directory / "complete_summary.json"
    if summary_path.exists():
        results['summary'] = load_json(summary_path)
    
    return results


def extract_final_and_best_acc(metrics: dict, acc_key: str = 'test_acc') -> Tuple[float, float]:
    """Extract final and best accuracy from metrics."""
    if 'best_val_acc' in metrics:
        best_acc = metrics['best_val_acc']
    else:
        best_acc = None
    
    final_acc = None
    
    for key in [acc_key, 'test_acc', 'val_acc']:
        if key in metrics:
            accs = [a for a in metrics[key] if a is not None and not (isinstance(a, float) and np.isnan(a))]
            if accs:
                final_acc = accs[-1]
                if best_acc is None:
                    best_acc = max(accs)
                break
    
    return final_acc, best_acc


# ============================================================================
# Visualization Functions
# ============================================================================
def plot_central_baseline(main_data: dict, extended_data: dict = None):
    """Plot central baseline training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Training Loss
    ax1 = axes[0]
    epochs_main = range(1, len(main_data['train_loss']) + 1)
    ax1.plot(epochs_main, main_data['train_loss'], 
             color=COLORS['100_rounds'], linewidth=2, label='20 Epochs', marker='o', markersize=4)
    
    if extended_data:
        epochs_ext = range(1, len(extended_data['train_loss']) + 1)
        ax1.plot(epochs_ext, extended_data['train_loss'], 
                 color=COLORS['300_rounds'], linewidth=2, label='30 Epochs', marker='s', markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Central Baseline - Training Loss')
    ax1.legend()
    
    # Validation Accuracy
    ax2 = axes[1]
    n_val_main = len(main_data['val_acc'])
    val_epochs_main = np.linspace(1, len(main_data['train_loss']), n_val_main).astype(int)
    ax2.plot(val_epochs_main, main_data['val_acc'], 
             color=COLORS['100_rounds'], linewidth=2, label='20 Epochs', marker='o', markersize=6)
    
    if extended_data:
        n_val_ext = len(extended_data['val_acc'])
        val_epochs_ext = np.linspace(1, len(extended_data['train_loss']), n_val_ext).astype(int)
        ax2.plot(val_epochs_ext, extended_data['val_acc'], 
                 color=COLORS['300_rounds'], linewidth=2, label='30 Epochs', marker='s', markersize=6)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Central Baseline - Validation Accuracy')
    ax2.legend()
    
    best_acc_main = max(main_data['val_acc'])
    ax2.axhline(y=best_acc_main, color=COLORS['100_rounds'], linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'central_baseline_curves.png')
    plt.savefig(FIGURES_DIR / 'central_baseline_curves.pdf')
    plt.close()
    print("✓ Saved: central_baseline_curves.png/pdf")


def plot_fedavg_iid_comparison(main_data: dict, extended_data: dict = None):
    """Plot FedAvg IID convergence comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    ax1 = axes[0]
    rounds_main = main_data['round']
    ax1.plot(rounds_main, main_data['train_loss'], 
             color=COLORS['100_rounds'], linewidth=2, label='100 Rounds')
    
    if extended_data:
        rounds_ext = extended_data['round']
        ax1.plot(rounds_ext, extended_data['train_loss'], 
                 color=COLORS['300_rounds'], linewidth=2, label='300 Rounds')
        ax1.axvline(x=100, color='gray', linestyle=':', alpha=0.7)
    
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('FedAvg IID - Training Loss')
    ax1.legend()
    
    ax2 = axes[1]
    acc_key = 'test_acc' if 'test_acc' in main_data else 'val_acc'
    
    acc_main = np.array(main_data[acc_key])
    rounds_arr_main = np.array(rounds_main)
    valid_mask_main = ~np.isnan(acc_main)
    
    ax2.plot(rounds_arr_main[valid_mask_main], acc_main[valid_mask_main], 
             color=COLORS['100_rounds'], linewidth=2, marker='o', markersize=4,
             label='100 Rounds')
    
    if extended_data:
        acc_ext = np.array(extended_data[acc_key])
        rounds_arr_ext = np.array(extended_data['round'])
        valid_mask_ext = ~np.isnan(acc_ext)
        
        ax2.plot(rounds_arr_ext[valid_mask_ext], acc_ext[valid_mask_ext], 
                 color=COLORS['300_rounds'], linewidth=2, marker='s', markersize=4,
                 label='300 Rounds')
        ax2.axvline(x=100, color='gray', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('FedAvg IID - Test Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fedavg_iid_convergence.png')
    plt.savefig(FIGURES_DIR / 'fedavg_iid_convergence.pdf')
    plt.close()
    print("✓ Saved: fedavg_iid_convergence.png/pdf")


def plot_noniid_heatmap(noniid_results: Dict[Tuple[int, int], dict]):
    """Create heatmap of Non-IID results (Nc × J)."""
    if not noniid_results:
        print("⚠ No non-IID results found, skipping heatmap")
        return None
    
    ncs = sorted(set(k[0] for k in noniid_results.keys()))
    js = sorted(set(k[1] for k in noniid_results.keys()))
    
    final_acc_matrix = np.full((len(ncs), len(js)), np.nan)
    
    for (nc, j), data in noniid_results.items():
        nc_idx = ncs.index(nc)
        j_idx = js.index(j)
        final_acc, _ = extract_final_and_best_acc(data)
        if final_acc is not None:
            final_acc_matrix[nc_idx, j_idx] = final_acc
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(final_acc_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=[f'J={j}' for j in js],
                yticklabels=[f'Nc={nc}' for nc in ncs],
                ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title('Non-IID: Final Test Accuracy (%) - Scaled Rounds')
    ax.set_xlabel('Local Steps (J)')
    ax.set_ylabel('Classes per Client (Nc)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'noniid_heatmap.png')
    plt.savefig(FIGURES_DIR / 'noniid_heatmap.pdf')
    plt.close()
    print("✓ Saved: noniid_heatmap.png/pdf")
    
    return create_noniid_table(noniid_results, ncs, js)


def create_noniid_table(noniid_results: Dict[Tuple[int, int], dict], 
                        ncs: List[int], js: List[int]) -> pd.DataFrame:
    """Create detailed non-IID results table."""
    rows = []
    for nc in ncs:
        for j in js:
            if (nc, j) in noniid_results:
                data = noniid_results[(nc, j)]
                final_acc, best_acc = extract_final_and_best_acc(data)
                # Calculate scaled rounds
                scaled_rounds = 400 // j  # BASE_J * BASE_ROUNDS / j
                rows.append({
                    'Nc': nc,
                    'J': j,
                    'Rounds': scaled_rounds,
                    'Final Test Acc (%)': final_acc,
                })
    
    df = pd.DataFrame(rows)
    return df


def plot_sparse_comparison_all_rules(sparse_results: Dict[str, dict], fedavg_iid_data: dict = None):
    """Plot sparse FedAvg comparison across ALL mask rules."""
    if not sparse_results:
        print("⚠ No sparse FedAvg results found, skipping")
        return
    
    # Color and label mapping for all mask rules
    exp_colors = {
        'exp_iid_ls': COLORS['sparse_ls'],
        'exp_noniid_least_sensitive': COLORS['sparse_ls'],
        'exp_noniid_most_sensitive': COLORS['sparse_ms'],
        'exp_noniid_lowest_magnitude': COLORS['sparse_lm'],
        'exp_noniid_highest_magnitude': COLORS['sparse_hm'],
        'exp_noniid_random': COLORS['sparse_rnd'],
        # Legacy names
        'exp_niid_ls': COLORS['sparse_ls'],
        'exp_niid_rnd': COLORS['sparse_rnd'],
    }
    
    exp_labels = {
        'exp_iid_ls': 'IID + Least Sensitive',
        'exp_noniid_least_sensitive': 'Least Sensitive',
        'exp_noniid_most_sensitive': 'Most Sensitive',
        'exp_noniid_lowest_magnitude': 'Lowest Magnitude',
        'exp_noniid_highest_magnitude': 'Highest Magnitude',
        'exp_noniid_random': 'Random',
        # Legacy
        'exp_niid_ls': 'Non-IID + Least Sensitive',
        'exp_niid_rnd': 'Non-IID + Random',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Convergence curves
    ax1 = axes[0]
    
    if fedavg_iid_data:
        acc = np.array(fedavg_iid_data.get('test_acc', fedavg_iid_data.get('val_acc', [])))
        rounds = np.array(fedavg_iid_data['round'])
        valid_mask = ~np.isnan(acc)
        if valid_mask.any():
            ax1.plot(rounds[valid_mask], acc[valid_mask],
                     color=COLORS['fedavg_iid'], linewidth=2, linestyle='--',
                     label='Dense FedAvg IID', alpha=0.8)
    
    for exp_name, data in sparse_results.items():
        color = exp_colors.get(exp_name, 'gray')
        label = exp_labels.get(exp_name, exp_name)
        acc_key = 'test_acc' if 'test_acc' in data else 'val_acc'
        if acc_key in data and data[acc_key]:
            ax1.plot(data['round'], data[acc_key],
                     color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Sparse FedAvg - Convergence Comparison')
    ax1.legend(loc='lower right', fontsize=9)
    
    # Plot 2: Bar chart comparison
    ax2 = axes[1]
    
    # Filter for non-IID mask rule experiments
    noniid_exps = {k: v for k, v in sparse_results.items() if 'noniid' in k.lower() or 'niid' in k.lower()}
    
    if noniid_exps:
        labels = []
        accs = []
        colors = []
        
        for exp_name, data in noniid_exps.items():
            final_acc, _ = extract_final_and_best_acc(data)
            if final_acc is not None:
                labels.append(exp_labels.get(exp_name, exp_name).replace('Non-IID + ', ''))
                accs.append(final_acc)
                colors.append(exp_colors.get(exp_name, 'gray'))
        
        # Sort by accuracy
        sorted_data = sorted(zip(labels, accs, colors), key=lambda x: x[1], reverse=True)
        labels, accs, colors = zip(*sorted_data) if sorted_data else ([], [], [])
        
        bars = ax2.bar(range(len(labels)), accs, color=colors)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=9)
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Mask Rule Comparison (Non-IID, Nc=1)')
        
        for bar, acc in zip(bars, accs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                     f'{acc:.1f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sparse_fedavg_comparison.png')
    plt.savefig(FIGURES_DIR / 'sparse_fedavg_comparison.pdf')
    plt.close()
    print("✓ Saved: sparse_fedavg_comparison.png/pdf")


def plot_ablation_studies(ablation_results: Dict[str, dict]):
    """Plot ablation study results."""
    has_calib = bool(ablation_results.get('calibration'))
    has_sparsity = bool(ablation_results.get('sparsity'))
    
    if not has_calib and not has_sparsity:
        print("⚠ No ablation results found, skipping")
        return
    
    n_plots = int(has_calib) + int(has_sparsity)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Calibration rounds ablation
    if has_calib:
        ax = axes[plot_idx]
        calib_data = ablation_results['calibration']
        
        rounds = sorted(calib_data.keys())
        accs = []
        for r in rounds:
            final_acc, _ = extract_final_and_best_acc(calib_data[r])
            accs.append(final_acc if final_acc else 0)
        
        ax.bar(range(len(rounds)), accs, color='steelblue')
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds)
        ax.set_xlabel('Number of Calibration Rounds')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Effect of Calibration Rounds')
        ax.grid(axis='y', alpha=0.3)
        
        for i, acc in enumerate(accs):
            ax.text(i, acc + 0.3, f'{acc:.1f}', ha='center', fontsize=9)
        
        plot_idx += 1
    
    # Sparsity ratio ablation
    if has_sparsity:
        ax = axes[plot_idx]
        sparsity_data = ablation_results['sparsity']
        
        ratios = sorted(sparsity_data.keys())
        accs = []
        for r in ratios:
            final_acc, _ = extract_final_and_best_acc(sparsity_data[r])
            accs.append(final_acc if final_acc else 0)
        
        ax.plot([r/100 for r in ratios], accs, 'o-', color='darkorange', linewidth=2, markersize=10)
        ax.set_xlabel('Sparsity Ratio')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Effect of Sparsity Ratio')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ablation_studies.png')
    plt.savefig(FIGURES_DIR / 'ablation_studies.pdf')
    plt.close()
    print("✓ Saved: ablation_studies.png/pdf")


# ============================================================================
# Summary Generation
# ============================================================================
def generate_summary_csv(main_central, extended_central, main_iid, extended_iid,
                         noniid_results, sparse_results, ablation_results) -> pd.DataFrame:
    """Generate comprehensive summary CSV."""
    rows = []
    
    # Central Baseline
    if main_central:
        rows.append({
            'Experiment': 'Central Baseline (20 epochs)',
            'Best Acc (%)': max(main_central['val_acc']),
            'Final Acc (%)': main_central['val_acc'][-1],
            'Rounds/Epochs': len(main_central['train_loss']),
            'Sparsity': '0%',
            'Category': 'Baseline'
        })
    
    if extended_central:
        rows.append({
            'Experiment': 'Central Baseline (30 epochs)',
            'Best Acc (%)': max(extended_central['val_acc']),
            'Final Acc (%)': extended_central['val_acc'][-1],
            'Rounds/Epochs': len(extended_central['train_loss']),
            'Sparsity': '0%',
            'Category': 'Baseline'
        })
    
    # FedAvg IID
    if main_iid:
        final_acc, best_acc = extract_final_and_best_acc(main_iid)
        rows.append({
            'Experiment': 'FedAvg IID (100 rounds)',
            'Best Acc (%)': best_acc,
            'Final Acc (%)': final_acc,
            'Rounds/Epochs': max(main_iid['round']),
            'Sparsity': '0%',
            'Category': 'Dense FL'
        })
    
    if extended_iid:
        final_acc, best_acc = extract_final_and_best_acc(extended_iid)
        rows.append({
            'Experiment': 'FedAvg IID (300 rounds)',
            'Best Acc (%)': best_acc,
            'Final Acc (%)': final_acc,
            'Rounds/Epochs': max(extended_iid['round']),
            'Sparsity': '0%',
            'Category': 'Dense FL'
        })
    
    # Non-IID (scaled rounds)
    for (nc, j), data in sorted(noniid_results.items()):
        final_acc, best_acc = extract_final_and_best_acc(data)
        scaled_rounds = 400 // j
        if final_acc is not None:
            rows.append({
                'Experiment': f'FedAvg Non-IID (Nc={nc}, J={j}, R={scaled_rounds})',
                'Best Acc (%)': best_acc,
                'Final Acc (%)': final_acc,
                'Rounds/Epochs': scaled_rounds,
                'Sparsity': '0%',
                'Category': 'Non-IID'
            })
    
    # Sparse experiments
    exp_labels = {
        'exp_iid_ls': 'Sparse IID (Least Sensitive)',
        'exp_noniid_least_sensitive': 'Sparse Non-IID (Least Sensitive)',
        'exp_noniid_most_sensitive': 'Sparse Non-IID (Most Sensitive)',
        'exp_noniid_lowest_magnitude': 'Sparse Non-IID (Lowest Magnitude)',
        'exp_noniid_highest_magnitude': 'Sparse Non-IID (Highest Magnitude)',
        'exp_noniid_random': 'Sparse Non-IID (Random)',
        'exp_niid_ls': 'Sparse Non-IID (Least Sensitive)',
        'exp_niid_rnd': 'Sparse Non-IID (Random)',
    }
    
    for exp_name, data in sparse_results.items():
        label = exp_labels.get(exp_name, exp_name)
        final_acc, best_acc = extract_final_and_best_acc(data)
        if final_acc is not None:
            rows.append({
                'Experiment': label,
                'Best Acc (%)': best_acc,
                'Final Acc (%)': final_acc,
                'Rounds/Epochs': max(data['round']) if data.get('round') else 100,
                'Sparsity': '80%',
                'Category': 'Sparse FL'
            })
    
    df = pd.DataFrame(rows)
    df = df.round(2)
    return df


def generate_latex_tables(summary_df: pd.DataFrame, noniid_df: pd.DataFrame):
    """Generate LaTeX formatted tables."""
    latex_summary = summary_df.to_latex(index=False, escape=False, 
                                         column_format='l' + 'c' * (len(summary_df.columns) - 1))
    
    with open(SCRIPT_DIR / 'summary_table.tex', 'w') as f:
        f.write("% Auto-generated LaTeX table - Summary Results\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Summary of Experiment Results}\n")
        f.write("\\label{tab:summary}\n")
        f.write(latex_summary)
        f.write("\\end{table}\n")
    
    print("✓ Saved: summary_table.tex")
    
    if noniid_df is not None and not noniid_df.empty:
        latex_noniid = noniid_df.to_latex(index=False, escape=False, column_format='cccc')
        
        with open(SCRIPT_DIR / 'noniid_table.tex', 'w') as f:
            f.write("% Auto-generated LaTeX table - Non-IID Results (Scaled Rounds)\n")
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\caption{Non-IID Results with Scaled Rounds (J × Rounds = 400)}\n")
            f.write("\\label{tab:noniid_scaled}\n")
            f.write(latex_noniid)
            f.write("\\end{table}\n")
        
        print("✓ Saved: noniid_table.tex")


# ============================================================================
# Main Execution
# ============================================================================
def main():
    print("=" * 60)
    print("Federated Learning Experiment Analysis")
    print("=" * 60)
    print()
    
    # Load all data
    print("Loading data...")
    
    main_central = load_central_baseline(MAIN_DIR)
    main_iid = load_fedavg_iid(MAIN_DIR)
    noniid_results = load_noniid_experiments(MAIN_DIR)
    
    extended_central = load_central_baseline(EXTENDED_DIR)
    extended_iid = load_fedavg_iid(EXTENDED_DIR)
    
    sparse_results = load_sparse_experiments(SPARSE_DIR)
    ablation_results = load_ablation_results(SPARSE_DIR)
    
    print(f"  ✓ Central baseline: {'Found' if main_central else 'Not found'}")
    print(f"  ✓ Extended central: {'Found' if extended_central else 'Not found'}")
    print(f"  ✓ FedAvg IID: {'Found' if main_iid else 'Not found'}")
    print(f"  ✓ Extended FedAvg IID: {'Found' if extended_iid else 'Not found'}")
    print(f"  ✓ Non-IID experiments: {len(noniid_results)} configurations")
    print(f"  ✓ Sparse FedAvg: {len(sparse_results)} experiments")
    print(f"  ✓ Ablation (calibration): {len(ablation_results.get('calibration', {}))} configs")
    print(f"  ✓ Ablation (sparsity): {len(ablation_results.get('sparsity', {}))} configs")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 40)
    
    if main_central:
        plot_central_baseline(main_central, extended_central)
    
    if main_iid:
        plot_fedavg_iid_comparison(main_iid, extended_iid)
    
    noniid_df = plot_noniid_heatmap(noniid_results)
    
    plot_sparse_comparison_all_rules(sparse_results, main_iid)
    
    plot_ablation_studies(ablation_results)
    
    print()
    
    # Generate summary
    print("Generating summary tables...")
    print("-" * 40)
    
    summary_df = generate_summary_csv(main_central, extended_central,
                                       main_iid, extended_iid,
                                       noniid_results, sparse_results, ablation_results)
    
    summary_df.to_csv(SCRIPT_DIR / 'summary_results.csv', index=False)
    print("✓ Saved: summary_results.csv")
    
    generate_latex_tables(summary_df, noniid_df)
    
    print()
    print("=" * 60)
    print("Summary Results")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    
    # Key findings
    print("=" * 60)
    print("Key Findings")
    print("=" * 60)
    
    if sparse_results:
        print("\n• Mask Rule Comparison (Non-IID):")
        noniid_sparse = {k: v for k, v in sparse_results.items() if 'noniid' in k.lower() or 'niid' in k.lower()}
        for exp_name, data in sorted(noniid_sparse.items(), 
                                      key=lambda x: extract_final_and_best_acc(x[1])[0] or 0, 
                                      reverse=True):
            final_acc, _ = extract_final_and_best_acc(data)
            if final_acc:
                print(f"  → {exp_name}: {final_acc:.2f}%")
    
    print()
    print("✓ Analysis complete! Check 'figures/' directory for plots.")


if __name__ == "__main__":
    main()
