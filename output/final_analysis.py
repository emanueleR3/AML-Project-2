#!/usr/bin/env python3
"""
Federated Learning Experiment Analysis Script
=============================================
This script analyzes FL experiment results and generates publication-ready
figures and tables for the M9 final report.

Outputs:
- Central baseline training curves
- FedAvg IID convergence (100 vs 300 rounds comparison)
- Non-IID Nc×J heatmap
- Sparse FedAvg comparison across mask rules
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
    'sparse_ls': '#9b59b6',     # Purple
    'sparse_rnd': '#f39c12',    # Orange
    'sparse_hm': '#1abc9c',     # Teal
    '100_rounds': '#3498db',    # Blue
    '300_rounds': '#e74c3c',    # Red (lighter)
}

# Paths
SCRIPT_DIR = Path(__file__).parent
MAIN_DIR = SCRIPT_DIR / "main"
EXTENDED_DIR = SCRIPT_DIR / "extended"
MASKS_DIR = SCRIPT_DIR / "masks"
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
    for filepath in directory.glob("exp_*_metrics.json"):
        name = filepath.stem.replace('_metrics', '')
        results[name] = load_json(filepath)
    return results


def extract_final_and_best_acc(metrics: dict) -> Tuple[float, float]:
    """Extract final and best validation/test accuracy from metrics."""
    # Check for best_val_acc summary field first
    if 'best_val_acc' in metrics:
        best_acc = metrics['best_val_acc']
        # For final acc, use last valid test_acc or val_acc
        final_acc = best_acc  # Default to best
        
        for acc_key in ['test_acc', 'val_acc']:
            if acc_key in metrics:
                accs = [a for a in metrics[acc_key] if a is not None and not (isinstance(a, float) and np.isnan(a))]
                if accs:
                    final_acc = accs[-1]
                    break
        return final_acc, best_acc
    
    # Fallback: Check for test_acc first, then val_acc
    for acc_key in ['test_acc', 'val_acc']:
        if acc_key in metrics:
            accs = [a for a in metrics[acc_key] if a is not None and not (isinstance(a, float) and np.isnan(a))]
            if accs:
                return accs[-1], max(accs)
    return None, None


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
             color=COLORS['100_rounds'], linewidth=2, label='100 Epochs', marker='o', markersize=4)
    
    if extended_data:
        epochs_ext = range(1, len(extended_data['train_loss']) + 1)
        ax1.plot(epochs_ext, extended_data['train_loss'], 
                 color=COLORS['300_rounds'], linewidth=2, label='300 Epochs', marker='s', markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Central Baseline - Training Loss')
    ax1.legend()
    
    # Validation Accuracy
    ax2 = axes[1]
    # For central baseline, val_acc is recorded every N epochs
    n_val_main = len(main_data['val_acc'])
    val_epochs_main = np.linspace(1, len(main_data['train_loss']), n_val_main).astype(int)
    ax2.plot(val_epochs_main, main_data['val_acc'], 
             color=COLORS['100_rounds'], linewidth=2, label='100 Epochs', marker='o', markersize=6)
    
    if extended_data:
        n_val_ext = len(extended_data['val_acc'])
        val_epochs_ext = np.linspace(1, len(extended_data['train_loss']), n_val_ext).astype(int)
        ax2.plot(val_epochs_ext, extended_data['val_acc'], 
                 color=COLORS['300_rounds'], linewidth=2, label='300 Epochs', marker='s', markersize=6)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Central Baseline - Validation Accuracy')
    ax2.legend()
    
    # Add best accuracy annotation
    best_acc_main = max(main_data['val_acc'])
    ax2.axhline(y=best_acc_main, color=COLORS['100_rounds'], linestyle='--', alpha=0.5)
    ax2.annotate(f'Best: {best_acc_main:.2f}%', 
                 xy=(val_epochs_main[-1], best_acc_main), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'central_baseline_curves.png')
    plt.savefig(FIGURES_DIR / 'central_baseline_curves.pdf')
    plt.close()
    print("✓ Saved: central_baseline_curves.png/pdf")


def plot_fedavg_iid_comparison(main_data: dict, extended_data: dict = None):
    """Plot FedAvg IID convergence comparison (100 vs 300 rounds)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Training Loss
    ax1 = axes[0]
    rounds_main = main_data['round']
    ax1.plot(rounds_main, main_data['train_loss'], 
             color=COLORS['100_rounds'], linewidth=2, label='100 Rounds', alpha=0.9)
    
    if extended_data:
        rounds_ext = extended_data['round']
        ax1.plot(rounds_ext, extended_data['train_loss'], 
                 color=COLORS['300_rounds'], linewidth=2, label='300 Rounds', alpha=0.9)
        # Mark where 100 rounds ends
        ax1.axvline(x=100, color='gray', linestyle=':', alpha=0.7, label='100 Round Mark')
    
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('FedAvg IID - Training Loss Convergence')
    ax1.legend()
    
    # Test/Validation Accuracy - prefer test_acc as it has more data points
    ax2 = axes[1]
    
    # Use test_acc if available (more data points), otherwise val_acc
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
    
    # Annotate final accuracies
    final_acc_main = acc_main[valid_mask_main][-1]
    ax2.annotate(f'{final_acc_main:.1f}%', 
                 xy=(rounds_arr_main[valid_mask_main][-1], final_acc_main),
                 xytext=(5, 0), textcoords='offset points', fontsize=9,
                 color=COLORS['100_rounds'])
    
    if extended_data:
        final_acc_ext = acc_ext[valid_mask_ext][-1]
        ax2.annotate(f'{final_acc_ext:.1f}%', 
                     xy=(rounds_arr_ext[valid_mask_ext][-1], final_acc_ext),
                     xytext=(5, 0), textcoords='offset points', fontsize=9,
                     color=COLORS['300_rounds'])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fedavg_iid_convergence.png')
    plt.savefig(FIGURES_DIR / 'fedavg_iid_convergence.pdf')
    plt.close()
    print("✓ Saved: fedavg_iid_convergence.png/pdf")


def plot_noniid_heatmap(noniid_results: Dict[Tuple[int, int], dict]):
    """Create heatmap of Non-IID results (Nc × J) - Final Test Accuracy only."""
    if not noniid_results:
        print("⚠ No non-IID results found, skipping heatmap")
        return
    
    # Extract all unique Nc and J values
    ncs = sorted(set(k[0] for k in noniid_results.keys()))
    js = sorted(set(k[1] for k in noniid_results.keys()))
    
    # Create matrix for final accuracy only
    final_acc_matrix = np.full((len(ncs), len(js)), np.nan)
    
    for (nc, j), data in noniid_results.items():
        nc_idx = ncs.index(nc)
        j_idx = js.index(j)
        final_acc, best_acc = extract_final_and_best_acc(data)
        if final_acc is not None:
            final_acc_matrix[nc_idx, j_idx] = final_acc
    
    # Create figure with single heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Final Accuracy Heatmap only
    sns.heatmap(final_acc_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=[f'J={j}' for j in js],
                yticklabels=[f'Nc={nc}' for nc in ncs],
                ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title('Non-IID: Final Test Accuracy (%)')
    ax.set_xlabel('Local Steps (J)')
    ax.set_ylabel('Classes per Client (Nc)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'noniid_heatmap.png')
    plt.savefig(FIGURES_DIR / 'noniid_heatmap.pdf')
    plt.close()
    print("✓ Saved: noniid_heatmap.png/pdf")
    
    # Also create a detailed table
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
                rows.append({
                    'Nc': nc,
                    'J': j,
                    'Final Test Acc (%)': final_acc,
                    'Best Test Acc (%)': best_acc,
                })
    
    df = pd.DataFrame(rows)
    return df


def plot_sparse_fedavg_comparison(sparse_results: Dict[str, dict], 
                                  fedavg_iid_data: dict = None):
    """Plot sparse FedAvg comparison across mask rules - using test accuracy only."""
    if not sparse_results:
        print("⚠ No sparse FedAvg results found, skipping")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Color mapping for experiments
    exp_colors = {
        'exp_iid_ls': COLORS['sparse_ls'],
        'exp_niid_ls': COLORS['noniid'],
        'exp_niid_rnd': COLORS['sparse_rnd'],
    }
    
    exp_labels = {
        'exp_iid_ls': 'IID + Least Sensitive',
        'exp_niid_ls': 'Non-IID + Least Sensitive',
        'exp_niid_rnd': 'Non-IID + Random',
    }
    
    # Plot 1: Test Accuracy Curves
    ax1 = axes[0]
    
    # Plot dense FedAvg IID as baseline
    if fedavg_iid_data:
        # Use test_acc
        acc = np.array(fedavg_iid_data['test_acc'])
        rounds = np.array(fedavg_iid_data['round'])
        valid_mask = ~np.isnan(acc)
        ax1.plot(rounds[valid_mask], acc[valid_mask],
                 color=COLORS['fedavg_iid'], linewidth=2, linestyle='--',
                 label='Dense FedAvg IID', alpha=0.8)
    
    for exp_name, data in sparse_results.items():
        color = exp_colors.get(exp_name, 'gray')
        label = exp_labels.get(exp_name, exp_name)
        # Use test_acc instead of val_acc
        ax1.plot(data['round'], data['test_acc'],
                 color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Sparse FedAvg (80% Sparsity) - Convergence')
    ax1.legend(loc='lower right')
    
    # Plot 2: Final/Best Accuracy Bar Chart (using test_acc)
    ax2 = axes[1]
    
    exp_names = list(sparse_results.keys())
    final_accs = []
    best_accs = []
    labels = []
    colors = []
    
    for exp_name in exp_names:
        data = sparse_results[exp_name]
        # Use test_acc
        final_acc = data['test_acc'][-1] if data['test_acc'] else 0
        best_acc = max(data['test_acc']) if data['test_acc'] else 0
        final_accs.append(final_acc)
        best_accs.append(best_acc)
        labels.append(exp_labels.get(exp_name, exp_name))
        colors.append(exp_colors.get(exp_name, 'gray'))
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, final_accs, width, label='Final Acc', color=colors, alpha=0.7)
    bars2 = ax2.bar(x + width/2, best_accs, width, label='Best Acc', color=colors, alpha=1.0)
    
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Sparse FedAvg - Final vs Best Test Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace(' + ', '\n') for l in labels], fontsize=9)
    ax2.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    # Add dense baseline line using test_acc
    if fedavg_iid_data:
        acc = np.array(fedavg_iid_data['test_acc'])
        valid_acc = acc[~np.isnan(acc)]
        dense_best = max(valid_acc) if len(valid_acc) > 0 else 0
        ax2.axhline(y=dense_best, color=COLORS['fedavg_iid'], linestyle='--', 
                    alpha=0.8, label=f'Dense FedAvg Best ({dense_best:.1f}%)')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sparse_fedavg_comparison.png')
    plt.savefig(FIGURES_DIR / 'sparse_fedavg_comparison.pdf')
    plt.close()
    print("✓ Saved: sparse_fedavg_comparison.png/pdf")


def plot_extended_training_analysis(main_iid: dict, extended_iid: dict,
                                    main_central: dict, extended_central: dict):
    """
    Analyze extended training (300 vs 100 rounds) to show diminishing returns.
    This supports the argument about model capacity saturation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Use test_acc if available (more data points)
    acc_key = 'test_acc' if 'test_acc' in main_iid else 'val_acc'
    
    # Top row: FedAvg IID comparison
    ax1, ax2 = axes[0]
    
    # Training loss comparison
    ax1.plot(main_iid['round'], main_iid['train_loss'], 
             color=COLORS['100_rounds'], linewidth=2, label='100 Rounds')
    if extended_iid:
        ax1.plot(extended_iid['round'], extended_iid['train_loss'],
                 color=COLORS['300_rounds'], linewidth=2, label='300 Rounds')
        ax1.axvline(x=100, color='gray', linestyle=':', alpha=0.7)
        
        # Highlight the region after 100 rounds
        ax1.axvspan(100, 300, alpha=0.1, color='gray', label='Extended Training')
    
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('FedAvg IID: Training Loss')
    ax1.legend()
    
    # Test accuracy with saturation analysis
    acc_main = np.array(main_iid[acc_key])
    rounds_main = np.array(main_iid['round'])
    valid_mask_main = ~np.isnan(acc_main)
    
    ax2.plot(rounds_main[valid_mask_main], acc_main[valid_mask_main],
             color=COLORS['100_rounds'], linewidth=2, marker='o', markersize=4,
             label='100 Rounds')
    
    if extended_iid:
        acc_ext = np.array(extended_iid[acc_key])
        rounds_ext = np.array(extended_iid['round'])
        valid_mask_ext = ~np.isnan(acc_ext)
        
        ax2.plot(rounds_ext[valid_mask_ext], acc_ext[valid_mask_ext],
                 color=COLORS['300_rounds'], linewidth=2, marker='s', markersize=4,
                 label='300 Rounds')
        ax2.axvline(x=100, color='gray', linestyle=':', alpha=0.7)
        
        # Calculate improvement after 100 rounds
        # Find accuracy at round 100 from extended run
        acc_at_100_mask = rounds_ext[valid_mask_ext] <= 100
        if acc_at_100_mask.any():
            acc_at_100 = acc_ext[valid_mask_ext][acc_at_100_mask][-1]
        else:
            acc_at_100 = acc_ext[valid_mask_ext][0]
        acc_at_300 = acc_ext[valid_mask_ext][-1]
        improvement = acc_at_300 - acc_at_100
        
        ax2.annotate(f'Δ = +{improvement:.2f}%\n(+200 rounds)',
                     xy=(250, (acc_at_100 + acc_at_300) / 2),
                     fontsize=10, ha='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('FedAvg IID: Test Accuracy (Saturation Analysis)')
    ax2.legend()
    
    # Bottom row: Central baseline comparison
    ax3, ax4 = axes[1]
    
    # Central training loss
    epochs_main = range(1, len(main_central['train_loss']) + 1)
    ax3.plot(epochs_main, main_central['train_loss'],
             color=COLORS['100_rounds'], linewidth=2, label='100 Epochs')
    
    if extended_central:
        epochs_ext = range(1, len(extended_central['train_loss']) + 1)
        ax3.plot(epochs_ext, extended_central['train_loss'],
                 color=COLORS['300_rounds'], linewidth=2, label='300 Epochs')
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Loss')
    ax3.set_title('Central Baseline: Training Loss')
    ax3.legend()
    
    # Central validation accuracy
    n_val_main = len(main_central['val_acc'])
    val_epochs_main = np.linspace(1, len(main_central['train_loss']), n_val_main).astype(int)
    ax4.plot(val_epochs_main, main_central['val_acc'],
             color=COLORS['100_rounds'], linewidth=2, marker='o', markersize=6,
             label='100 Epochs')
    
    if extended_central:
        n_val_ext = len(extended_central['val_acc'])
        val_epochs_ext = np.linspace(1, len(extended_central['train_loss']), n_val_ext).astype(int)
        ax4.plot(val_epochs_ext, extended_central['val_acc'],
                 color=COLORS['300_rounds'], linewidth=2, marker='s', markersize=6,
                 label='300 Epochs')
        
        # Calculate improvement
        acc_100_central = max(main_central['val_acc'])
        acc_300_central = max(extended_central['val_acc'])
        improvement_central = acc_300_central - acc_100_central
        
        ax4.annotate(f'Δ = +{improvement_central:.2f}%\n(+200 epochs)',
                     xy=(val_epochs_ext[-1] * 0.75, (acc_100_central + acc_300_central) / 2),
                     fontsize=10, ha='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.set_title('Central Baseline: Validation Accuracy (Saturation Analysis)')
    ax4.legend()
    
    plt.suptitle('Extended Training Analysis: Diminishing Returns', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'extended_training_analysis.png')
    plt.savefig(FIGURES_DIR / 'extended_training_analysis.pdf')
    plt.close()
    print("✓ Saved: extended_training_analysis.png/pdf")


def plot_iid_vs_noniid_comparison(fedavg_iid: dict, noniid_results: Dict[Tuple[int, int], dict]):
    """Plot IID vs Non-IID (various Nc) comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use test_acc if available
    acc_key = 'test_acc' if 'test_acc' in fedavg_iid else 'val_acc'
    
    # Plot IID baseline
    acc_iid = np.array(fedavg_iid[acc_key])
    rounds_iid = np.array(fedavg_iid['round'])
    valid_mask = ~np.isnan(acc_iid)
    ax.plot(rounds_iid[valid_mask], acc_iid[valid_mask],
            color=COLORS['fedavg_iid'], linewidth=2.5, label='IID', marker='o', markersize=4)
    
    # Plot non-IID with varying Nc (fix J=4 for comparison)
    nc_colors = {1: '#e74c3c', 5: '#f39c12', 10: '#9b59b6', 50: '#27ae60'}
    
    for (nc, j), data in sorted(noniid_results.items()):
        if j == 4:  # Fix J for fair comparison
            # Use test_acc if available
            data_acc_key = 'test_acc' if 'test_acc' in data else 'val_acc'
            acc = np.array(data[data_acc_key]) if data_acc_key in data else np.array([])
            rounds = np.array(data['round'])
            valid = ~np.isnan(acc)
            
            if valid.any():
                color = nc_colors.get(nc, 'gray')
                ax.plot(rounds[valid], acc[valid],
                        color=color, linewidth=2, label=f'Non-IID (Nc={nc})',
                        marker='s', markersize=3, alpha=0.8)
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('IID vs Non-IID Data Distribution (J=4 clients/round)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'iid_vs_noniid_comparison.png')
    plt.savefig(FIGURES_DIR / 'iid_vs_noniid_comparison.pdf')
    plt.close()
    print("✓ Saved: iid_vs_noniid_comparison.png/pdf")


# ============================================================================
# Summary Generation Functions
# ============================================================================
def generate_summary_csv(main_central: dict, extended_central: dict,
                         main_iid: dict, extended_iid: dict,
                         noniid_results: Dict[Tuple[int, int], dict],
                         sparse_results: Dict[str, dict]) -> pd.DataFrame:
    """Generate comprehensive summary CSV."""
    rows = []
    
    # Central Baseline
    if main_central:
        rows.append({
            'Experiment': 'Central Baseline (100 epochs)',
            'Best Acc (%)': max(main_central['val_acc']),
            'Final Acc (%)': main_central['val_acc'][-1],
            'Rounds/Epochs': len(main_central['train_loss']),
            'Sparsity': '0%'
        })
    
    if extended_central:
        rows.append({
            'Experiment': 'Central Baseline (300 epochs)',
            'Best Acc (%)': max(extended_central['val_acc']),
            'Final Acc (%)': extended_central['val_acc'][-1],
            'Rounds/Epochs': len(extended_central['train_loss']),
            'Sparsity': '0%'
        })
    
    # FedAvg IID
    if main_iid:
        val_acc = np.array(main_iid['val_acc'])
        valid_acc = val_acc[~np.isnan(val_acc)]
        rows.append({
            'Experiment': 'FedAvg IID (100 rounds)',
            'Best Acc (%)': max(valid_acc),
            'Final Acc (%)': valid_acc[-1],
            'Rounds/Epochs': max(main_iid['round']),
            'Sparsity': '0%'
        })
    
    if extended_iid:
        val_acc = np.array(extended_iid['val_acc'])
        valid_acc = val_acc[~np.isnan(val_acc)]
        rows.append({
            'Experiment': 'FedAvg IID (300 rounds)',
            'Best Acc (%)': max(valid_acc),
            'Final Acc (%)': valid_acc[-1],
            'Rounds/Epochs': max(extended_iid['round']),
            'Sparsity': '0%'
        })
    
    # Non-IID experiments
    for (nc, j), data in sorted(noniid_results.items()):
        final_acc, best_acc = extract_final_and_best_acc(data)
        if final_acc is not None:
            rows.append({
                'Experiment': f'FedAvg Non-IID (Nc={nc}, J={j})',
                'Best Acc (%)': best_acc,
                'Final Acc (%)': final_acc,
                'Rounds/Epochs': max(data['round']),
                'Sparsity': '0%'
            })
    
    # Sparse FedAvg experiments
    exp_labels = {
        'exp_iid_ls': 'Sparse FedAvg IID (Least Sensitive)',
        'exp_niid_ls': 'Sparse FedAvg Non-IID (Least Sensitive)',
        'exp_niid_rnd': 'Sparse FedAvg Non-IID (Random)',
    }
    
    for exp_name, data in sparse_results.items():
        label = exp_labels.get(exp_name, exp_name)
        best_acc = max(data['val_acc']) if data['val_acc'] else 0
        final_acc = data['val_acc'][-1] if data['val_acc'] else 0
        rows.append({
            'Experiment': label,
            'Best Acc (%)': best_acc,
            'Final Acc (%)': final_acc,
            'Rounds/Epochs': max(data['round']),
            'Sparsity': '80%'
        })
    
    df = pd.DataFrame(rows)
    df = df.round(2)
    return df


def generate_latex_tables(summary_df: pd.DataFrame, noniid_df: pd.DataFrame):
    """Generate LaTeX formatted tables for the report."""
    
    # Main summary table
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
    
    # Non-IID heatmap table
    if noniid_df is not None and not noniid_df.empty:
        latex_noniid = noniid_df.to_latex(index=False, escape=False,
                                           column_format='cccc')
        
        with open(SCRIPT_DIR / 'noniid_table.tex', 'w') as f:
            f.write("% Auto-generated LaTeX table - Non-IID Results\n")
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\caption{Non-IID Experiment Results (Nc × J sweep)}\n")
            f.write("\\label{tab:noniid}\n")
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
    
    # Main results (100 rounds/epochs)
    main_central = load_central_baseline(MAIN_DIR)
    main_iid = load_fedavg_iid(MAIN_DIR)
    noniid_results = load_noniid_experiments(MAIN_DIR)
    
    # Extended results (300 rounds/epochs)
    extended_central = load_central_baseline(EXTENDED_DIR)
    extended_iid = load_fedavg_iid(EXTENDED_DIR)
    
    # Sparse FedAvg results
    sparse_results = load_sparse_experiments(MASKS_DIR)
    
    print(f"  ✓ Central baseline: {'100 epochs' if main_central else 'Not found'}")
    print(f"  ✓ Extended central: {'300 epochs' if extended_central else 'Not found'}")
    print(f"  ✓ FedAvg IID: {'100 rounds' if main_iid else 'Not found'}")
    print(f"  ✓ Extended FedAvg IID: {'300 rounds' if extended_iid else 'Not found'}")
    print(f"  ✓ Non-IID experiments: {len(noniid_results)} configurations")
    print(f"  ✓ Sparse FedAvg: {len(sparse_results)} experiments")
    print()
    
    # Generate all visualizations
    print("Generating visualizations...")
    print("-" * 40)
    
    # 1. Central baseline curves
    if main_central:
        plot_central_baseline(main_central, extended_central)
    
    # 2. FedAvg IID convergence comparison
    if main_iid:
        plot_fedavg_iid_comparison(main_iid, extended_iid)
    
    # 3. Non-IID heatmap
    noniid_df = plot_noniid_heatmap(noniid_results)
    
    # 4. Sparse FedAvg comparison
    plot_sparse_fedavg_comparison(sparse_results, main_iid)
    
    # 5. Extended training analysis (diminishing returns)
    if main_iid and extended_iid and main_central and extended_central:
        plot_extended_training_analysis(main_iid, extended_iid, 
                                        main_central, extended_central)
    
    # 6. IID vs Non-IID comparison
    if main_iid and noniid_results:
        plot_iid_vs_noniid_comparison(main_iid, noniid_results)
    
    print()
    
    # Generate summary tables
    print("Generating summary tables...")
    print("-" * 40)
    
    summary_df = generate_summary_csv(main_central, extended_central,
                                       main_iid, extended_iid,
                                       noniid_results, sparse_results)
    
    # Save CSV
    summary_df.to_csv(SCRIPT_DIR / 'summary_results.csv', index=False)
    print("✓ Saved: summary_results.csv")
    
    # Generate LaTeX tables
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
    
    if main_central and extended_central:
        improvement = max(extended_central['val_acc']) - max(main_central['val_acc'])
        print(f"• Central baseline: +{improvement:.2f}% from 100→300 epochs")
        print(f"  → Best accuracy: {max(main_central['val_acc']):.2f}% (100 epochs) vs {max(extended_central['val_acc']):.2f}% (300 epochs)")
    
    if main_iid and extended_iid:
        best_main = main_iid.get('best_val_acc', 0)
        best_ext = extended_iid.get('best_val_acc', 0)
        improvement = best_ext - best_main
        print(f"• FedAvg IID: +{improvement:.2f}% from 100→300 rounds")
        print(f"  → Best accuracy: {best_main:.2f}% (100 rounds) vs {best_ext:.2f}% (300 rounds)")
        print(f"  → Note: Significant gap vs central baseline ({max(main_central['val_acc']):.2f}%) indicates need for more rounds")
    
    if sparse_results and main_iid:
        dense_best = main_iid.get('best_val_acc', 0)
        print(f"\n• Sparse FedAvg (80% sparsity) vs Dense FedAvg IID ({dense_best:.2f}%):")
        
        for exp_name, data in sparse_results.items():
            sparse_best = max(data['val_acc'])
            diff = sparse_best - dense_best
            sign = '+' if diff >= 0 else ''
            print(f"  → {exp_name}: {sparse_best:.2f}% ({sign}{diff:.2f}%)")
        
        # Highlight key insight
        if 'exp_iid_ls' in sparse_results:
            sparse_iid = max(sparse_results['exp_iid_ls']['val_acc'])
            if sparse_iid > dense_best:
                print(f"\n  ★ Sparse IID outperforms Dense IID! (+{sparse_iid - dense_best:.2f}%)")
                print(f"    This suggests sparsity may provide regularization benefits")
    
    print()
    print("✓ Analysis complete! Check the 'figures' directory for plots.")


if __name__ == "__main__":
    main()
