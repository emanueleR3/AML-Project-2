#!/usr/bin/env python3
"""
Milestone M8 — FedAvg with Sparse training (SparseSGDM + mask calibration).

Stages:
1. calibrate_mask: Compute and save mask using Fisher/sensitivity scores
2. train_sparse: Run FedAvg with SparseSGDM using calibrated mask

Usage:
    python scripts/run_sparse_fedavg.py --config configs/sparse_fedavg.yaml --stage calibrate_mask
    python scripts/run_sparse_fedavg.py --config configs/sparse_fedavg.yaml --stage train_sparse
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src import (
    load_dino_backbone, DINOClassifier, build_model,
    get_transforms, load_cifar100, create_dataloader,
    partition_iid, partition_non_iid,
    run_fedavg_round, fedavg_aggregate,
    evaluate, local_train, set_seed, ensure_dir,
    save_checkpoint, load_checkpoint, save_metrics_json,
    SparseSGDM
)
from src.masking import (
    compute_sensitivity_scores, create_mask, get_mask_sparsity,
    save_mask, load_mask, MaskRule
)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    ensure_dir(str(log_dir))
    
    logger = logging.getLogger('SparseFedAvg')
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(log_dir / 'run.log')
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def calibrate_mask_stage(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    logger: logging.Logger,
    output_dir: Path
) -> Dict[str, torch.Tensor]:
    """
    Stage 1: Calibrate mask using Fisher information or magnitude-based method.
    
    Args:
        model: Model to calibrate mask for
        train_loader: Training data loader (used for Fisher computation)
        device: Device to use
        config: Configuration dict
        logger: Logger instance
        output_dir: Directory to save mask
        
    Returns:
        Mask dict {param_name -> tensor}
    """
    logger.info("=" * 70)
    logger.info("STAGE 1: CALIBRATE MASK")
    logger.info("=" * 70)
    
    sparsity_ratio = config.get('sparsity_ratio', 0.8)
    mask_rule = config.get('mask_rule', 'least_sensitive')
    num_calib_batches = config.get('num_calib_batches', 50)
    seed = config.get('seed', 42)
    
    logger.info(f"Mask Rule: {mask_rule}")
    logger.info(f"Sparsity Ratio: {sparsity_ratio}")
    logger.info(f"Calibration Batches: {num_calib_batches}")
    
    model.eval()
    
    # Compute sensitivity scores using Fisher information
    logger.info("Computing Fisher diagonal information...")
    scores = compute_sensitivity_scores(
        model, train_loader, device,
        num_batches=num_calib_batches,
        method='fisher'
    )
    
    # Create mask
    logger.info(f"Creating mask with rule '{mask_rule}'...")
    mask = create_mask(
        scores, model, sparsity_ratio,
        rule=mask_rule,
        seed=seed
    )
    
    # Log mask statistics
    mask_sparsity = get_mask_sparsity(mask)
    logger.info(f"Mask Sparsity (active ratio): {mask_sparsity:.4f}")
    
    for name, m in mask.items():
        active = m.sum().item()
        total = m.numel()
        logger.info(f"  {name}: {active}/{total} active ({100*active/total:.1f}%)")
    
    # Save mask
    mask_path = output_dir / f"mask_{mask_rule}_{sparsity_ratio:.2f}.pt"
    save_mask(mask, str(mask_path))
    logger.info(f"Mask saved to {mask_path}")
    
    return mask


def client_update_sparse(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float,
    weight_decay: float,
    device: torch.device,
    local_steps: int,
    mask: Dict[str, torch.Tensor],
    criterion: Optional[nn.Module] = None
) -> Tuple[Dict[str, torch.Tensor], float, float, int]:
    """
    Performs local training on a client using SparseSGDM.
    
    Args:
        model: Model with global weights
        train_loader: Client's training data
        lr: Learning rate
        weight_decay: Weight decay
        device: Device
        local_steps: Local steps (J)
        mask: Mask dict for sparse training
        criterion: Loss function
        
    Returns:
        Tuple of (state_dict, avg_loss, avg_acc, n_samples)
    """
    import copy
    
    local_model = copy.deepcopy(model)
    local_model.to(device)
    local_model.train()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Use SparseSGDM instead of standard SGD
    optimizer = SparseSGDM(
        local_model.get_trainable_params(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        mask=mask,
        apply_wd_to_masked_only=True
    )
    
    avg_loss, avg_acc, n_samples = local_train(
        local_model, train_loader, optimizer, criterion, device, local_steps
    )
    
    return local_model.state_dict(), avg_loss, avg_acc, n_samples


def run_fedavg_sparse_round(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    selected_clients: List[int],
    lr: float,
    weight_decay: float,
    device: torch.device,
    local_steps: int,
    mask: Dict[str, torch.Tensor],
    criterion: Optional[nn.Module] = None,
    show_progress: bool = False
) -> Tuple[float, float]:
    """
    Run a single FedAvg round with sparse training (SparseSGDM).
    
    Args:
        global_model: Global model
        client_loaders: List of all client data loaders
        selected_clients: Indices of selected clients
        lr: Learning rate
        weight_decay: Weight decay
        device: Device
        local_steps: Local steps
        mask: Mask dict for sparse training
        criterion: Loss function
        show_progress: Show progress
        
    Returns:
        Tuple of (avg_loss, avg_acc)
    """
    from src.utils import AverageMeter
    
    client_state_dicts = []
    client_weights = []
    round_loss = AverageMeter()
    round_acc = AverageMeter()
    
    iterator = tqdm(selected_clients, desc='Clients', leave=False) if show_progress else selected_clients
    
    for client_idx in iterator:
        loader = client_loaders[client_idx]
        
        state_dict, loss, acc, n_samples = client_update_sparse(
            global_model, loader, lr, weight_decay, device, local_steps, mask, criterion
        )
        
        client_state_dicts.append(state_dict)
        client_weights.append(n_samples)
        round_loss.update(loss, n_samples)
        round_acc.update(acc, n_samples)
    
    # Aggregate
    fedavg_aggregate(global_model, client_state_dicts, client_weights)
    
    return round_loss.avg, round_acc.avg


def train_sparse_stage(
    model: nn.Module,
    client_loaders: List[DataLoader],
    val_loader: DataLoader,
    test_loader: DataLoader,
    mask: Dict[str, torch.Tensor],
    device: torch.device,
    config: Dict[str, Any],
    logger: logging.Logger,
    output_dir: Path
) -> Dict[str, List[float]]:
    """
    Stage 2: Run FedAvg training with SparseSGDM and calibrated mask.
    
    Args:
        model: Global model
        client_loaders: List of client data loaders
        val_loader: Validation data
        test_loader: Test data
        mask: Calibrated mask
        device: Device
        config: Configuration dict
        logger: Logger instance
        output_dir: Directory to save results
        
    Returns:
        History dict with metrics
    """
    logger.info("=" * 70)
    logger.info("STAGE 2: TRAIN SPARSE FEDAVG")
    logger.info("=" * 70)
    
    num_rounds = config.get('num_rounds', 100)
    num_clients = config.get('num_clients', 100)
    clients_per_round = config.get('clients_per_round', 0.1)
    local_steps = config.get('local_steps', 4)
    lr = config.get('lr', 0.01)
    weight_decay = config.get('weight_decay', 1e-4)
    seed = config.get('seed', 42)
    eval_freq = config.get('eval_freq', 1)
    
    np.random.seed(seed)
    criterion = nn.CrossEntropyLoss()
    m = max(1, int(clients_per_round * num_clients))
    
    logger.info(f"Configuration: K={num_clients}, m={m}, J={local_steps}, R={num_rounds}")
    logger.info(f"Learning rate: {lr}, Weight decay: {weight_decay}")
    
    history = {
        'round': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_val_acc = 0.0
    best_state_dict = None
    
    for round_idx in range(1, num_rounds + 1):
        selected_clients = np.random.choice(num_clients, m, replace=False).tolist()
        
        # Run sparse round
        train_loss, train_acc = run_fedavg_sparse_round(
            model, client_loaders, selected_clients,
            lr, weight_decay, device, local_steps, mask, criterion
        )
        
        # Evaluate
        if round_idx % eval_freq == 0 or round_idx == num_rounds:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, show_progress=False)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device, show_progress=False)
        else:
            val_loss, val_acc = float('nan'), float('nan')
            test_loss, test_acc = float('nan'), float('nan')
        
        history['round'].append(round_idx)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = __import__('copy').deepcopy(model.state_dict())
        
        if round_idx % max(1, num_rounds // 10) == 0 or round_idx == num_rounds:
            logger.info(f"Round {round_idx}/{num_rounds} | "
                       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                       f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    history['best_val_acc'] = best_val_acc
    
    logger.info(f"Training completed. Best Val Acc: {best_val_acc:.2f}%")
    
    return history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Sparse FedAvg Training (M8)')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--stage', type=str, choices=['calibrate_mask', 'train_sparse'],
                       required=True, help='Execution stage')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')
    parser.add_argument('--dry_run', type=int, default=0, help='Dry run mode (quick test)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.seed is not None:
        config['seed'] = args.seed
    
    set_seed(config.get('seed', 42))
    
    # Setup output directory
    exp_name = config.get('exp_name', 'sparse_fedavg')
    output_dir = Path('outputs') / exp_name
    ensure_dir(str(output_dir))
    
    logger = setup_logging(output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Config:\n{json.dumps(config, indent=2)}")
    
    # Load data
    logger.info("Loading CIFAR-100 data...")
    train_full, test_data = load_cifar100(data_dir='./data', download=True)
    
    # Split train into train/val (80/20)
    train_size = int(0.8 * len(train_full))
    val_size = len(train_full) - train_size
    train_data, val_data = torch.utils.data.random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )
    
    # Partition data
    logger.info("Partitioning data...")
    num_clients = config.get('num_clients', 100)
    is_iid = config.get('iid', True)
    
    if is_iid:
        client_subsets = partition_iid(train_data, num_clients)
        logger.info(f"IID partition: {num_clients} clients")
    else:
        nc = config.get('nc', 5)  # Classes per client
        client_subsets = partition_non_iid(train_data, num_clients, nc)
        logger.info(f"Non-IID partition: {num_clients} clients, {nc} classes each")
    
    # Create data loaders
    batch_size = config.get('batch_size', 32)
    client_loaders = [
        create_dataloader(subset, batch_size, shuffle=True)
        for subset in client_subsets
    ]
    val_loader = create_dataloader(val_data, batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, batch_size, shuffle=False)
    
    logger.info(f"Data ready: {len(client_loaders)} clients, "
               f"val={len(val_data)}, test={len(test_data)}")
    
    # Load model
    logger.info("Building model...")
    model_config = {
        'model_name': config.get('model', 'dino_vits16'),
        'num_classes': 100,
        'dropout': 0.1,
        'freeze_policy': 'head_only',
        'device': device
    }
    model = build_model(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # === STAGE: CALIBRATE MASK ===
    if args.stage == 'calibrate_mask':
        # Use only first few batches for calibration
        if args.dry_run:
            config['num_calib_batches'] = 1
        
        mask = calibrate_mask_stage(
            model, client_loaders[0], device, config, logger, output_dir
        )
        
        logger.info(f"✅ Mask calibration complete")
        return
    
    # === STAGE: TRAIN SPARSE ===
    if args.stage == 'train_sparse':
        # Load mask
        mask_rule = config.get('mask_rule', 'least_sensitive')
        sparsity_ratio = config.get('sparsity_ratio', 0.8)
        mask_path = output_dir / f"mask_{mask_rule}_{sparsity_ratio:.2f}.pt"
        
        if not mask_path.exists():
            logger.error(f"Mask not found at {mask_path}")
            logger.error("Please run --stage calibrate_mask first")
            return
        
        logger.info(f"Loading mask from {mask_path}")
        mask = load_mask(str(mask_path))
        
        # Override config for dry run
        if args.dry_run:
            config['num_rounds'] = 2
        
        # Run sparse training
        history = train_sparse_stage(
            model, client_loaders, val_loader, test_loader,
            mask, device, config, logger, output_dir
        )
        
        # Save history
        history_path = output_dir / f"history_{mask_rule}_{sparsity_ratio:.2f}.json"
        save_metrics_json(str(history_path), history)
        logger.info(f"History saved to {history_path}")
        
        # Save final model
        final_path = output_dir / f"final_model_{mask_rule}_{sparsity_ratio:.2f}.pt"
        save_checkpoint({
            'model': model.state_dict(),
            'mask': mask,
            'config': config
        }, str(final_path))
        logger.info(f"Final model saved to {final_path}")
        
        logger.info(f"✅ Sparse training complete")


if __name__ == '__main__':
    main()
