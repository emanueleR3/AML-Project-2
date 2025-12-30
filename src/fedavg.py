"""FedAvg (Federated Averaging) algorithm implementation."""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from .train import local_train, evaluate
from .utils import AverageMeter


def client_update(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float,
    weight_decay: float,
    device: torch.device,
    local_steps: int,
    criterion: Optional[nn.Module] = None
) -> Tuple[Dict[str, torch.Tensor], float, float, int]:
    """
    Performs local training on a single client.
    
    Args:
        model: Model with global weights (will be copied)
        train_loader: Client's training data loader
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to use
        local_steps: Number of local steps (J)
        criterion: Loss function (default: CrossEntropyLoss)
        
    Returns:
        Tuple of (updated state_dict, avg loss, avg accuracy, num samples)
    """
    local_model = copy.deepcopy(model)
    local_model.to(device)
    local_model.train()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(
        local_model.get_trainable_params(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )
    
    avg_loss, avg_acc, n_samples = local_train(
        local_model, train_loader, optimizer, criterion, device, local_steps
    )
    
    return local_model.state_dict(), avg_loss, avg_acc, n_samples


def fedavg_aggregate(
    global_model: nn.Module,
    client_state_dicts: List[Dict[str, torch.Tensor]],
    client_weights: List[float]
) -> None:
    """
    Aggregates client models using FedAvg (weighted average).
    
    Args:
        global_model: Global model to update in-place
        client_state_dicts: List of client state dicts
        client_weights: Weights for each client (sum to 1)
    """
    # Normalize weights
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    # Initialize aggregated state dict
    global_state = global_model.state_dict()
    aggregated_state = {k: torch.zeros_like(v) for k, v in global_state.items()}
    
    # Weighted average
    for state_dict, weight in zip(client_state_dicts, normalized_weights):
        for key in aggregated_state:
            aggregated_state[key] += weight * state_dict[key].to(aggregated_state[key].device)
    
    # Update global model
    global_model.load_state_dict(aggregated_state)


def run_fedavg_round(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    selected_clients: List[int],
    lr: float,
    weight_decay: float,
    device: torch.device,
    local_steps: int,
    criterion: Optional[nn.Module] = None,
    show_progress: bool = False
) -> Tuple[float, float]:
    """
    Runs a single FedAvg round.
    
    Args:
        global_model: Global model
        client_loaders: List of all client data loaders
        selected_clients: Indices of selected clients for this round
        lr: Learning rate
        weight_decay: Weight decay
        device: Device to use
        local_steps: Number of local steps (J)
        criterion: Loss function
        show_progress: Whether to show progress
        
    Returns:
        Tuple of (average round loss, average round accuracy)
    """
    client_state_dicts = []
    client_weights = []
    round_loss = AverageMeter()
    round_acc = AverageMeter()
    
    iterator = tqdm(selected_clients, desc='Clients', leave=False) if show_progress else selected_clients
    
    for client_idx in iterator:
        loader = client_loaders[client_idx]
        
        state_dict, loss, acc, n_samples = client_update(
            global_model, loader, lr, weight_decay, device, local_steps, criterion
        )
        
        client_state_dicts.append(state_dict)
        client_weights.append(n_samples)
        round_loss.update(loss, n_samples)
        round_acc.update(acc, n_samples)
    
    # Aggregate
    fedavg_aggregate(global_model, client_state_dicts, client_weights)
    
    return round_loss.avg, round_acc.avg


def run_fedavg(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    callbacks: Optional[Dict[str, callable]] = None
) -> Dict[str, List[float]]:
    """
    Runs the complete FedAvg training.
    
    Args:
        global_model: Initial global model
        client_loaders: List of client data loaders
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration dict with keys:
            - num_rounds: Number of FL rounds
            - num_clients: Total number of clients (K)
            - clients_per_round: Fraction of clients per round (C)
            - local_steps: Local steps per client (J)
            - lr: Learning rate
            - weight_decay: Weight decay
            - seed: Random seed
        device: Device to use
        callbacks: Optional dict with 'on_round_end' callback
        
    Returns:
        History dict with training metrics
    """
    num_rounds = config.get('num_rounds', 100)
    num_clients = config.get('num_clients', 100)
    clients_per_round = config.get('clients_per_round', 0.1)
    local_steps = config.get('local_steps', 4)
    lr = config.get('lr', 0.01)
    weight_decay = config.get('weight_decay', 1e-4)
    seed = config.get('seed', 42)
    
    np.random.seed(seed)
    criterion = nn.CrossEntropyLoss()
    
    # Number of clients to sample each round
    m = max(1, int(clients_per_round * num_clients))
    
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
        # Client selection
        selected_clients = np.random.choice(num_clients, m, replace=False).tolist()
        
        # Run round
        train_loss, train_acc = run_fedavg_round(
            global_model, client_loaders, selected_clients,
            lr, weight_decay, device, local_steps, criterion
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(global_model, val_loader, criterion, device, show_progress=False)
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device, show_progress=False)
        
        # Log
        history['round'].append(round_idx)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(global_model.state_dict())
        
        # Callback
        if callbacks and 'on_round_end' in callbacks:
            callbacks['on_round_end'](round_idx, history, global_model)
        
        print(f"Round {round_idx}/{num_rounds} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")
    
    # Restore best model
    if best_state_dict is not None:
        global_model.load_state_dict(best_state_dict)
    
    history['best_val_acc'] = best_val_acc
    
    return history
