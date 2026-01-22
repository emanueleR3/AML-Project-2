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
    criterion: Optional[nn.Module] = None,
    mask: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[Dict[str, torch.Tensor], float, float, int]:
    local_model = copy.deepcopy(model)
    local_model.to(device)
    local_model.train()
    
    handles = []
    if mask is not None:
        def get_mask_hook(mask_tensor):
            def hook(grad):
                return grad * mask_tensor.to(grad.device)
            return hook
            
        for n, p in local_model.named_parameters():
            if n in mask and p.requires_grad:
                h = p.register_hook(get_mask_hook(mask[n]))
                handles.append(h)
    
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
    
    for h in handles: h.remove()
    
    return local_model.state_dict(), avg_loss, avg_acc, n_samples


def fedavg_aggregate(
    global_model: nn.Module,
    client_state_dicts: List[Dict[str, torch.Tensor]],
    client_weights: List[float]
) -> None:
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    global_state = global_model.state_dict()
    aggregated_state = {k: torch.zeros_like(v) for k, v in global_state.items()}
    
    for state_dict, weight in zip(client_state_dicts, normalized_weights):
        for key in aggregated_state:
            aggregated_state[key] += weight * state_dict[key].to(aggregated_state[key].device)
    
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
    show_progress: bool = False,
    mask: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[float, float]:
    client_state_dicts = []
    client_weights = []
    round_loss = AverageMeter()
    round_acc = AverageMeter()
    
    iterator = tqdm(selected_clients, desc='Clients', leave=False) if show_progress else selected_clients
    
    for client_idx in iterator:
        loader = client_loaders[client_idx]
        
        state_dict, loss, acc, n_samples = client_update(
            global_model, loader, lr, weight_decay, device, local_steps, criterion, mask
        )
        
        client_state_dicts.append(state_dict)
        client_weights.append(n_samples)
        round_loss.update(loss, n_samples)
        round_acc.update(acc, n_samples)
    
    fedavg_aggregate(global_model, client_state_dicts, client_weights)
    
    return round_loss.avg, round_acc.avg


def run_fedavg(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    callbacks: Optional[Dict[str, callable]] = None,
    mask: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, List[float]]:
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
        selected_clients = np.random.choice(num_clients, m, replace=False).tolist()
        
        train_loss, train_acc = run_fedavg_round(
            global_model, client_loaders, selected_clients,
            lr, weight_decay, device, local_steps, criterion, False, mask
        )
        
        if round_idx % eval_freq == 0 or round_idx == num_rounds:
            val_loss, val_acc = evaluate(global_model, val_loader, criterion, device, show_progress=False)
            test_loss, test_acc = evaluate(global_model, test_loader, criterion, device, show_progress=False)
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
            best_state_dict = copy.deepcopy(global_model.state_dict())
        
        if callbacks and 'on_round_end' in callbacks:
            callbacks['on_round_end'](round_idx, history, global_model)
        
        print(f"Round {round_idx}/{num_rounds} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")
    
    if best_state_dict is not None:
        global_model.load_state_dict(best_state_dict)
    
    history['best_val_acc'] = best_val_acc
    
    return history
