import copy
import torch
import torch.nn as nn
from src.fedavg import fedavg_aggregate
from src.optim import SparseSGDM

from .train import local_train, evaluate
from .utils import AverageMeter

def client_update_sparse(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    lr: float,
    weight_decay: float,
    device: torch.device,
    local_steps: int,
    mask: dict,
    criterion=None
):
    """Performs local training using SparseSGDM."""
    local_model = copy.deepcopy(model)
    local_model.to(device)
    local_model.train()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
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
    client_loaders: list,
    selected_clients: list,
    lr: float,
    weight_decay: float,
    device: torch.device,
    local_steps: int,
    mask: dict,
    criterion=None
):
    client_state_dicts = []
    client_weights = []
    round_loss = AverageMeter()
    round_acc = AverageMeter()
    
    for client_idx in selected_clients:
        loader = client_loaders[client_idx]
        
        state_dict, loss, acc, n_samples = client_update_sparse(
            global_model, loader, lr, weight_decay, device, local_steps, mask, criterion
        )
        
        client_state_dicts.append(state_dict)
        client_weights.append(n_samples)
        round_loss.update(loss, n_samples)
        round_acc.update(acc, n_samples)
    
    fedavg_aggregate(global_model, client_state_dicts, client_weights)
    
    return round_loss.avg, round_acc.avg