"""Training utilities for Federated Learning experiments."""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

from .utils import AverageMeter, accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True
) -> Tuple[float, float]:
    """
    Trains the model for one full epoch.
    
    Args:
        model: Neural network model
        loader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to use for training
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    iterator = tqdm(loader, desc='Train', leave=False) if show_progress else loader
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        acc1 = accuracy(outputs, labels, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
        
        if show_progress:
            iterator.set_postfix(loss=f'{loss_meter.avg:.4f}', acc=f'{acc_meter.avg:.2f}%')
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True
) -> Tuple[float, float]:
    """
    Evaluates the model on the given data loader.
    
    Args:
        model: Neural network model
        loader: Evaluation data loader
        criterion: Loss function
        device: Device to use for evaluation
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    iterator = tqdm(loader, desc='Eval', leave=False) if show_progress else loader
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        acc1 = accuracy(outputs, labels, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
    
    return loss_meter.avg, acc_meter.avg


def local_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    local_steps: int
) -> Tuple[float, float, int]:
    """
    Performs J local training steps (for Federated Learning).
    
    Args:
        model: Neural network model
        loader: Client's training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to use for training
        local_steps: Number of local steps (J)
        
    Returns:
        Tuple of (average loss, average accuracy, number of samples processed)
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_samples = 0
    
    step = 0
    data_iter = iter(loader)
    
    while step < local_steps:
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        acc1 = accuracy(outputs, labels, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
        total_samples += images.size(0)
        
        step += 1
    
    return loss_meter.avg, acc_meter.avg, total_samples
