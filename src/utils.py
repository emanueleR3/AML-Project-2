import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_parameters(model: nn.Module, trainable_only: bool = False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(state: Dict[str, Any], filepath: str):
    ensure_dir(os.path.dirname(filepath))
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: nn.Module, optimizer=None, device=None):
    if device is None:
        device = get_device()
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def save_metrics_json(filepath: str, metrics: Dict):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
