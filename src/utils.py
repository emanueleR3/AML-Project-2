"""
Utility functions for DINO + CIFAR-100 Federated Learning project.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# =============================================================================
# Device Management
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_device_info() -> Dict[str, Any]:
    """Get detailed information about available devices."""
    info = {
        'device': str(get_device()),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info


# =============================================================================
# DINO Model Utilities
# =============================================================================

def load_dino_model(
    model_name: str = 'dino_vits16',
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load a DINO pretrained model from Facebook Research repository.
    
    Args:
        model_name: Options: 'dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_resnet50'
        pretrained: Whether to load pretrained weights
        device: Device to load the model on
    """
    if device is None:
        device = get_device()
    model = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model


def get_dino_embedding_dim(model_name: str = 'dino_vits16') -> int:
    """Get the embedding dimension for a DINO model."""
    dims = {
        'dino_vits16': 384, 'dino_vits8': 384,
        'dino_vitb16': 768, 'dino_vitb8': 768,
        'dino_resnet50': 2048,
    }
    return dims.get(model_name, 384)


def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Extract features from images using a DINO model."""
    if device is None:
        device = get_device()
    images = images.to(device)
    with torch.no_grad():
        features = model(images)
    return features


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module) -> None:
    """Freeze all parameters of a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all parameters of a model."""
    for param in model.parameters():
        param.requires_grad = True


# =============================================================================
# Checkpoint Utilities
# =============================================================================

def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None
) -> None:
    """Save a model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        import shutil
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load a model checkpoint."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


# =============================================================================
# Metrics
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self) -> None:
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions."""
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
