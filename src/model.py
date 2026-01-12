import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, Optional

from .utils import count_parameters, freeze_model, get_device, unfreeze_model


DINO_EMBED_DIMS = {
    'dino_vits16': 384, 'dino_vits8': 384,
    'dino_vitb16': 768, 'dino_vitb8': 768,
    'dino_resnet50': 2048,
}


def load_dino_backbone(model_name: str = 'dino_vits16', device=None):
    if device is None:
        device = get_device()
    model = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    return model


def _unfreeze_vit_last_blocks(backbone: nn.Module, last_n_blocks: int) -> None:
    if last_n_blocks <= 0:
        return
    if not hasattr(backbone, 'blocks'):
        return
    blocks = getattr(backbone, 'blocks')
    try:
        n_blocks = len(blocks)
    except Exception:
        return
    start = max(0, n_blocks - int(last_n_blocks))
    for b in blocks[start:]:
        for p in b.parameters():
            p.requires_grad = True
    if hasattr(backbone, 'norm'):
        for p in backbone.norm.parameters():
            p.requires_grad = True


def _unfreeze_resnet_last_layers(backbone: nn.Module, last_n_blocks: int) -> None:
    if last_n_blocks <= 0:
        return
    layers = []
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(backbone, name):
            layers.append(getattr(backbone, name))
    if not layers:
        return
    start = max(0, len(layers) - int(last_n_blocks))
    for layer in layers[start:]:
        for p in layer.parameters():
            p.requires_grad = True


def apply_freeze_policy(
    backbone: nn.Module,
    freeze_policy: str = 'head_only',
    last_n_blocks: int = 1,
) -> bool:
    policy = (freeze_policy or 'head_only').lower()

    if policy == 'head_only':
        freeze_model(backbone)
        return True

    if policy == 'finetune_all':
        unfreeze_model(backbone)
        return False

    if policy == 'last_blocks_only':
        freeze_model(backbone)
        _unfreeze_vit_last_blocks(backbone, last_n_blocks=last_n_blocks)
        _unfreeze_resnet_last_layers(backbone, last_n_blocks=last_n_blocks)
        return False

    raise ValueError("freeze_policy must be one of: 'head_only', 'finetune_all', 'last_blocks_only'")


class DINOClassifier(nn.Module):
    def __init__(self, model_name: str = 'dino_vits16', num_classes: int = 100,
                 freeze_backbone: bool = True, dropout: float = 0.1, device=None,
                 freeze_policy: Optional[str] = None, last_n_blocks: int = 1):
        super().__init__()
        self.backbone = load_dino_backbone(model_name, device)

        if freeze_policy is not None:
            self.freeze_backbone = apply_freeze_policy(self.backbone, freeze_policy, last_n_blocks=last_n_blocks)
        else:
            self.freeze_backbone = bool(freeze_backbone)
            if self.freeze_backbone:
                freeze_model(self.backbone)
        
        embed_dim = DINO_EMBED_DIMS.get(model_name, 384)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.backbone(x)
        return self.classifier(features)
    
    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def get_trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return int(count_parameters(model, trainable_only=trainable_only))


def build_model(config: Dict[str, Any]) -> DINOClassifier:
    model_name = config.get('model_name', 'dino_vits16')
    num_classes = int(config.get('num_classes', 100))
    dropout = float(config.get('dropout', 0.1))
    freeze_policy = config.get('freeze_policy', 'head_only')
    last_n_blocks = int(config.get('last_n_blocks', 1))
    device = config.get('device', None)

    return DINOClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        device=device,
        freeze_policy=freeze_policy,
        last_n_blocks=last_n_blocks,
    )
