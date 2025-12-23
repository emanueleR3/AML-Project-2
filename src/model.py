"""
Model definitions for DINO-based classification on CIFAR-100.
"""

import torch
import torch.nn as nn
from typing import Optional
from .utils import load_dino_model, get_dino_embedding_dim, freeze_model


class DINOClassifier(nn.Module):
    """Classifier built on top of frozen DINO features."""
    
    def __init__(
        self,
        dino_model_name: str = 'dino_vits16',
        num_classes: int = 100,
        freeze_backbone: bool = True,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.freeze_backbone = freeze_backbone
        
        # Load DINO backbone
        self.backbone = load_dino_model(dino_model_name, pretrained=True, device=device)
        if freeze_backbone:
            freeze_model(self.backbone)
        
        self.embed_dim = get_dino_embedding_dim(dino_model_name)
        
        # Classification head
        if hidden_dim:
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x)


class LinearProbe(nn.Module):
    """Simple linear probe for evaluating DINO features."""
    
    def __init__(self, embed_dim: int = 384, num_classes: int = 100):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPHead(nn.Module):
    """MLP classification head."""
    
    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 512,
        num_classes: int = 100,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        in_dim = embed_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def create_dino_classifier(
    num_classes: int = 100,
    dino_model: str = 'dino_vits16',
    classifier_type: str = 'linear',
    hidden_dim: int = 512,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
) -> DINOClassifier:
    """Factory function to create a DINO classifier."""
    return DINOClassifier(
        dino_model_name=dino_model,
        num_classes=num_classes,
        freeze_backbone=True,
        hidden_dim=hidden_dim if classifier_type == 'mlp' else None,
        dropout=dropout,
        device=device
    )
