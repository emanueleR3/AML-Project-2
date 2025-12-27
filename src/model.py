import torch
import torch.nn as nn
from .utils import get_device, freeze_model


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


class DINOClassifier(nn.Module):
    def __init__(self, model_name: str = 'dino_vits16', num_classes: int = 100,
                 freeze_backbone: bool = True, dropout: float = 0.1, device=None):
        super().__init__()
        self.backbone = load_dino_backbone(model_name, device)
        if freeze_backbone:
            freeze_model(self.backbone)
        
        embed_dim = DINO_EMBED_DIMS.get(model_name, 384)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        self.freeze_backbone = freeze_backbone
    
    def forward(self, x: torch.Tensor):
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.backbone(x)
        return self.classifier(features)
    
    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
