import torch
import torch.nn as nn
from typing import Dict, Literal


MaskRule = Literal[
    'least_sensitive',
    'most_sensitive',
    'random',
    'highest_magnitude',
    'lowest_magnitude'
]


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = None
) -> Dict[str, torch.Tensor]:
    
    model.eval()
    criterion = nn.CrossEntropyLoss()

    fisher = {
        name: torch.zeros_like(p)
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += p.grad.pow(2) * inputs.size(0)

        total_samples += inputs.size(0)

    for name in fisher:
        fisher[name] /= total_samples

    return fisher


def compute_sensitivity_scores(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = None,
    method: str = 'fisher'
) -> Dict[str, torch.Tensor]:

    if method != 'fisher':
        raise ValueError(f"Unknown method: {method}")

    return compute_fisher_diagonal(
        model=model,
        dataloader=dataloader,
        device=device,
        num_batches=num_batches
    )


def create_mask(
    scores: Dict[str, torch.Tensor],
    model: nn.Module,
    sparsity_ratio: float,
    rule: MaskRule = 'least_sensitive',
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    mask = {}
    
    keep_ratio = 1.0 - sparsity_ratio

    # Random case
    if rule == 'random':
        for name, p in model.named_parameters():
            if p.requires_grad:
                mask[name] = (torch.rand_like(p) < keep_ratio).float()
        return mask

    # Collect all reference values
    if rule in ['highest_magnitude', 'lowest_magnitude']:
        all_values = torch.cat([
            p.abs().flatten()
            for _, p in model.named_parameters()
            if p.requires_grad
        ])
    else:
        all_values = torch.cat([
            scores[name].flatten()
            for name in scores
        ])

    # Number of weights to KEEP
    k = int(len(all_values) * keep_ratio)
    k = max(1, k)  # Keep at least 1 weight
    
    # For least_sensitive: keep weights with LOWEST Fisher scores
    # For most_sensitive: keep weights with HIGHEST Fisher scores
    keep_largest = rule in ['most_sensitive', 'highest_magnitude']
    threshold = torch.topk(all_values, k, largest=keep_largest).values[-1]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if rule == 'least_sensitive':
            mask[name] = (scores[name] <= threshold).float()

        elif rule == 'most_sensitive':
            mask[name] = (scores[name] >= threshold).float()

        elif rule == 'lowest_magnitude':
            mask[name] = (p.abs() <= threshold).float()

        elif rule == 'highest_magnitude':
            mask[name] = (p.abs() >= threshold).float()

    return mask


def get_mask_sparsity(mask: Dict[str, torch.Tensor]) -> float:
    total = 0
    active = 0

    for m in mask.values():
        total += m.numel()
        active += m.sum().item()

    return 1.0 - (active / total)  

def save_mask(mask: Dict[str, torch.Tensor], path: str):
    torch.save(mask, path)


def load_mask(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path)
