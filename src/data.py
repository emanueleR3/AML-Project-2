"""
Data loading and preprocessing for CIFAR-100 with DINO.
"""

import os
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms


# Normalization constants
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_dino_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """Get transforms for DINO model input (224x224, ImageNet normalization)."""
    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_cifar100(
    data_dir: str = './data',
    download: bool = True,
    use_dino_transforms: bool = True,
    image_size: int = 224
) -> Tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100]:
    """Load CIFAR-100 dataset with appropriate transforms."""
    if use_dino_transforms:
        train_transform = get_dino_transforms(image_size, is_training=True)
        test_transform = get_dino_transforms(image_size, is_training=False)
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=download, transform=test_transform
    )
    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and testing."""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


# =============================================================================
# Federated Learning Data Partitioning
# =============================================================================

def partition_dataset_iid(dataset: Dataset, num_clients: int, seed: int = 42) -> List[Subset]:
    """Partition dataset into IID subsets for federated learning."""
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    split_indices = np.array_split(indices, num_clients)
    return [Subset(dataset, idx.tolist()) for idx in split_indices]


def partition_dataset_non_iid(
    dataset: Dataset,
    num_clients: int,
    num_classes_per_client: int = 10,
    seed: int = 42
) -> List[Subset]:
    """Partition dataset into non-IID subsets (each client gets subset of classes)."""
    np.random.seed(seed)
    
    labels = np.array(dataset.targets if hasattr(dataset, 'targets') 
                      else [dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    client_subsets = []
    for _ in range(num_clients):
        client_classes = np.random.choice(num_classes, size=num_classes_per_client, replace=False)
        client_indices = []
        for cls in client_classes:
            cls_idx = class_indices[cls]
            n_samples = len(cls_idx) // (num_clients // (num_classes // num_classes_per_client))
            selected = np.random.choice(cls_idx, size=min(n_samples, len(cls_idx)), replace=False)
            client_indices.extend(selected.tolist())
        client_subsets.append(Subset(dataset, client_indices))
    
    return client_subsets


def get_client_dataloader(client_dataset: Subset, batch_size: int = 32, num_workers: int = 2) -> DataLoader:
    """Create a DataLoader for a client's local dataset."""
    return DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
