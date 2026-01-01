import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, List


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(image_size: int = 224, is_training: bool = True):
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


def load_cifar100(data_dir: str = './data', image_size: int = 224, download: bool = True):
    """
    Load CIFAR-100 dataset with DINO-compatible transforms.
    
    Args:
        data_dir: Directory to store/load the dataset
        image_size: Size to resize images to (default 224 for DINO)
        download: Whether to download the dataset if not present (default True)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=download,
        transform=get_transforms(image_size, is_training=True)
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=download,
        transform=get_transforms(image_size, is_training=False)
    )
    return train_dataset, test_dataset


def create_dataloader(dataset: Dataset, batch_size: int = 64, shuffle: bool = True, num_workers: int = 4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=shuffle)


def partition_iid(dataset: Dataset, num_clients: int, seed: int = 42):
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    return [Subset(dataset, idx.tolist()) for idx in np.array_split(indices, num_clients)]


def partition_non_iid(dataset: Dataset, num_clients: int, num_classes_per_client: int, seed: int = 42):
    np.random.seed(seed)
    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        targets = np.array(dataset.targets)
    
    num_classes = len(np.unique(targets))
    class_indices = {i: np.where(targets == i)[0] for i in range(num_classes)}
    
    client_subsets = []
    for _ in range(num_clients):
        client_classes = np.random.choice(num_classes, num_classes_per_client, replace=False)
        client_indices = []
        for cls in client_classes:
            cls_idx = class_indices[cls]
            n_samples = len(cls_idx) // (num_clients // (num_classes // num_classes_per_client))
            selected = np.random.choice(cls_idx, min(n_samples, len(cls_idx)), replace=False)
            client_indices.extend(selected.tolist())
        client_subsets.append(Subset(dataset, client_indices))
    return client_subsets
