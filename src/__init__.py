"""
AML Project 2: DINO + CIFAR-100 Federated Learning
"""

from .utils import (
    set_seed, get_device, get_device_info,
    load_dino_model, get_dino_embedding_dim, extract_features,
    count_parameters, freeze_model, unfreeze_model,
    save_checkpoint, load_checkpoint,
    AverageMeter, accuracy
)

from .data import (
    CIFAR100_MEAN, CIFAR100_STD, IMAGENET_MEAN, IMAGENET_STD,
    get_dino_transforms, load_cifar100, create_dataloaders,
    partition_dataset_iid, partition_dataset_non_iid, get_client_dataloader
)

from .model import (
    DINOClassifier, LinearProbe, MLPHead, create_dino_classifier
)

__version__ = '0.1.0'
