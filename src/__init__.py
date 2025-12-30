
from .utils import (
    set_seed, get_device, ensure_dir, 
    count_parameters, freeze_model, unfreeze_model,
    save_checkpoint, load_checkpoint, save_metrics_json,
    AverageMeter, accuracy
)

from .data import (
    get_transforms, load_cifar100, create_dataloader, partition_iid, partition_non_iid
)

from .model import (
    load_dino_backbone,
    DINOClassifier,
    apply_freeze_policy,
    get_trainable_params,
    count_params,
    build_model,
)

from .train import (
    train_one_epoch,
    evaluate,
    local_train,
)

from .fedavg import (
    client_update,
    fedavg_aggregate,
    run_fedavg_round,
    run_fedavg,
)

__version__ = '0.1.0'
