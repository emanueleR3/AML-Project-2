# AML Project 2: Federated Learning with DINO on CIFAR-100

> **Advanced Machine Learning** - Federated Learning with Self-Supervised Vision Transformers

## Overview

This project explores **Federated Learning** using pretrained **DINO ViT-S/16** features on **CIFAR-100**. We compare centralized baselines with FedAvg under IID and non-IID data distributions, and propose a sparse fine-tuning approach based on Fisher Information.

### Key Components
- **DINO ViT-S/16**: Self-supervised Vision Transformer (Facebook Research)
- **CIFAR-100**: 100-class image classification (60,000 images)
- **FedAvg**: Federated Averaging for distributed learning
- **Sparse FedAvg**: Communication-efficient variant with Fisher-based masking

## Project Structure

```
AML-Project-2/
├── notebooks/                      # Jupyter notebooks (Kaggle/Colab)
│   ├── kaggle_runner1.ipynb       # Central baseline + FedAvg IID + Non-IID (scaled rounds)
│   ├── kaggle_runner2.ipynb       # Sparse FedAvg experiments
│   └── kaggle_runner_scaled.ipynb # Alternative scaled rounds notebook
├── src/                           # Source code
│   ├── data.py                   # Data loading & partitioning (IID/non-IID)
│   ├── model.py                  # DINO classifier models
│   ├── utils.py                  # Utility functions
│   ├── train.py                  # Training logic (local & centralized)
│   ├── fedavg.py                 # FedAvg implementation
│   ├── sparse_fedavg.py          # Sparse FedAvg with masking
│   └── masking.py                # Fisher Information & gradient sparsification
├── output/                        # Experiment results
│   ├── main/                     # Main results (checkpoints + metrics)
│   └── figures/                  # Generated plots
├── report/                        # LaTeX report
└── requirements.txt               # Python dependencies
```

---

## Running the Experiments

### Prerequisites

1. **GPU Required**: All experiments require a GPU (Kaggle T4/P100, Colab T4, or local CUDA)
2. **Python 3.8+** with PyTorch >= 2.0

### Option 1: Kaggle (Recommended)

1. Upload the repository to Kaggle or clone from GitHub
2. Create a new notebook and attach GPU
3. Run the appropriate runner notebook

### Option 2: Google Colab

```python
# Clone repository
!git clone https://github.com/emanueleR3/AML-Project-2.git
%cd AML-Project-2
!pip install -r requirements.txt
```

### Option 3: Local Setup

```bash
git clone https://github.com/emanueleR3/AML-Project-2.git
cd AML-Project-2
pip install -r requirements.txt
```

---

## 📊 Reproducing Experiments

### Experiment 1: Central Baseline (Section 4.2)

**Notebook**: `kaggle_runner1.ipynb` (first cells)

**What it does**:
- Loads DINO ViT-S/16 pretrained backbone
- Fine-tunes classification head on CIFAR-100
- Trains for 20 epochs with AdamW + Cosine Annealing

**Expected Output**:
- `output/main/central_baseline.pt` - Best model checkpoint
- `output/main/central_baseline_metrics.json` - Training history
- **Test Accuracy**: ~68.80%

---

### Experiment 2: FedAvg IID (Section 4.2)

**Notebook**: `kaggle_runner1.ipynb` (FedAvg IID section)

**Configuration**:
| Parameter | Value |
|-----------|-------|
| Clients (K) | 100 |
| Participation (C) | 0.1 |
| Local Steps (J) | 4 |
| Rounds | 300 |

**Expected Output**:
- `output/main/fedavg_iid_metrics.json`
- `output/main/fedavg_iid_best.pt`
- **Test Accuracy**: ~57.99% (300 rounds)

---

### Experiment 3: Non-IID Sweep with Scaled Rounds (Section 4.3)

**Notebook**: `kaggle_runner1.ipynb` (Non-IID Sweep section)

**What it does**:
Sweeps over heterogeneity levels and local steps with **scaled rounds** to maintain constant total computation:

| J | Rounds | Total Steps |
|---|--------|-------------|
| 4 | 100 | 400 |
| 8 | 50 | 400 |
| 16 | 25 | 400 |

**Nc (Classes per Client)**: [1, 5, 10, 50]

**Expected Output**:
- `output/main/noniid_nc{Nc}_j{J}.json` for each configuration
- 12 total experiments (4 Nc × 3 J values)

---

### Experiment 4: Sparse FedAvg (Section 4.4)

**Notebook**: `kaggle_runner2.ipynb`

**What it does**:
- Computes Fisher Information scores on pretrained model
- Creates masks selecting least-sensitive 20% of parameters (80% sparsity)
- Runs FedAvg with sparse updates

**Configurations**:
1. **Sparse IID** - Least Sensitive mask
2. **Sparse Non-IID (Nc=1)** - Least Sensitive mask
3. **Sparse Non-IID (Nc=1)** - Random mask (baseline)

**Expected Output**:
- `output/main/sparse_fedavg_*.json`
- Sparse IID Test Accuracy: ~61.68%
- Sparse Non-IID (LS) Test Accuracy: ~35.76%

---

## ⏱️ Estimated Runtime

| Experiment | GPU (T4) | GPU (P100) |
|------------|----------|------------|
| Central Baseline (20 epochs) | ~15 min | ~10 min |
| FedAvg IID (300 rounds) | ~3 hours | ~2 hours |
| Non-IID Sweep (12 configs × 100 rounds) | ~6 hours | ~4 hours |
| Sparse FedAvg (3 configs × 100 rounds) | ~1.5 hours | ~1 hour |

---

## 📁 Output Files

### Metrics JSON Format

Each experiment saves a JSON file with:
```json
{
  "round": [1, 2, ..., N],
  "train_loss": [...],
  "train_acc": [...],
  "val_loss": [...],
  "val_acc": [...],
  "test_loss": [...],
  "test_acc": [...],
  "best_val_acc": 0.0
}
```

### Checkpoints

Model checkpoints (`.pt` files) contain:
```python
{
  "model_state_dict": model.state_dict()
}
```

To load:
```python
checkpoint = torch.load('output/main/central_baseline.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Key Parameters

### FedAvg Configuration

```python
config = {
    'num_clients': 100,        # K: Total clients
    'clients_per_round': 0.1,  # C: Fraction sampled per round
    'local_steps': 4,          # J: Local SGD steps
    'num_rounds': 100,         # Communication rounds
    'batch_size': 64,
    'lr': 0.001,               # Learning rate
    'weight_decay': 1e-4,
    'seed': 42,
    'eval_freq': 10            # Evaluate every N rounds
}
```

### Non-IID Partitioning

```python
from src.data import partition_non_iid

# Nc = number of classes per client
client_datasets = partition_non_iid(
    train_dataset, 
    num_clients=100, 
    num_classes_per_client=5,  # Nc
    seed=42
)
```

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
tqdm
pandas
```

Install with:
```bash
pip install -r requirements.txt
```

---

## References

- [DINO](https://arxiv.org/abs/2104.14294) - Self-Supervised Vision Transformers (Caron et al., 2021)
- [FedAvg](https://arxiv.org/abs/1602.05629) - Communication-Efficient Learning (McMahan et al., 2017)
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) - Dataset (Krizhevsky, 2009)

---

## License

Educational project for the Advanced Machine Learning course at Politecnico di Torino.
