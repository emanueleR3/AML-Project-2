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
│   └── kaggle_runner2.ipynb       # Sparse FedAvg: ablations + all 5 mask rules
├── src/                           # Source code
│   ├── data.py                   # Data loading & partitioning (IID/non-IID)
│   ├── model.py                  # DINO classifier models
│   ├── utils.py                  # Utility functions
│   ├── train.py                  # Training logic (local & centralized)
│   ├── fedavg.py                 # FedAvg implementation
│   ├── sparse_fedavg.py          # Sparse FedAvg with masking
│   ├── optim.py                  # SparseSGDM optimizer
│   └── masking.py                # Fisher Information & gradient sparsification
├── output/                        # Experiment results
│   ├── main/                     # Dense FedAvg results (baselines + non-IID)
│   ├── extended/                 # Extended training results (300 rounds)
│   ├── sparse/                   # Sparse FedAvg results (all mask rules)
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

---

## Reproducing Experiments

### Notebook 1: `kaggle_runner1.ipynb` (~6 hours)

**Contains:**
- Central Baseline (20 epochs)
- FedAvg IID (100-300 rounds)
- Non-IID Sweep with **Scaled Rounds**

**Scaled Rounds Configuration:**
When increasing local steps J, rounds are scaled inversely to keep total computation constant:

| J | Rounds | Total Steps |
|---|--------|-------------|
| 4 | 100 | 400 |
| 8 | 50 | 400 |
| 16 | 25 | 400 |

**Nc Values**: 1, 5, 10, 50 (classes per client)

**Output:**
- `output/main/central_baseline.pt` - Pretrained baseline
- `output/main/fedavg_iid_metrics.json` - IID results
- `output/main/noniid_nc{Nc}_j{J}.json` - Non-IID results

---

### Notebook 2: `kaggle_runner2.ipynb` (~10.5 hours)

**Phase 1: Ablation Studies (50 rounds each)**

1. **Calibration Rounds Sweep**: 1, 3, 5, 10 rounds
   - Multi-round Fisher Information calibration (Paper [15] Sec. 4.2)
   
2. **Sparsity Ratio Sweep**: 60%, 70%, 80%, 90%

**Phase 2: Final Experiments (100 rounds each)**

All 5 mask rules tested on Non-IID (Nc=1) data:

| Mask Rule | Description |
|-----------|-------------|
| `least_sensitive` | Lowest Fisher Information scores |
| `most_sensitive` | Highest Fisher Information scores |
| `lowest_magnitude` | Smallest absolute weights |
| `highest_magnitude` | Largest absolute weights |
| `random` | Random parameter selection |

Plus: IID + Least Sensitive baseline

**Output:**
- `output/sparse/ablation_calib{N}.json` - Calibration ablations
- `output/sparse/ablation_sparsity{N}.json` - Sparsity ablations
- `output/sparse/exp_iid_ls.json` - IID sparse results
- `output/sparse/exp_noniid_{rule}.json` - Non-IID with each mask rule
- `output/sparse/complete_summary.json` - Combined summary

---

## Estimated Runtime

| Experiment | GPU (T4) | GPU (P100) |
|------------|----------|------------|
| Central Baseline (20 epochs) | ~15 min | ~10 min |
| FedAvg IID (300 rounds) | ~3 hours | ~2 hours |
| Non-IID Sweep (12 configs, scaled) | ~3 hours | ~2 hours |
| Sparse Ablations (8 configs × 50 rounds) | ~4 hours | ~3 hours |
| Sparse Final (6 configs × 100 rounds) | ~6 hours | ~4 hours |

---

## Analysis Script

After running experiments, generate figures and tables:

```bash
cd output
python final_analysis.py
```

**Generates:**
- `figures/central_baseline_curves.pdf`
- `figures/fedavg_iid_convergence.pdf`
- `figures/noniid_heatmap.pdf`
- `figures/sparse_fedavg_comparison.pdf`
- `figures/ablation_studies.pdf`
- `summary_results.csv`
- `summary_table.tex`
- `noniid_table.tex`

---

## Key Parameters

### FedAvg Configuration

```python
config = {
    'num_clients': 100,        # K: Total clients
    'clients_per_round': 0.1,  # C: Fraction sampled per round (10 clients)
    'local_steps': 4,          # J: Local SGD steps
    'num_rounds': 100,         # Communication rounds
    'batch_size': 64,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'seed': 42,
    'eval_freq': 10
}
```

### Sparse FedAvg Configuration

```python
OPTIMAL_CALIB = 3      # Calibration rounds (from ablation)
OPTIMAL_SPARSITY = 0.8 # 80% sparsity (20% params updated)

MASK_RULES = [
    'least_sensitive',
    'most_sensitive',
    'lowest_magnitude',
    'highest_magnitude',
    'random'
]
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
seaborn
```

Install:
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
