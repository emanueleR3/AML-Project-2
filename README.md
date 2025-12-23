# AML Project 2: Federated Learning with DINO on CIFAR-100

> **Advanced Machine Learning** - Federated Learning with Self-Supervised Vision Transformers

## 📋 Overview

This project explores **Federated Learning** using pretrained **DINO ViT-S/16** features on **CIFAR-100**. We compare centralized baselines with FedAvg under IID and non-IID data distributions.

### Key Components
- 🦕 **DINO ViT-S/16**: Self-supervised Vision Transformer (Facebook Research)
- 📊 **CIFAR-100**: 100-class image classification (60,000 images)
- 🔄 **FedAvg**: Federated Averaging for distributed learning
- 📉 **Sparse FedAvg**: Communication-efficient variant

## 🏗️ Project Structure

```
AML-Project-2/
├── notebooks/                  # Jupyter notebooks (run on Colab)
│   ├── 00_setup_colab.ipynb   # ⚙️ Environment setup
│   ├── 02_central_baseline.ipynb
│   ├── 03_fedavg_iid.ipynb
│   ├── 04_fedavg_noniid_sweep.ipynb
│   ├── 06_sparse_fedavg.ipynb
│   ├── 07_extension_mask_rules.ipynb
│   └── 99_make_plots_for_report.ipynb
├── src/                        # Source code
│   ├── data.py                # Data loading & partitioning
│   ├── model.py               # DINO classifier models
│   ├── utils.py               # Utility functions
│   ├── train.py               # Training logic
│   ├── fedavg.py              # FedAvg implementation
│   └── masking.py             # Gradient sparsification
├── configs/                    # YAML configurations
├── colab/                      # Colab-specific files
├── report/                     # Final report
└── outputs/                    # Results & checkpoints
```

## 🚀 Quick Start (Google Colab)

### 1. Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload or clone the repository to your Google Drive:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository (or upload manually)
!git clone https://github.com/yourusername/AML-Project-2.git
%cd AML-Project-2
```

### 2. Install Dependencies

```python
!pip install -r requirements.txt
```

### 3. Run Setup Notebook

Start with `00_setup_colab.ipynb` to download DINO and CIFAR-100:

```python
import torch
import torchvision

# Load DINO ViT-S/16 from Facebook Research
dino_vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

# Download CIFAR-100
train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
```

### 4. Enable GPU

⚠️ **Important**: Enable GPU runtime for faster training:
- `Runtime` → `Change runtime type` → `T4 GPU` (or better)

## 📊 Experiments

| Notebook | Description | Colab GPU |
|----------|-------------|-----------|
| `00_setup_colab.ipynb` | Setup: download DINO & CIFAR-100 | Optional |
| `02_central_baseline.ipynb` | Centralized training baseline | ✅ Required |
| `03_fedavg_iid.ipynb` | FedAvg with IID data | ✅ Required |
| `04_fedavg_noniid_sweep.ipynb` | FedAvg with Non-IID data | ✅ Required |
| `06_sparse_fedavg.ipynb` | Communication-efficient FedAvg | ✅ Required |
| `07_extension_mask_rules.ipynb` | Custom masking strategies | ✅ Required |
| `99_make_plots_for_report.ipynb` | Generate report figures | Optional |

## 💾 Saving Results on Colab

To persist results across sessions, save to Google Drive:

```python
import shutil

# Save outputs to Drive
shutil.copytree('outputs/', '/content/drive/MyDrive/AML-Project-2/outputs/')
```

## 📦 Dependencies

- PyTorch >= 2.0.0
- Torchvision >= 0.15.0
- NumPy, Matplotlib, scikit-learn
- tqdm, tensorboard

## 📖 References

- [DINO](https://arxiv.org/abs/2104.14294) - Self-Supervised Vision Transformers
- [FedAvg](https://arxiv.org/abs/1602.05629) - Federated Learning
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) - Dataset

## 📄 License

Educational project for the Advanced Machine Learning course.
