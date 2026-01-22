# Advanced Machine Learning Project 2: Federated Learning with DINO on CIFAR-100

## 1. Project Overview

This project investigates the application of Federated Learning (FL) techniques using self-supervised Vision Transformer (ViT) features. Specifically, we utilize features extracted from a DINO ViT-S/16 model pre-trained on ImageNet. The primary objective is to evaluate the performance of the Federated Averaging (FedAvg) algorithm on the CIFAR-100 dataset under various conditions, including Independent and Identically Distributed (IID) and Non-IID data partitions.

Furthermore, this work explores communication-efficient training through sparse fine-tuning. We implement and analyze a method based on Task Arithmetic, employing Fisher Information matrices to construct gradient masks. This allows for updating only a subset of significant parameters, thereby reducing communication overhead. An extension of this work compares different parameter selection strategies for gradient masking in non-IID settings.

## 2. Repository Structure

The repository is organized as follows:

```
AML-Project-2/
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── kaggle_runner1.ipynb    # Baseline and standard FedAvg experiments
│   └── kaggle_runner2.ipynb    # Sparse FedAvg and ablation studies
├── src/                        # Source code modules
│   ├── data.py                 # Dataset loading and partitioning logic
│   ├── model.py                # Model architecture definitions
│   ├── train.py                # Local and centralized training loops
│   ├── fedavg.py               # Federated Averaging implementation
│   ├── sparse_fedavg.py        # Sparse FedAvg implementation
│   ├── optim.py                # Custom optimizer (SparseSGDM)
│   ├── masking.py              # Gradient masking and Fisher information utilities
│   └── utils.py                # General utility functions
├── output/                     # Directory for experiment artifacts
│   ├── main/                   # Results for dense baselines
│   ├── sparse/                 # Results for sparse experiments
│   └── figures/                # Generated visualization plots
├── report/                     # LaTeX source for the final report
└── requirements.txt            # Python dependency specification
```

## 3. Methodology and Experiments

The experimental campaign addresses the following key areas as defined in the project specifications:

### 3.1. Centralized Baseline
We establish a centralized baseline by training a linear classifier on top of frozen DINO features. The training process involves:
- **Optimizer**: SGDM (Stochastic Gradient Descent with Momentum).
- **Scheduler**: Cosine Annealing, selected after a comparative analysis of learning rate schedulers.
- **Goal**: To determine the optimal hyperparameters and convergence behavior (epochs required) for the downstream task on CIFAR-100.

### 3.2. Federated Averaging (FedAvg) Baseline
We implement the FedAvg algorithm [McMahan et al., 2017] with the following configuration:
- **Clients ($K$)**: 100
- **Sampling Fraction ($C$)**: 0.1 (10 clients per round)
- **Local Steps ($J$)**: 4
- **Data Distribution**: IID sharding of CIFAR-100.

### 3.3. Heterogeneity Analysis
To simulate realistic federated settings, we evaluate performance under statistical heterogeneity (Non-IID data).
- **Label Distribution**: We vary the number of classes per client ($N_c \in \{1, 5, 10, 50\}$).
- **Local Computation**: We analyze the impact of local training steps ($J \in \{4, 8, 16\}$) on convergence and accuracy.
- **Note**: The number of communication rounds is scaled inversely with $J$ to maintain a constant total computation budget across comparisons.

### 3.4. Sparse Federated Learning (Task Arithmetic)
We implement a communication-efficient variation of FedAvg using sparse updates.
- **Method**: Computes the Fisher Information Matrix to estimate parameter sensitivity.
- **Masking**: A binary mask is calibrated to freeze "least-sensitive" parameters, allowing only the most informative weights to be updated.
- **Optimizer**: A custom `SparseSGDM` optimizer is implemented to respect the gradient masks during local training.
- **Ablation Studies**: We analyze the effect of varying the sparsity ratio and the number of calibration rounds.

### 3.5. Extension: Masking Rules Comparison
We extend the sparse training methodology by comparing different criteria for parameter selection in Non-IID settings ($N_c=1$). The masking rules evaluated are:
1.  **Least Sensitive**: Standard approach (lowest Fisher information).
2.  **Most Sensitive**: Updating only parameters with the highest Fisher information.
3.  **Lowest Magnitude**: Pruning based on weight magnitude (smallest absolute value).
4.  **Highest Magnitude**: Pruning based on weight magnitude (largest absolute value).
5.  **Random**: Random selection of parameters.

## 4. Reproducing Results

Experiments are designed to be run in a GPU-accelerated environment. The workflows are encapsulated in Jupyter notebooks for reproducibility.

### Prerequisites
- Python 3.8 or higher
- PyTorch $\ge$ 2.0
- CUDA-enabled GPU (Recommended: NVIDIA T4 or better)

To install dependencies:
```bash
pip install -r requirements.txt
```

### Execution Steps

#### Part 1: Baselines and Dense FedAvg
1.  **Pre-training Head**: Run `notebooks/pretrain_head.ipynb`.
    *   **Goal**: Train a linear classifier on the frozen DINO backbone (required initialization).
    *   **Output**: `output/main/pretrained_head.pt`, `output/main/pretrained_head_metrics.json`

2.  **Central Baseline**: Run `notebooks/central_baseline.ipynb`.
    *   **Goal**: Train the centralized baseline model (fine-tuning backbone).
    *   **Output**: `output/main/central_baseline.pt`, `output/main/central_baseline_metrics.json`

3.  **IID FedAvg**: Run `notebooks/fedavg_iid.ipynb`.
    *   **Goal**: Run standard FedAvg on IID data.
    *   **Output**: `output/main/fedavg_iid_best.pt`, `output/main/fedavg_iid_metrics.json`

4.  **Non-IID Sweep**: Run `notebooks/scaled_noniid.ipynb`.
    *   **Goal**: Run FedAvg on Non-IID data with varying $N_c$ and $J$ (scaled rounds).
    *   **Output**: `output/scaled/noniid_nc{Nc}_j{J}.json`

#### Part 2: Sparse Federated Learning and Extensions
1.  **Ablation Studies**: Run `notebooks/sparse_ablation.ipynb`.
    *   **Goal**: Determine optimal calibration rounds and sparsity ratio.
    *   **Output**: `output/sparse_ablation/ablation_calib{N}.json`, `output/sparse_ablation/ablation_sparsity{N}.json`

2.  **Sparse IID Comparison**: Run `notebooks/sparse_fedavg_iid.ipynb`.
    *   **Goal**: Compare Sparse FedAvg vs Dense FedAvg on IID data.
    *   **Output**: `output/comparison/sparse_iid_metrics.json`

3.  **Masking Rules Comparison**: Run `notebooks/sparse_noniid_sweep.ipynb`.
    *   **Goal**: Compare 5 masking rules (Least Sensitive, Most Sensitive, Lowest Magnitude, Highest Magnitude, Random) on Non-IID data.
    *   **Output**: `output/sparse_noniid_{rule}/sparse_nc{nc}_j{j}_{rule}.json`

### Analysis and Plotting
To generate the tables and figures used in the report, run the analysis script after completing the experiments:
```bash
cd output
python final_analysis.py
python generate_results.py
```
This will produce PDF plots in `output/figures/` and LaTeX tables for the report.
