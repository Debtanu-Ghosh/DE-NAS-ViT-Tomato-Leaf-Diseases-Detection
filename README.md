# Differential Evolution-Based Neural Architecture Search (DE-NAS) for Vision Transformers (ViT) on tomato leaf diseases detection

This repository contains the implementation of a **Differential Evolution (DE)**-based Neural Architecture Search (NAS) framework for optimizing Vision Transformer (ViT) architectures. The proposed approach efficiently explores the hyperparameter search space to discover optimal configurations for ViTs, leveraging the power of DE to balance exploration and exploitation.

---

## 📋 Features
- **Hyperparameter Search for ViT**: Search space includes patch size, embedding size, number of layers, attention heads, dropout rate, activation functions, and feed-forward dimension (d_ff).
- **Differential Evolution**: Implements DE with customizable scaling factor (F) and crossover rate (CR).
- **Fitness Evaluation**: Utilizes cross-entropy loss to evaluate architectures' performance.
- **Configurable Parameters**: Easily customize the population size, number of generations, and search constraints.

---

## 🛠️ Methodology

1. **Population Initialization**: A population of \(NP\) individuals, each representing a unique ViT configuration, is randomly initialized within the predefined search space.

2. **Fitness Evaluation**: The fitness of each individual is calculated using cross-entropy loss on a given dataset.

3. **Mutation and Crossover**:
   - Three random individuals are selected to calculate the donor vector (\(u_1^g\)) using the formula:
     \[
     u_1^g = X_{r_1} + F \cdot (X_{r_2} - X_{r_3})
     \]
   - A trial vector (\(v_1^g\)) is generated by performing crossover between the donor vector and the target vector.

4. **Selection**: The individual with the lower fitness (target or trial vector) is carried forward to the next generation.

5. **Termination**: The process continues until the maximum number of generations is reached or a convergence criterion is satisfied.

---

## 🔧 Hyperparameter Search Space

| Parameter       | Search Space                           |
|-----------------|----------------------------------------|
| patch\_size     | \{4, 8, 16\}                          |
| emb\_size       | \{32, 64, 128, 256, 512, 768\}        |
| num\_layers     | [4, 16]                               |
| num\_head       | \{2, 4, 8, 16\}                       |
| DR (Dropout Rate)| [0.1, 0.5]                            |
| acf (Activation)| \{relu, gelu\}      |
| d\_ff (Feed Forward Dim) | \{128, 256, 512, 1024, 3072\} |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `numpy`, `torch`, `transformers`, `matplotlib`, `torchvision`, `einops`, `Pillow`, `tqdm`, `accelerate`, `torchinfo`, `datasets`, `scikit-learn`

---

DE-NAS-ViT-Tomato-Leaf-Diseases-Detection/

      ├── tomato/New Plant Diseases Dataset(Augmented)/                     # Dataset folder
      ├── models.ipynb/                                                     # Main code
      ├── results/                                                          # Results imgaes
      ├── README.md
