# Differential Evolution-Based Neural Architecture Search (DE-NAS) for Vision Transformers (ViT) on tomato leaf diseases detection

This repository contains the implementation of a **Differential Evolution (DE)**-based Neural Architecture Search (NAS) framework for optimizing Vision Transformer (ViT) architectures. The proposed approach efficiently explores the hyperparameter search space to discover optimal configurations for ViTs, leveraging the power of DE to balance exploration and exploitation.

---

## 📋 Features
- **Hyperparameter Search for ViT**: Search space includes patch size, embedding size, number of layers, attention heads, dropout rate, activation functions, and feed-forward dimension (d_ff).
- **Differential Evolution**: Implements DE with customizable scaling factor (F) and crossover rate (CR).
- **Fitness Evaluation**: Utilizes cross-entropy loss to evaluate architectures' performance.
- **Configurable Parameters**: Easily customize the population size, number of generations, and search constraints.

---
