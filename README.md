# Fracture Detection

This project explores nonlinear AI methods for image classification through two complementary tasks:

1. **Multiclass object classification** on CIFAR-10.
2. **Binary fracture detection** on X-ray images with class imbalance.

The goal is to compare multiple model families, justify design choices, and measure performance with relevant metrics.

## Project Objectives

- Build and compare classical ML and deep learning pipelines.
- Evaluate model quality with robust, task-specific metrics.
- Handle imbalanced data using weighting, sampling, and augmentation.
- Test generalization on data not seen during training.

---

## Part 1 - CIFAR-10 Classification

### Dataset
- **Name:** CIFAR-10 (TensorFlow/Keras dataset)
- **Size:** 60,000 RGB images
- **Resolution:** 32x32
- **Split:** 50,000 training / 10,000 test
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Required Comparisons
- At least **2 classical ML algorithms** (examples: SVM, Random Forest, k-NN).
- A **CNN family** with **3 architecture variants**.
- One **hybrid CNN approach** (for example: CNN feature extractor + classical classifier).

---

## Part 2 - Fracture Detection (Binary)

### Dataset
- **Name:** Fracture Classification Dataset
- **Source:** https://www.kaggle.com/datasets/akshayramakrishnan28/fracture-classification-dataset
- **Total images:** 4,083
- **Fracture images:** 717
- **Challenge:** strong class imbalance

### Problem Statement
Bone fracture detection from X-ray images is typically performed by radiologists. Automating part of this process can improve triage speed and support decision-making, especially in emergency contexts.

### Required Work
- Analyze data distribution and quality.
- Train at least **3 AI model types** and tune hyperparameters.
- Propose and apply **data augmentation** (standard or generative).
- Quantify augmentation impact on model performance.
- Evaluate on **unseen data**.
---

## Setup

```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install numpy pandas matplotlib scikit-learn tensorflow opencv-python jupyter kaggle
```
---