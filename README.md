# ControlAugment (Ctrl-A)

This repository contains the official implementation of **Ctrl-A**, a control-driven, online data augmentation framework proposed in the paper:

> **Ctrl-A: Control-Driven Online Data Augmentation**

Ctrl-A dynamically adapts data augmentation strength during training using feedback from training dynamics, enabling improved generalization compared to static or heuristic augmentation strategies.

---

## Overview

Data augmentation is a key regularization technique in deep learning, yet most approaches rely on fixed or predefined policies. Ctrl-A introduces a **closed-loop control mechanism** that adjusts augmentation parameters online based on training feedback.

Key ideas:
- Augmentation strength is **adapted during training**
- Control signals are derived from training dynamics (e.g., loss behavior)
- The method balances optimization and regularization automatically

This repository provides:
- Control-driven augmentation (Ctrl-A)
- Standard and wide augmentation baselines
- TrivialAugment baseline
- Reproducible training pipelines for image classification

---

## Installation

```bash
# Clone the repo
git clone https://github.com/jbcdfm/ControlAugment.git
cd ControlAugment
```

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Run an experiment
An experiment can be run using the client file in the folder control_augment/
```bash
python train_model_cli.py
```
Alternatively, an IDE-friendly local version also exists as train_model_local.py. 


## Customizing experiments









