# ControlAugment (Ctrl-A)

This repository contains the official implementation of **Ctrl-A**, a control-driven, online data augmentation framework proposed in the paper:

> **Ctrl-A: Control-Driven Online Data Augmentation**

Ctrl-A dynamically adapts data augmentation strength during training using feedback from training dynamics, enabling improved generalization compared to static or heuristic augmentation strategies.

---

## Overview

Data augmentation is a key regularization technique in deep learning, yet most approaches rely on fixed or predefined policies. Ctrl-A introduces a **closed-loop control mechanism** that adjusts augmentation parameters online based on training feedback.

Key ideas:
- Augmentation strengths are individually **adapted during training**
- Control signals are derived from training dynamics (e.g., loss behavior)
- The method balances optimization and regularization automatically


<img src="figures/ControlAugment_sketch.png" alt="Ctrl-A Overview" width="800"/>


---

## Installation

```bash
# Clone the repo
git clone https://github.com/jbcdfm/ControlAugment.git
cd ControlAugment
```

```bash
# Create a descriptive virtual environment
python -m venv ctrla_env

# Activate the environment
# Linux / Mac:
# source ctrla_env/bin/activate
# Windows:
ctrla_env\Scripts\activate
```

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Quick start
An experiment can be run using the client file in the folder control_augment/
```bash
python control_augment/train_model_cli.py
```
Alternatively, an IDE-friendly local version also exists as control_augment/train_model_local.py. 


## Customizing experiments
The configuration files reside in src/configs, and may be selected as
```bash
python control_augment/train_model_cli.py --config config_cifar10_standard
```
In addition, single arguments may configured as
```bash
python control_augment/train_model_cli.py --config config_cifar10_modified --epochs 300 --N 2 --kappa_sp 1
```


## Citation




## Reference repos
This work and repository were largely inspired by the [TrivialAugment](https://github.com/automl/trivialaugment) project.
The data augmentation operations are implemented using PyTorch’s `torchvision` package.



















