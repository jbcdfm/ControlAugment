# ControlAugment (Ctrl-A)

**Ctrl-A: Control-Driven Online Data Augmentation**

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

```bash
git clone https://github.com/jbcdfm/ControlAugment.git
cd ControlAugment
