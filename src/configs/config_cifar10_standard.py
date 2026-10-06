# src/configs/config_cifar10_standard.py
DATASET = "cifar10"
EPOCHS = 200
BATCH_SIZE = 125
LEARNING_RATE = 0.1
LEARNING_RATE_TYPE = "cos"
WEIGHT_DECAY = 5e-4
MODEL_NAME = "WideResNet-28-10"
DA_TYPE = "CtrlA"
N_AUGS = 2
KAPPA_SP = 2.0
PHASE_LENGTH = 5
SETUP = "standard"
VAL_SET = "train_subset"
AUG_SPACE = "Control"
CUTOUT = 16
