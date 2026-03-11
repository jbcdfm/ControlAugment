# src/configs/config_cifar10_modified.py
DATASET = "cifar10"
EPOCHS = 500
BATCH_SIZE = 125
LEARNING_RATE = 0.05
LEARNING_RATE_TYPE = "erf"
WEIGHT_DECAY = 2.5e-4
MODEL_NAME = "WideResNet-28-10"
DA_TYPE = "CtrlA"
N_AUGS = 2
KAPPA_SP = 1.5
PHASE_LENGTH = 5
SETUP = "modified"
VAL_SET = "test_subset"
AUG_SPACE = "Control"
