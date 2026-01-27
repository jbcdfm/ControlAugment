# src/configs/config_cifar100_standard.py
DATASET = "cifar100"
EPOCHS = 200
BATCH_SIZE = 125
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
MODEL_NAME = "WideResNet-28-10"
DA_TYPE = "CtrlA"
N_AUGS = 2
KAPPA_SP = 1.5
PHASE_LENGTH = 5
SETUP = "standard"
VAL_SET = "test_subset"
