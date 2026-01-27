# src/configs/config_svhnc_standard.py
DATASET = "svhn-c"
EPOCHS = 300
BATCH_SIZE = 125
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.005
MODEL_NAME = "WideResNet-28-10"
DA_TYPE = "CtrlA"
N_AUGS = 2
KAPPA_SP = 1.5
PHASE_LENGTH = 5
SETUP = "modified"
VAL_SET = "test_subset"
AUG_SPACE = "Control"
