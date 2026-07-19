import torch 
from pathlib import Path 

class Config:
    DATA_ROOT = "/kaggle/input/datasets/thanhtgfg/b0q-01/Dataset-Crop"
    RESULT_DIR = Path('./results')

    IMAGE_SIZE = 128 
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.20
    TEST_RATIO = 0.10

    num_Qubits = 4
    num_Layers = 6

    EPOCH = 100
    BATCH_SIZE = 32
    LR = 3e-4
    LR_CNN = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    DROPOUT = 0.08
    PATIENCE = 7
    SEED = 42

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()
cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)