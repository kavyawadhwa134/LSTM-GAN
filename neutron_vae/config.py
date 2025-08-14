# config.py
import torch

# Data
CSV_FILE = 'Sheet.csv'
SEQ_LEN = 200
N_TRACKS = 10

# Normalization bounds (from your reactor)
BOUNDS = {
    'min': [-0.63, -0.63, -26.499],
    'max': [0.63, 0.63, 21.468]
}

# Model
INPUT_DIM = 3
COND_DIM = 6  # [x0, y0, z0, vx, vy, vz]
LATENT_DIM = 16
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Training
BATCH_SIZE = 5
N_EPOCHS = 3000
LR = 1e-3
BETA = 0.1  # KL weight

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")