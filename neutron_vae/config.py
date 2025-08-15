# config.py
import torch
import os

# Data
CSV_FILE = os.path.join(os.path.dirname(__file__), 'Sheet.csv')
SEQ_LEN = 200
N_TRACKS = 10

# Normalization bounds (from your reactor)
BOUNDS = {
    'min': [-0.63, -0.63, -26.499],
    'max': [0.63, 0.63, 21.468]
}

# Model - Match the trained model
INPUT_DIM = 3
COND_DIM = 6  # [x0, y0, z0, vx, vy, vz]
LATENT_DIM = 64  # Match trained model
HIDDEN_SIZE = 256  # Match trained model
NUM_LAYERS = 4  # Match trained model

# Training - Match the trained model
BATCH_SIZE = 10
N_EPOCHS = 15000
LR = 1e-4
BETA = 0.001
ALPHA = 0.01

# Advanced training settings
WARMUP_EPOCHS = 2000
BETA_SCHEDULE = True
GRADIENT_CLIP = 0.1
WEIGHT_DECAY = 1e-5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")