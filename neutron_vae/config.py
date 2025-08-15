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

# Model - Ultra High Fidelity
INPUT_DIM = 3
COND_DIM = 6  # [x0, y0, z0, vx, vy, vz]
LATENT_DIM = 64  # Increased for better representation
HIDDEN_SIZE = 256  # Much larger for ultra-high fidelity
NUM_LAYERS = 4  # Deeper network

# Training - Ultra High Fidelity
BATCH_SIZE = 10  # Increased batch size
N_EPOCHS = 15000  # More training epochs
LR = 1e-4  # Lower learning rate for ultra-stability
BETA = 0.001  # Very low KL weight to focus on reconstruction
ALPHA = 0.01  # Lower Smooth L1 weight

# Advanced training settings
WARMUP_EPOCHS = 2000  # Longer warmup period
BETA_SCHEDULE = True  # Dynamic KL weight scheduling
GRADIENT_CLIP = 0.1  # Very tight gradient clipping
WEIGHT_DECAY = 1e-5  # Lower weight decay

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")