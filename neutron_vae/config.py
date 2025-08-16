# config.py
import torch
import os

# Data
CSV_FILE = os.path.join(os.path.dirname(__file__), 'Sheet.csv')
SEQ_LEN = 200
N_TRACKS = 10

# Real data bounds (from analysis of Sheet.csv)
BOUNDS = {
    'min': [-0.63, -0.63, -10.204],  # Real X, Y, Z minimums
    'max': [0.63, 0.63, 9.862]       # Real X, Y, Z maximums
}

# Spatial constraints for realistic generation
SPATIAL_CONSTRAINTS = {
    'volume_bounds': BOUNDS,  # Enforce volume constraints
    'max_track_length': 20.0,  # Maximum track length in real units
    'min_track_length': 1.0,   # Minimum track length in real units
    'smoothness_weight': 0.1,  # Weight for track smoothness
    'volume_weight': 0.05      # Weight for volume constraint loss
}

# Model - Match the trained model
INPUT_DIM = 3
COND_DIM = 6  # [x0, y0, z0, vx, vy, vz]
LATENT_DIM = 64  # Match trained model
HIDDEN_SIZE = 256  # Match trained model
NUM_LAYERS = 4  # Match trained model

# Training - Optimized for GPU with maximum memory usage
BATCH_SIZE = 8192 if torch.cuda.is_available() else 10  # Optimal batch size for A10 GPU
N_EPOCHS = 15000
LR = 1e-4
BETA = 0.001
ALPHA = 0.01

# GPU Memory Optimization
MAX_GPU_MEMORY_FRACTION = 0.95  # Use 95% of available GPU memory
GRADIENT_ACCUMULATION_STEPS = 1  # Increase for larger effective batch sizes

# Advanced training settings
WARMUP_EPOCHS = 2000
BETA_SCHEDULE = True
GRADIENT_CLIP = 0.1
WEIGHT_DECAY = 1e-5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")