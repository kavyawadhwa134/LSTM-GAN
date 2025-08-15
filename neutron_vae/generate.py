# generate.py
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.model import VAE
from neutron_vae.config import *

def generate_track(model, condition, xyz_min, xyz_max):
    model.eval()
    cond = torch.tensor(condition, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        fake_norm = model.decode(z, cond)  # [-1,1]
    
    # Denormalize
    fake_norm = (fake_norm.squeeze().cpu().numpy() + 1) / 2  # [0,1]
    fake_real = fake_norm * (xyz_max - xyz_min) + xyz_min
    return fake_real

# Example
if __name__ == "__main__":
    from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks
    
    model = VAE().to(DEVICE)
    model.load_state_dict(torch.load('neutron_vae.pth'))

    # Load normalization parameters
    raw_tracks = load_and_split_tracks()
    _, _, xyz_min, xyz_max = preprocess_tracks(raw_tracks)

    condition = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.5])  # start at origin, go down-z
    new_track = generate_track(model, condition, xyz_min, xyz_max)
    np.savetxt('generated_track.csv', new_track, header='x,y,z', delimiter=',', fmt='%.6f')
    print("✅ Generated track saved.")