#!/usr/bin/env python3
"""
Simple Generation Script for Quick Training Model
"""

import torch
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.model import UltraHighFidelityVAE
from neutron_vae.config import *

def main():
    print("🎲 SIMPLE GENERATION FROM QUICK TRAINING MODEL")
    print("=" * 60)
    
    # Check if model exists
    model_path = 'neutron_vae_quick_best.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Load model
    print(f"📥 Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    
    model = UltraHighFidelityVAE().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Training loss: {checkpoint['loss']:.6f}")
    
    # Generate tracks
    print(f"\n🎲 Generating 20 tracks...")
    n_tracks = 20
    seq_len = 200
    
    # Generate conditions and latent vectors
    conditions = torch.randn(n_tracks, COND_DIM, device=device) * 0.1
    latent_vectors = torch.randn(n_tracks, LATENT_DIM, device=device)
    
    with torch.no_grad():
        generated_tracks = model.decode(latent_vectors, conditions)
    
    # Convert to numpy
    tracks_np = generated_tracks.cpu().numpy()
    
    print(f"✅ Generated tracks shape: {tracks_np.shape}")
    
    # Save as numpy array
    npy_filename = 'quick_generated_neutron_tracks.npy'
    np.save(npy_filename, tracks_np)
    print(f"✅ Saved as numpy array: {npy_filename}")
    
    # Save as CSV
    csv_filename = 'quick_generated_neutron_tracks.csv'
    try:
        import pandas as pd
        
        # Flatten tracks for CSV
        csv_data = []
        for track_id, track in enumerate(tracks_np):
            for point_id, point in enumerate(track):
                csv_data.append({
                    'track_id': track_id,
                    'point_id': point_id,
                    'x': point[0],
                    'y': point[1],
                    'z': point[2]
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_filename, index=False)
        print(f"✅ Saved as CSV: {csv_filename}")
        print(f"   Rows: {len(df)}")
        
    except ImportError:
        print("⚠️ Pandas not available, skipping CSV export")
    
    # Analyze tracks
    print(f"\n📊 Track Analysis:")
    print(f"   Number of tracks: {tracks_np.shape[0]}")
    print(f"   Points per track: {tracks_np.shape[1]}")
    print(f"   Dimensions: {tracks_np.shape[2]}")
    
    # Calculate track lengths
    track_lengths = []
    for track in tracks_np:
        diffs = np.diff(track, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        track_lengths.append(length)
    
    track_lengths = np.array(track_lengths)
    print(f"   Mean length: {np.mean(track_lengths):.4f}")
    print(f"   Std length: {np.std(track_lengths):.4f}")
    
    # Calculate spatial bounds
    all_points = tracks_np.reshape(-1, 3)
    bounds = {
        'x': (np.min(all_points[:, 0]), np.max(all_points[:, 0])),
        'y': (np.min(all_points[:, 1]), np.max(all_points[:, 1])),
        'z': (np.min(all_points[:, 2]), np.max(all_points[:, 2]))
    }
    
    print(f"   X bounds: [{bounds['x'][0]:.4f}, {bounds['x'][1]:.4f}]")
    print(f"   Y bounds: [{bounds['y'][0]:.4f}, {bounds['y'][1]:.4f}]")
    print(f"   Z bounds: [{bounds['z'][0]:.4f}, {bounds['z'][1]:.4f}]")
    
    print(f"\n🎉 Generation completed successfully!")
    print(f"📁 Files created:")
    print(f"   - {npy_filename}")
    print(f"   - {csv_filename}")

if __name__ == "__main__":
    main()
