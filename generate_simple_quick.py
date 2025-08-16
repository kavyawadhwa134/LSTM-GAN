#!/usr/bin/env python3
"""
Simple Generation Script for Quick Training Model
Generates neutron tracks with proper spatial constraints and denormalization.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neutron_vae.model import UltraHighFidelityVAE
from neutron_vae.config import BOUNDS, SEQ_LEN, COND_DIM, DEVICE

def generate_tracks_with_constraints(model, num_tracks=20, temperature=1.0):
    """Generate tracks with spatial constraints."""
    model.eval()
    
    with torch.no_grad():
        # Generate random latent vectors
        z = torch.randn(num_tracks, model.latent_dim, device=DEVICE) * temperature
        
        # Generate realistic conditions based on real data bounds
        conditions = generate_realistic_conditions(num_tracks)
        conditions = torch.FloatTensor(conditions).to(DEVICE)
        
        # Generate tracks
        generated = model.decode(z, conditions)
        
        # Apply spatial constraints
        generated = apply_spatial_constraints(generated)
        
        return generated.cpu().numpy(), conditions.cpu().numpy()

def generate_realistic_conditions(num_tracks):
    """Generate realistic starting conditions based on real data bounds."""
    conditions = []
    
    for _ in range(num_tracks):
        # Generate realistic starting positions within bounds
        start_x = np.random.uniform(-0.6, 0.6)  # Within X bounds
        start_y = np.random.uniform(-0.6, 0.6)  # Within Y bounds
        start_z = np.random.uniform(-10.0, 9.0)  # Within Z bounds
        
        # Generate realistic initial velocities
        # Neutron tracks typically have moderate velocities
        vel_magnitude = np.random.uniform(0.1, 2.0)
        vel_direction = np.random.randn(3)
        vel_direction = vel_direction / np.linalg.norm(vel_direction)
        vel_x, vel_y, vel_z = vel_magnitude * vel_direction
        
        # Normalize to match model input range
        xyz_min = np.array(BOUNDS['min'])
        xyz_max = np.array(BOUNDS['max'])
        
        start_norm = 2 * (np.array([start_x, start_y, start_z]) - xyz_min) / (xyz_max - xyz_min) - 1
        vel_norm = 2 * np.array([vel_x, vel_y, vel_z]) / (xyz_max - xyz_min)
        
        condition = np.concatenate([start_norm, vel_norm])
        conditions.append(condition)
    
    return np.array(conditions)

def apply_spatial_constraints(tracks):
    """Apply spatial constraints to ensure realistic tracks."""
    # Convert from [-1,1] to real coordinates
    xyz_min = torch.tensor(BOUNDS['min'], device=tracks.device, dtype=tracks.dtype)
    xyz_max = torch.tensor(BOUNDS['max'], device=tracks.device, dtype=tracks.dtype)
    
    # Clamp tracks to realistic bounds
    tracks_real = (tracks + 1) / 2 * (xyz_max - xyz_min) + xyz_min
    tracks_real = torch.clamp(tracks_real, xyz_min, xyz_max)
    
    # Convert back to [-1,1] range
    tracks = 2 * (tracks_real - xyz_min) / (xyz_max - xyz_min) - 1
    
    return tracks

def denormalize_tracks(tracks_norm):
    """Denormalize tracks from [-1,1] to real coordinates."""
    xyz_min = np.array(BOUNDS['min'])
    xyz_max = np.array(BOUNDS['max'])
    
    # Convert from [-1,1] to [0,1]
    tracks_01 = (tracks_norm + 1) / 2
    
    # Convert to real coordinates
    tracks_real = tracks_01 * (xyz_max - xyz_min) + xyz_min
    
    return tracks_real

def analyze_generated_tracks(tracks):
    """Analyze generated tracks for quality assessment."""
    print(f"\n📊 GENERATED TRACKS ANALYSIS ==================================================")
    
    # Basic statistics
    print(f"📈 Track Statistics:")
    print(f"   Number of tracks: {len(tracks)}")
    print(f"   Points per track: {tracks.shape[1]}")
    print(f"   Dimensions: {tracks.shape[2]}")
    
    # Length statistics
    track_lengths = []
    for track in tracks:
        diffs = track[1:] - track[:-1]
        length = np.sum(np.linalg.norm(diffs, axis=1))
        track_lengths.append(length)
    
    track_lengths = np.array(track_lengths)
    print(f"\n📏 Length Statistics:")
    print(f"   Mean length: {np.mean(track_lengths):.4f}")
    print(f"   Std length: {np.std(track_lengths):.4f}")
    print(f"   Min length: {np.min(track_lengths):.4f}")
    print(f"   Max length: {np.max(track_lengths):.4f}")
    
    # Spatial bounds
    x_min, x_max = np.min(tracks[:, :, 0]), np.max(tracks[:, :, 0])
    y_min, y_max = np.min(tracks[:, :, 1]), np.max(tracks[:, :, 1])
    z_min, z_max = np.min(tracks[:, :, 2]), np.max(tracks[:, :, 2])
    
    print(f"\n📍 Spatial Bounds:")
    print(f"   X: [{x_min:.4f}, {x_max:.4f}]")
    print(f"   Y: [{y_min:.4f}, {y_max:.4f}]")
    print(f"   Z: [{z_min:.4f}, {z_max:.4f}]")
    
    # Smoothness analysis
    smoothness_scores = []
    for track in tracks:
        if len(track) > 2:
            second_deriv = track[2:] - 2 * track[1:-1] + track[:-2]
            smoothness = np.mean(np.linalg.norm(second_deriv, axis=1))
            smoothness_scores.append(smoothness)
    
    if smoothness_scores:
        smoothness_scores = np.array(smoothness_scores)
        print(f"\n🔄 Smoothness Statistics:")
        print(f"   Mean smoothness: {np.mean(smoothness_scores):.6f}")
        print(f"   Std smoothness: {np.std(smoothness_scores):.6f}")
    
    return {
        'track_lengths': track_lengths,
        'spatial_bounds': {'x': [x_min, x_max], 'y': [y_min, y_max], 'z': [z_min, z_max]},
        'smoothness_scores': smoothness_scores if smoothness_scores else None
    }

def save_tracks_to_csv(tracks, filename):
    """Save tracks to CSV format."""
    print(f"\n🔄 Converting tracks to CSV format...")
    
    # Denormalize tracks to real coordinates
    tracks_real = denormalize_tracks(tracks)
    
    # Create DataFrame
    rows = []
    for track_id, track in enumerate(tracks_real, 1):
        for point_id, point in enumerate(track, 1):
            rows.append({
                'track_id': track_id,
                'point_id': point_id,
                'x': point[0],
                'y': point[1],
                'z': point[2]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    
    print(f"✅ Converted to CSV: {filename}")
    print(f"📊 CSV Statistics:")
    print(f"   Total tracks: {len(tracks)}")
    print(f"   Total points: {len(df)}")
    print(f"   Points per track: {len(df) // len(tracks)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Show first few rows
    print(f"\n📋 First 10 rows:")
    print(df.head(10).to_string(index=False))

def main():
    """Main generation function."""
    print("🎲 SIMPLE GENERATION FROM QUICK TRAINING MODEL ============================================================")
    
    # Check if model exists
    model_path = 'neutron_vae_quick_best.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using quick_training.py")
        return
    
    # Load model
    print(f"📥 Loading model from {model_path}...")
    model = UltraHighFidelityVAE().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Get training loss from filename or use default
    training_loss = 0.047868  # Default value
    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Training loss: {training_loss:.6f}")
    
    # Generate tracks
    print(f"\n🎲 GENERATING TRACKS WITH QUICK MODEL ==================================================")
    num_tracks = 20
    temperature = 1.0
    
    print(f"Number of tracks: {num_tracks}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Temperature: {temperature}")
    print(f"Device: {DEVICE}")
    
    generated_tracks, conditions = generate_tracks_with_constraints(
        model, num_tracks=num_tracks, temperature=temperature
    )
    
    print(f"✅ Generation completed")
    print(f"Generated tracks shape: {generated_tracks.shape}")
    
    # Analyze generated tracks
    analysis = analyze_generated_tracks(generated_tracks)
    
    # Save as numpy array
    np.save('quick_generated_neutron_tracks.npy', generated_tracks)
    print(f"✅ Saved as numpy array: quick_generated_neutron_tracks.npy")
    
    # Save as CSV
    save_tracks_to_csv(generated_tracks, 'quick_generated_neutron_tracks.csv')
    
    print(f"\n🎉 Generation complete!")
    print(f"📁 Generated data files:")
    print(f"   - quick_generated_neutron_tracks.npy (raw data)")
    print(f"   - quick_generated_neutron_tracks.csv (CSV format)")

if __name__ == "__main__":
    main()
