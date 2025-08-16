#!/usr/bin/env python3
"""
Generate more accurate neutron tracks with improved conditions and post-processing
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neutron_vae.model import UltraHighFidelityVAE
from neutron_vae.config import *

def load_real_data_conditions():
    """Load real data to extract realistic conditions"""
    try:
        csv_file = os.path.join('neutron_vae', 'Sheet.csv')
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Extract conditions from real data
            conditions = []
            for line in lines[1:]:
                values = line.strip().split(',')
                if len(values) >= 3:
                    try:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        # Create condition from position
                        condition = [x/10, y/10, z/10, 0.1, 0.1, 0.1]  # Normalized position + velocity
                        conditions.append(condition)
                    except ValueError:
                        continue
            
            print(f"✅ Loaded {len(conditions)} real conditions")
            return np.array(conditions)
        else:
            print(f"📝 Real data file not found, using synthetic conditions")
            return None
    except Exception as e:
        print(f"❌ Error loading real conditions: {e}")
        return None

def generate_accurate_samples(n_samples=20):
    """Generate more accurate neutron track samples"""
    print(f"🎲 Generating {n_samples} accurate neutron track samples...")
    
    # Load the trained model
    model = UltraHighFidelityVAE().to(DEVICE)
    
    # Load the checkpoint
    checkpoint_path = 'neutron_vae_best.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from {checkpoint_path}")
            print(f"   Training loss: {checkpoint['loss']:.6f}")
            print(f"   Epoch: {checkpoint['epoch']}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model from {checkpoint_path} (old format)")
    else:
        print(f"❌ Model file {checkpoint_path} not found")
        return
    
    model.eval()
    
    # Load real conditions
    real_conditions = load_real_data_conditions()
    
    # Generate samples
    generated_tracks = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Sample random latent vector with better distribution
            z = torch.randn(1, LATENT_DIM, device=DEVICE) * 0.5  # Reduced variance for more stable generation
            
            # Use realistic conditions if available
            if real_conditions is not None and len(real_conditions) > 0:
                # Randomly select a real condition
                cond_idx = np.random.randint(0, len(real_conditions))
                condition = torch.tensor(real_conditions[cond_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            else:
                # Create more realistic synthetic condition
                condition = torch.randn(1, COND_DIM, device=DEVICE) * 0.05  # Smaller variance for more realistic conditions
            
            # Generate track
            recon = model.decode(z, condition)
            
            # Post-process the generated track for better accuracy
            track = recon[0].cpu().numpy()
            
            # Apply smoothing to reduce noise
            track = apply_smoothing(track)
            
            # Ensure physical constraints
            track = apply_physical_constraints(track)
            
            generated_tracks.append(track)
            
            print(f"   Generated accurate track {i+1}: shape {track.shape}")
    
    # Save generated tracks
    generated_tracks = np.array(generated_tracks)
    np.save('accurate_neutron_tracks.npy', generated_tracks)
    
    print(f"\n✅ Generated {n_samples} accurate tracks successfully!")
    print(f"📁 Saved as: accurate_neutron_tracks.npy")
    print(f"📊 Shape: {generated_tracks.shape}")
    
    # Print some statistics
    print(f"\n📈 Accurate Track Statistics:")
    print(f"   Mean track length: {np.mean([len(track) for track in generated_tracks]):.2f}")
    print(f"   Track shape: {generated_tracks[0].shape}")
    print(f"   Value range: [{generated_tracks.min():.4f}, {generated_tracks.max():.4f}]")
    
    return generated_tracks

def apply_smoothing(track, window_size=3):
    """Apply smoothing to reduce noise in the track"""
    if len(track) < window_size:
        return track
    
    smoothed_track = np.copy(track)
    
    # Apply moving average smoothing
    for i in range(window_size, len(track) - window_size):
        smoothed_track[i] = np.mean(track[i-window_size:i+window_size+1], axis=0)
    
    return smoothed_track

def apply_physical_constraints(track):
    """Apply physical constraints to make the track more realistic"""
    # Ensure the track doesn't have extreme jumps
    max_jump = 0.5  # Maximum allowed jump between consecutive points
    
    constrained_track = np.copy(track)
    
    for i in range(1, len(track)):
        jump = np.linalg.norm(track[i] - track[i-1])
        if jump > max_jump:
            # Reduce the jump by interpolating
            direction = (track[i] - track[i-1]) / jump
            constrained_track[i] = track[i-1] + direction * max_jump
    
    return constrained_track

def convert_to_csv_accurate(tracks):
    """Convert accurate tracks to CSV format"""
    print("🔄 Converting accurate tracks to CSV format...")
    
    # Create CSV data
    csv_data = []
    
    for track_idx, track in enumerate(tracks):
        for point_idx, point in enumerate(track):
            csv_data.append({
                'track_id': track_idx + 1,
                'point_id': point_idx + 1,
                'x': point[0],
                'y': point[1], 
                'z': point[2]
            })
    
    # Create DataFrame and save to CSV
    import pandas as pd
    df = pd.DataFrame(csv_data)
    
    # Save to CSV
    csv_filename = 'accurate_neutron_tracks.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"✅ Converted to CSV: {csv_filename}")
    print(f"📊 CSV Statistics:")
    print(f"   Total tracks: {len(tracks)}")
    print(f"   Total points: {len(csv_data)}")
    print(f"   Points per track: {len(tracks[0])}")
    
    return df

def main():
    print("🎯 GENERATING ACCURATE NEUTRON TRACKS")
    print("=" * 50)
    
    # Generate accurate samples
    tracks = generate_accurate_samples(n_samples=20)
    
    if tracks is not None:
        # Convert to CSV
        convert_to_csv_accurate(tracks)
        
        print(f"\n🎉 Accurate generation complete!")
        print(f"📁 Files created:")
        print(f"   - accurate_neutron_tracks.npy (accurate raw data)")
        print(f"   - accurate_neutron_tracks.csv (accurate CSV format)")
    else:
        print("❌ Accurate generation failed!")

if __name__ == "__main__":
    main()
