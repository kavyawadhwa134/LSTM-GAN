#!/usr/bin/env python3
"""
Accuracy check for newly generated neutron tracks
"""

import numpy as np
import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neutron_vae.model import UltraHighFidelityVAE
from neutron_vae.config import *

def load_real_data():
    """Load real neutron track data"""
    try:
        # Try to load the original CSV data
        import pandas as pd
        csv_file = os.path.join('neutron_vae', 'Sheet.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded real data from {csv_file}")
            return df
        else:
            print(f"📝 Real data file not found: {csv_file}")
            return None
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        return None

def load_generated_data():
    """Load generated neutron track data"""
    if os.path.exists('generated_neutron_tracks.npy'):
        tracks = np.load('generated_neutron_tracks.npy')
        print(f"✅ Loaded generated data: {tracks.shape}")
        return tracks
    else:
        print("❌ Generated data not found!")
        return None

def calculate_track_statistics(tracks):
    """Calculate statistics for tracks"""
    if tracks is None:
        return {}
    
    stats = {
        'lengths': [],
        'curvatures': [],
        'smoothness': [],
        'velocities': [],
        'spatial_range': []
    }
    
    for track in tracks:
        # Track length
        diffs = np.diff(track, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        stats['lengths'].append(length)
        
        # Average curvature
        if len(track) > 2:
            curvatures_track = []
            for j in range(1, len(track) - 1):
                v1 = track[j] - track[j-1]
                v2 = track[j+1] - track[j]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    curvatures_track.append(angle)
            if curvatures_track:
                stats['curvatures'].append(np.mean(curvatures_track))
        
        # Smoothness (variance of second derivatives)
        if len(track) > 2:
            second_derivs = np.diff(track, n=2, axis=0)
            stats['smoothness'].append(np.var(second_derivs))
        
        # Average velocity
        if len(diffs) > 0:
            stats['velocities'].append(np.mean(np.linalg.norm(diffs, axis=1)))
        
        # Spatial range
        spatial_range = np.max(track, axis=0) - np.min(track, axis=0)
        stats['spatial_range'].append(spatial_range)
    
    # Convert to numpy arrays
    for key in stats:
        if stats[key]:
            stats[key] = np.array(stats[key])
    
    return stats

def compare_statistics(real_stats, gen_stats):
    """Compare statistics between real and generated data"""
    print("\n📊 STATISTICS COMPARISON")
    print("=" * 50)
    
    for stat_name in ['lengths', 'curvatures', 'smoothness', 'velocities']:
        if stat_name in real_stats and stat_name in gen_stats and len(real_stats[stat_name]) > 0 and len(gen_stats[stat_name]) > 0:
            real_mean = np.mean(real_stats[stat_name])
            real_std = np.std(real_stats[stat_name])
            gen_mean = np.mean(gen_stats[stat_name])
            gen_std = np.std(gen_stats[stat_name])
            
            print(f"\n{stat_name.upper()}:")
            print(f"  Real:  Mean={real_mean:.4f}, Std={real_std:.4f}")
            print(f"  Gen:   Mean={gen_mean:.4f}, Std={gen_std:.4f}")
            print(f"  Diff:  Mean={abs(real_mean-gen_mean):.4f}, Std={abs(real_std-gen_std):.4f}")
            
            # Calculate similarity score
            mean_similarity = 1 - abs(real_mean - gen_mean) / (abs(real_mean) + 1e-8)
            std_similarity = 1 - abs(real_std - gen_std) / (abs(real_std) + 1e-8)
            overall_similarity = (mean_similarity + std_similarity) / 2
            
            print(f"  Score: {overall_similarity:.2%}")

def calculate_reconstruction_accuracy():
    """Calculate reconstruction accuracy using the trained model"""
    print("\n🔄 RECONSTRUCTION ACCURACY")
    print("=" * 50)
    
    # Load the trained model
    model = UltraHighFidelityVAE().to(DEVICE)
    
    checkpoint_path = 'neutron_vae_best.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model: Loss={checkpoint['loss']:.6f}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded model (old format)")
    else:
        print("❌ Model not found!")
        return
    
    model.eval()
    
    # Load generated data
    tracks = load_generated_data()
    if tracks is None:
        return
    
    # Calculate reconstruction error
    total_error = 0
    n_tracks = min(5, len(tracks))  # Test with first 5 tracks
    
    with torch.no_grad():
        for i in range(n_tracks):
            track = tracks[i]
            
            # Normalize track to [-1, 1] range
            track_tensor = torch.tensor(track, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            track_tensor = 2 * track_tensor - 1  # [0,1] -> [-1,1]
            
            # Create condition
            condition = torch.randn(1, COND_DIM, device=DEVICE) * 0.1
            
            # Encode and decode
            mu, logvar = model.encode(track_tensor, condition)
            z = model.reparameterize(mu, logvar)
            recon = model.decode(z, condition)
            
            # Calculate error
            error = torch.mean((recon - track_tensor) ** 2).item()
            total_error += error
            
            print(f"  Track {i+1}: MSE = {error:.6f}")
    
    avg_error = total_error / n_tracks
    print(f"\n📈 Average Reconstruction MSE: {avg_error:.6f}")
    
    # Calculate accuracy score
    accuracy_score = max(0, 100 * (1 - avg_error * 10))
    print(f"🎯 Reconstruction Accuracy Score: {accuracy_score:.2f}/100")

def main():
    print("🧪 ACCURACY CHECK FOR GENERATED DATA")
    print("=" * 60)
    
    # Load data
    real_data = load_real_data()
    generated_tracks = load_generated_data()
    
    if generated_tracks is None:
        print("❌ Cannot proceed without generated data!")
        return
    
    # Calculate statistics
    print("\n📊 Calculating statistics...")
    gen_stats = calculate_track_statistics(generated_tracks)
    
    if real_data is not None:
        # Convert real data to tracks format
        real_tracks = []
        for track_id in real_data['track_id'].unique():
            track_data = real_data[real_data['track_id'] == track_id]
            track = track_data[['x', 'y', 'z']].values
            real_tracks.append(track)
        
        real_tracks = np.array(real_tracks)
        real_stats = calculate_track_statistics(real_tracks)
        
        # Compare statistics
        compare_statistics(real_stats, gen_stats)
    else:
        print("\n📊 Generated Data Statistics:")
        for stat_name, values in gen_stats.items():
            if len(values) > 0:
                print(f"  {stat_name}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")
    
    # Calculate reconstruction accuracy
    calculate_reconstruction_accuracy()
    
    print(f"\n🎉 Accuracy check complete!")

if __name__ == "__main__":
    main()
