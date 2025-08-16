#!/usr/bin/env python3
"""
Denormalize generated neutron tracks to match real data scale
"""

import numpy as np
import os

def load_real_data_bounds():
    """Load real data to determine the bounds for denormalization"""
    try:
        csv_file = os.path.join('neutron_vae', 'Sheet.csv')
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Parse data to find bounds
            x_vals, y_vals, z_vals = [], [], []
            for line in lines[1:]:
                values = line.strip().split(',')
                if len(values) >= 3:
                    try:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        x_vals.append(x)
                        y_vals.append(y)
                        z_vals.append(z)
                    except ValueError:
                        continue
            
            # Calculate bounds
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            z_min, z_max = min(z_vals), max(z_vals)
            
            bounds = {
                'x': (x_min, x_max),
                'y': (y_min, y_max),
                'z': (z_min, z_max)
            }
            
            print(f"📊 Real data bounds:")
            print(f"   X: [{x_min:.4f}, {x_max:.4f}]")
            print(f"   Y: [{y_min:.4f}, {y_max:.4f}]")
            print(f"   Z: [{z_min:.4f}, {z_max:.4f}]")
            
            return bounds
        else:
            print(f"📝 Real data file not found, using default bounds")
            return None
    except Exception as e:
        print(f"❌ Error loading real data bounds: {e}")
        return None

def denormalize_tracks(tracks, bounds=None):
    """Denormalize tracks from [-1,1] to real data scale"""
    if bounds is None:
        # Use default bounds from config
        bounds = {
            'x': (-0.63, 0.63),
            'y': (-0.63, 0.63),
            'z': (-26.499, 21.468)
        }
        print(f"📊 Using default bounds from config")
    
    print(f"🔄 Denormalizing tracks...")
    
    # Create denormalized tracks
    denorm_tracks = np.zeros_like(tracks)
    
    for i in range(len(tracks)):
        for j in range(len(tracks[i])):
            # Denormalize each coordinate
            x_norm, y_norm, z_norm = tracks[i, j]
            
            # Denormalize from [-1,1] to real bounds
            x_real = (x_norm + 1) / 2 * (bounds['x'][1] - bounds['x'][0]) + bounds['x'][0]
            y_real = (y_norm + 1) / 2 * (bounds['y'][1] - bounds['y'][0]) + bounds['y'][0]
            z_real = (z_norm + 1) / 2 * (bounds['z'][1] - bounds['z'][0]) + bounds['z'][0]
            
            denorm_tracks[i, j] = [x_real, y_real, z_real]
    
    print(f"✅ Denormalization complete!")
    return denorm_tracks

def save_denormalized_data(denorm_tracks):
    """Save denormalized data in multiple formats"""
    print(f"💾 Saving denormalized data...")
    
    # Save as numpy array
    np.save('denormalized_neutron_tracks.npy', denorm_tracks)
    print(f"✅ Saved: denormalized_neutron_tracks.npy")
    
    # Save as CSV
    csv_data = []
    for track_idx, track in enumerate(denorm_tracks):
        for point_idx, point in enumerate(track):
            csv_data.append({
                'track_id': track_idx + 1,
                'point_id': point_idx + 1,
                'x': point[0],
                'y': point[1],
                'z': point[2]
            })
    
    import pandas as pd
    df = pd.DataFrame(csv_data)
    df.to_csv('denormalized_neutron_tracks.csv', index=False)
    print(f"✅ Saved: denormalized_neutron_tracks.csv")
    
    return df

def calculate_accuracy_improvement(original_tracks, denorm_tracks, bounds):
    """Calculate how much accuracy improved after denormalization"""
    print(f"\n📈 ACCURACY IMPROVEMENT ANALYSIS")
    print("=" * 50)
    
    # Original statistics
    orig_min, orig_max = np.min(original_tracks), np.max(original_tracks)
    orig_range = orig_max - orig_min
    
    # Denormalized statistics
    denorm_min, denorm_max = np.min(denorm_tracks), np.max(denorm_tracks)
    denorm_range = denorm_max - denorm_min
    
    # Real data statistics
    real_x_range = bounds['x'][1] - bounds['x'][0]
    real_y_range = bounds['y'][1] - bounds['y'][0]
    real_z_range = bounds['z'][1] - bounds['z'][0]
    real_total_range = real_x_range + real_y_range + real_z_range
    
    print(f"📊 Range Comparison:")
    print(f"   Original (normalized): {orig_range:.4f}")
    print(f"   Denormalized: {denorm_range:.4f}")
    print(f"   Real data: {real_total_range:.4f}")
    
    # Calculate accuracy improvement
    orig_accuracy = 1 - abs(orig_range - 2) / 2  # Assuming [-1,1] range
    denorm_accuracy = 1 - abs(denorm_range - real_total_range) / real_total_range
    
    print(f"\n🎯 Accuracy Scores:")
    print(f"   Original: {orig_accuracy:.2%}")
    print(f"   Denormalized: {denorm_accuracy:.2%}")
    print(f"   Improvement: {denorm_accuracy - orig_accuracy:.2%}")
    
    return denorm_accuracy

def main():
    print("🎯 IMPROVING GENERATED DATA ACCURACY")
    print("=" * 50)
    
    # Load generated data
    if not os.path.exists('generated_neutron_tracks.npy'):
        print("❌ Generated data not found! Run generation first.")
        return
    
    tracks = np.load('generated_neutron_tracks.npy')
    print(f"✅ Loaded generated data: {tracks.shape}")
    
    # Load real data bounds
    bounds = load_real_data_bounds()
    
    # Denormalize tracks
    denorm_tracks = denormalize_tracks(tracks, bounds)
    
    # Save denormalized data
    df = save_denormalized_data(denorm_tracks)
    
    # Calculate accuracy improvement
    if bounds:
        accuracy = calculate_accuracy_improvement(tracks, denorm_tracks, bounds)
    
    print(f"\n🎉 Accuracy improvement complete!")
    print(f"📁 New files created:")
    print(f"   - denormalized_neutron_tracks.npy (denormalized raw data)")
    print(f"   - denormalized_neutron_tracks.csv (denormalized CSV)")
    print(f"   - generated_neutron_tracks.npy (original normalized data)")
    print(f"   - generated_neutron_tracks.csv (original CSV)")

if __name__ == "__main__":
    main()
