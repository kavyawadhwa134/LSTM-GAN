#!/usr/bin/env python3
"""
Simple accuracy check for generated neutron tracks
"""

import numpy as np
import os

def load_generated_data():
    """Load generated neutron track data"""
    if os.path.exists('generated_neutron_tracks.npy'):
        tracks = np.load('generated_neutron_tracks.npy')
        print(f"✅ Loaded generated data: {tracks.shape}")
        return tracks
    else:
        print("❌ Generated data not found!")
        return None

def load_real_data():
    """Load real neutron track data"""
    try:
        csv_file = os.path.join('neutron_vae', 'Sheet.csv')
        if os.path.exists(csv_file):
            # Read CSV manually to avoid pandas issues
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            header = lines[0].strip().split(',')
            print(f"📋 Real data columns: {header}")
            
            # Parse data
            data = []
            for line in lines[1:]:
                values = line.strip().split(',')
                if len(values) >= 3:  # At least x, y, z
                    try:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        data.append([x, y, z])
                    except ValueError:
                        continue
            
            real_tracks = np.array(data)
            print(f"✅ Loaded real data: {real_tracks.shape}")
            return real_tracks
        else:
            print(f"📝 Real data file not found: {csv_file}")
            return None
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        return None

def calculate_track_statistics(tracks):
    """Calculate basic statistics for tracks"""
    if tracks is None or len(tracks) == 0:
        return {}
    
    stats = {
        'lengths': [],
        'spatial_range': [],
        'value_range': []
    }
    
    # If tracks is 2D (single track), reshape to 3D
    if len(tracks.shape) == 2:
        tracks = tracks.reshape(1, -1, 3)
    
    for track in tracks:
        # Track length
        diffs = np.diff(track, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        stats['lengths'].append(length)
        
        # Spatial range
        spatial_range = np.max(track, axis=0) - np.min(track, axis=0)
        stats['spatial_range'].append(spatial_range)
        
        # Value range
        value_range = np.max(track) - np.min(track)
        stats['value_range'].append(value_range)
    
    # Convert to numpy arrays
    for key in stats:
        if stats[key]:
            stats[key] = np.array(stats[key])
    
    return stats

def compare_statistics(real_stats, gen_stats):
    """Compare statistics between real and generated data"""
    print("\n📊 STATISTICS COMPARISON")
    print("=" * 50)
    
    for stat_name in ['lengths', 'value_range']:
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

def calculate_quality_metrics(tracks):
    """Calculate quality metrics for generated tracks"""
    print("\n🎯 QUALITY METRICS")
    print("=" * 50)
    
    if tracks is None:
        return
    
    # Basic statistics
    print(f"📊 Generated Data Statistics:")
    print(f"  Number of tracks: {len(tracks)}")
    print(f"  Points per track: {tracks.shape[1]}")
    print(f"  Dimensions: {tracks.shape[2]}")
    
    # Value ranges
    min_val = np.min(tracks)
    max_val = np.max(tracks)
    mean_val = np.mean(tracks)
    std_val = np.std(tracks)
    
    print(f"\n📈 Value Statistics:")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std: {std_val:.4f}")
    print(f"  Range: {max_val - min_val:.4f}")
    
    # Track consistency
    track_lengths = []
    for track in tracks:
        diffs = np.diff(track, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        track_lengths.append(length)
    
    track_lengths = np.array(track_lengths)
    print(f"\n📏 Track Length Statistics:")
    print(f"  Mean length: {np.mean(track_lengths):.4f}")
    print(f"  Std length: {np.std(track_lengths):.4f}")
    print(f"  Min length: {np.min(track_lengths):.4f}")
    print(f"  Max length: {np.max(track_lengths):.4f}")
    
    # Smoothness check
    smoothness_scores = []
    for track in tracks:
        if len(track) > 2:
            second_derivs = np.diff(track, n=2, axis=0)
            smoothness = np.var(second_derivs)
            smoothness_scores.append(smoothness)
    
    if smoothness_scores:
        smoothness_scores = np.array(smoothness_scores)
        print(f"\n🔄 Smoothness Statistics:")
        print(f"  Mean smoothness: {np.mean(smoothness_scores):.6f}")
        print(f"  Std smoothness: {np.std(smoothness_scores):.6f}")
    
    # Calculate overall quality score
    length_consistency = 1 - np.std(track_lengths) / (np.mean(track_lengths) + 1e-8)
    value_distribution = 1 - abs(mean_val)  # Closer to 0 is better
    range_appropriateness = 1 - abs(max_val - min_val - 2) / 2  # Assuming [-1,1] range
    
    overall_quality = (length_consistency + value_distribution + range_appropriateness) / 3
    print(f"\n🏆 Overall Quality Score: {overall_quality:.2%}")

def main():
    print("🧪 SIMPLE ACCURACY CHECK FOR GENERATED DATA")
    print("=" * 60)
    
    # Load generated data
    generated_tracks = load_generated_data()
    
    if generated_tracks is None:
        print("❌ Cannot proceed without generated data!")
        return
    
    # Calculate quality metrics for generated data
    calculate_quality_metrics(generated_tracks)
    
    # Load real data if available
    real_tracks = load_real_data()
    
    if real_tracks is not None:
        # Calculate statistics
        print("\n📊 Calculating statistics...")
        gen_stats = calculate_track_statistics(generated_tracks)
        real_stats = calculate_track_statistics(real_tracks)
        
        # Compare statistics
        compare_statistics(real_stats, gen_stats)
    else:
        print("\n📊 Generated Data Statistics:")
        gen_stats = calculate_track_statistics(generated_tracks)
        for stat_name, values in gen_stats.items():
            if len(values) > 0:
                print(f"  {stat_name}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")
    
    print(f"\n🎉 Accuracy check complete!")
    print(f"📁 Generated files:")
    print(f"   - generated_neutron_tracks.npy (raw data)")
    print(f"   - generated_neutron_tracks.csv (CSV format)")
    print(f"   - generated_tracks_visualization.png (visualization)")

if __name__ == "__main__":
    main()
