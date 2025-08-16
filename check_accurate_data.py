#!/usr/bin/env python3
"""
Comprehensive accuracy check for improved accurate neutron tracks
"""

import numpy as np
import os

def load_accurate_data():
    """Load the improved accurate neutron track data"""
    if os.path.exists('accurate_neutron_tracks.npy'):
        tracks = np.load('accurate_neutron_tracks.npy')
        print(f"✅ Loaded accurate data: {tracks.shape}")
        return tracks
    else:
        print("❌ Accurate data not found!")
        return None

def load_real_data():
    """Load real neutron track data"""
    try:
        csv_file = os.path.join('neutron_vae', 'Sheet.csv')
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            header = lines[0].strip().split(',')
            print(f"📋 Real data columns: {header}")
            
            # Parse data
            data = []
            for line in lines[1:]:
                values = line.strip().split(',')
                if len(values) >= 3:
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

def calculate_comprehensive_statistics(tracks):
    """Calculate comprehensive statistics for tracks"""
    if tracks is None or len(tracks) == 0:
        return {}
    
    stats = {
        'lengths': [],
        'curvatures': [],
        'smoothness': [],
        'velocities': [],
        'spatial_range': [],
        'value_range': [],
        'consistency': []
    }
    
    # If tracks is 2D (single track), reshape to 3D
    if len(tracks.shape) == 2:
        tracks = tracks.reshape(1, -1, 3)
    
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
            smoothness = np.var(second_derivs)
            stats['smoothness'].append(smoothness)
        
        # Average velocity
        if len(diffs) > 0:
            velocities = np.linalg.norm(diffs, axis=1)
            stats['velocities'].append(np.mean(velocities))
        
        # Spatial range
        spatial_range = np.max(track, axis=0) - np.min(track, axis=0)
        stats['spatial_range'].append(spatial_range)
        
        # Value range
        value_range = np.max(track) - np.min(track)
        stats['value_range'].append(value_range)
        
        # Consistency (variance of track properties)
        consistency = np.var(velocities) if len(velocities) > 0 else 0
        stats['consistency'].append(consistency)
    
    # Convert to numpy arrays
    for key in stats:
        if stats[key]:
            stats[key] = np.array(stats[key])
    
    return stats

def compare_improved_statistics(real_stats, accurate_stats):
    """Compare statistics between real and accurate data"""
    print("\n📊 IMPROVED STATISTICS COMPARISON")
    print("=" * 50)
    
    total_score = 0
    n_metrics = 0
    
    for stat_name in ['lengths', 'curvatures', 'smoothness', 'velocities', 'value_range']:
        if stat_name in real_stats and stat_name in accurate_stats and len(real_stats[stat_name]) > 0 and len(accurate_stats[stat_name]) > 0:
            real_mean = np.mean(real_stats[stat_name])
            real_std = np.std(real_stats[stat_name])
            accurate_mean = np.mean(accurate_stats[stat_name])
            accurate_std = np.std(accurate_stats[stat_name])
            
            print(f"\n{stat_name.upper()}:")
            print(f"  Real:     Mean={real_mean:.4f}, Std={real_std:.4f}")
            print(f"  Accurate: Mean={accurate_mean:.4f}, Std={accurate_std:.4f}")
            print(f"  Diff:     Mean={abs(real_mean-accurate_mean):.4f}, Std={abs(real_std-accurate_std):.4f}")
            
            # Calculate similarity score
            mean_similarity = 1 - abs(real_mean - accurate_mean) / (abs(real_mean) + 1e-8)
            std_similarity = 1 - abs(real_std - accurate_std) / (abs(real_std) + 1e-8)
            overall_similarity = (mean_similarity + std_similarity) / 2
            
            print(f"  Score:    {overall_similarity:.2%}")
            
            total_score += overall_similarity
            n_metrics += 1
    
    if n_metrics > 0:
        average_score = total_score / n_metrics
        print(f"\n🏆 AVERAGE ACCURACY SCORE: {average_score:.2%}")
        return average_score
    
    return 0

def calculate_quality_improvement():
    """Calculate quality improvement over original data"""
    print("\n📈 QUALITY IMPROVEMENT ANALYSIS")
    print("=" * 50)
    
    # Load original data
    if os.path.exists('generated_neutron_tracks.npy'):
        original_tracks = np.load('generated_neutron_tracks.npy')
        print(f"✅ Loaded original data: {original_tracks.shape}")
    else:
        print("❌ Original data not found!")
        return
    
    # Load accurate data
    accurate_tracks = load_accurate_data()
    if accurate_tracks is None:
        return
    
    # Calculate statistics
    original_stats = calculate_comprehensive_statistics(original_tracks)
    accurate_stats = calculate_comprehensive_statistics(accurate_tracks)
    
    print(f"\n📊 Quality Comparison:")
    
    # Compare key metrics
    for metric in ['smoothness', 'consistency', 'value_range']:
        if metric in original_stats and metric in accurate_stats:
            orig_mean = np.mean(original_stats[metric])
            acc_mean = np.mean(accurate_stats[metric])
            
            print(f"  {metric.upper()}:")
            print(f"    Original:  {orig_mean:.6f}")
            print(f"    Accurate:  {acc_mean:.6f}")
            
            # Lower is better for these metrics
            if orig_mean > 0:
                improvement = (orig_mean - acc_mean) / orig_mean
                print(f"    Improvement: {improvement:.2%}")

def main():
    print("🧪 COMPREHENSIVE ACCURACY CHECK FOR IMPROVED DATA")
    print("=" * 60)
    
    # Load accurate data
    accurate_tracks = load_accurate_data()
    
    if accurate_tracks is None:
        print("❌ Cannot proceed without accurate data!")
        return
    
    # Load real data if available
    real_tracks = load_real_data()
    
    if real_tracks is not None:
        # Calculate comprehensive statistics
        print("\n📊 Calculating comprehensive statistics...")
        accurate_stats = calculate_comprehensive_statistics(accurate_tracks)
        real_stats = calculate_comprehensive_statistics(real_tracks)
        
        # Compare statistics
        accuracy_score = compare_improved_statistics(real_stats, accurate_stats)
    else:
        print("\n📊 Accurate Data Statistics:")
        accurate_stats = calculate_comprehensive_statistics(accurate_tracks)
        for stat_name, values in accurate_stats.items():
            if len(values) > 0:
                print(f"  {stat_name}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")
    
    # Calculate quality improvement
    calculate_quality_improvement()
    
    print(f"\n🎉 Comprehensive accuracy check complete!")
    print(f"📁 Accurate data files:")
    print(f"   - accurate_neutron_tracks.npy (improved raw data)")
    print(f"   - accurate_neutron_tracks.csv (improved CSV format)")

if __name__ == "__main__":
    main()
