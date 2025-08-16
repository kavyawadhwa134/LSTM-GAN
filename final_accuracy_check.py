#!/usr/bin/env python3
"""
Final comprehensive accuracy check comparing generated vs real data
"""

import numpy as np
import os

def load_real_data():
    """Load and organize real neutron track data"""
    try:
        csv_file = os.path.join('neutron_vae', 'Sheet.csv')
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            header = lines[0].strip().split(',')
            print(f"📋 Real data columns: {header}")
            
            # Parse data and organize into tracks
            tracks = []
            current_track = []
            current_track_id = None
            
            for line in lines[1:]:
                values = line.strip().split(',')
                if len(values) >= 4:  # x, y, z, arc_length
                    try:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        arc_length = float(values[3])
                        
                        # Use arc_length to determine track boundaries
                        if current_track_id is None:
                            current_track_id = arc_length
                        
                        # If arc_length resets, it's a new track
                        if arc_length < current_track_id:
                            if current_track:
                                tracks.append(np.array(current_track))
                                current_track = []
                            current_track_id = arc_length
                        
                        current_track.append([x, y, z])
                        
                    except ValueError:
                        continue
            
            # Add the last track
            if current_track:
                tracks.append(np.array(current_track))
            
            print(f"✅ Loaded {len(tracks)} real tracks")
            for i, track in enumerate(tracks):
                print(f"   Track {i+1}: {len(track)} points")
            
            return tracks
        else:
            print(f"📝 Real data file not found: {csv_file}")
            return None
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        return None

def load_generated_data():
    """Load the most accurate generated data"""
    # Try accurate data first
    if os.path.exists('accurate_neutron_tracks.npy'):
        tracks = np.load('accurate_neutron_tracks.npy')
        print(f"✅ Loaded accurate generated data: {tracks.shape}")
        return tracks, "accurate"
    elif os.path.exists('generated_neutron_tracks.npy'):
        tracks = np.load('generated_neutron_tracks.npy')
        print(f"✅ Loaded generated data: {tracks.shape}")
        return tracks, "original"
    else:
        print("❌ No generated data found!")
        return None, None

def normalize_real_tracks(real_tracks):
    """Normalize real tracks to [-1,1] range for fair comparison"""
    print("🔄 Normalizing real tracks for comparison...")
    
    # Calculate global bounds from all real tracks
    all_points = np.vstack(real_tracks)
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)
    
    print(f"📊 Real data bounds:")
    print(f"   X: [{x_min:.4f}, {x_max:.4f}]")
    print(f"   Y: [{y_min:.4f}, {y_max:.4f}]")
    print(f"   Z: [{z_min:.4f}, {z_max:.4f}]")
    
    # Normalize each track
    normalized_tracks = []
    for track in real_tracks:
        normalized_track = np.zeros_like(track)
        
        # Normalize to [0,1] first
        normalized_track[:, 0] = (track[:, 0] - x_min) / (x_max - x_min)
        normalized_track[:, 1] = (track[:, 1] - y_min) / (y_max - y_min)
        normalized_track[:, 2] = (track[:, 2] - z_min) / (z_max - z_min)
        
        # Convert to [-1,1]
        normalized_track = 2 * normalized_track - 1
        
        normalized_tracks.append(normalized_track)
    
    return normalized_tracks

def calculate_track_metrics(tracks):
    """Calculate comprehensive metrics for tracks"""
    # Handle different input formats
    if isinstance(tracks, np.ndarray):
        # Generated data format: (n_tracks, n_points, 3)
        track_list = [tracks[i] for i in range(tracks.shape[0])]
    elif isinstance(tracks, list):
        # Real data format: list of arrays
        track_list = tracks
    else:
        print(f"❌ Unknown tracks format: {type(tracks)}")
        return {}
    
    if len(track_list) == 0:
        return {}
    
    metrics = {
        'lengths': [],
        'curvatures': [],
        'smoothness': [],
        'velocities': [],
        'spatial_coverage': [],
        'point_density': []
    }
    
    for track in track_list:
        # Track length
        diffs = np.diff(track, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        metrics['lengths'].append(length)
        
        # Average curvature
        if len(track) > 2:
            curvatures = []
            for j in range(1, len(track) - 1):
                v1 = track[j] - track[j-1]
                v2 = track[j+1] - track[j]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    curvatures.append(angle)
            if curvatures:
                metrics['curvatures'].append(np.mean(curvatures))
        
        # Smoothness (variance of second derivatives)
        if len(track) > 2:
            second_derivs = np.diff(track, n=2, axis=0)
            smoothness = np.var(second_derivs)
            metrics['smoothness'].append(smoothness)
        
        # Average velocity
        if len(diffs) > 0:
            velocities = np.linalg.norm(diffs, axis=1)
            metrics['velocities'].append(np.mean(velocities))
        
        # Spatial coverage (volume covered by track)
        spatial_range = np.max(track, axis=0) - np.min(track, axis=0)
        coverage = np.prod(spatial_range)
        metrics['spatial_coverage'].append(coverage)
        
        # Point density (points per unit length)
        density = len(track) / (length + 1e-8)
        metrics['point_density'].append(density)
    
    # Convert to numpy arrays
    for key in metrics:
        if metrics[key]:
            metrics[key] = np.array(metrics[key])
    
    return metrics

def compare_metrics(real_metrics, gen_metrics):
    """Compare metrics between real and generated data"""
    print("\n📊 METRICS COMPARISON")
    print("=" * 50)
    
    total_score = 0
    n_metrics = 0
    
    for metric_name in ['lengths', 'curvatures', 'smoothness', 'velocities', 'spatial_coverage', 'point_density']:
        if metric_name in real_metrics and metric_name in gen_metrics and len(real_metrics[metric_name]) > 0 and len(gen_metrics[metric_name]) > 0:
            real_mean = np.mean(real_metrics[metric_name])
            real_std = np.std(real_metrics[metric_name])
            gen_mean = np.mean(gen_metrics[metric_name])
            gen_std = np.std(gen_metrics[metric_name])
            
            print(f"\n{metric_name.upper()}:")
            print(f"  Real:     Mean={real_mean:.4f}, Std={real_std:.4f}")
            print(f"  Generated: Mean={gen_mean:.4f}, Std={gen_std:.4f}")
            print(f"  Diff:     Mean={abs(real_mean-gen_mean):.4f}, Std={abs(real_std-gen_std):.4f}")
            
            # Calculate similarity score
            mean_similarity = 1 - abs(real_mean - gen_mean) / (abs(real_mean) + 1e-8)
            std_similarity = 1 - abs(real_std - gen_std) / (abs(real_std) + 1e-8)
            overall_similarity = (mean_similarity + std_similarity) / 2
            
            print(f"  Score:    {overall_similarity:.2%}")
            
            total_score += overall_similarity
            n_metrics += 1
    
    if n_metrics > 0:
        average_score = total_score / n_metrics
        print(f"\n🏆 OVERALL ACCURACY SCORE: {average_score:.2%}")
        return average_score
    
    return 0

def calculate_spatial_similarity(real_tracks, gen_tracks):
    """Calculate spatial similarity between real and generated tracks"""
    print("\n📍 SPATIAL SIMILARITY ANALYSIS")
    print("=" * 50)
    
    # Handle different input formats
    if isinstance(gen_tracks, np.ndarray):
        # Generated data format: (n_tracks, n_points, 3)
        gen_points = gen_tracks.reshape(-1, 3)
    elif isinstance(gen_tracks, list):
        # Real data format: list of arrays
        gen_points = np.vstack(gen_tracks)
    else:
        print(f"❌ Unknown generated tracks format: {type(gen_tracks)}")
        return 0
    
    # Real tracks are already in list format
    real_points = np.vstack(real_tracks)
    
    print(f"📊 Point distributions:")
    print(f"   Real points: {real_points.shape}")
    print(f"   Generated points: {gen_points.shape}")
    
    # Calculate spatial statistics
    real_bounds = np.min(real_points, axis=0), np.max(real_points, axis=0)
    gen_bounds = np.min(gen_points, axis=0), np.max(gen_points, axis=0)
    
    print(f"\n📏 Spatial bounds:")
    print(f"   Real:     X[{real_bounds[0][0]:.4f}, {real_bounds[1][0]:.4f}], Y[{real_bounds[0][1]:.4f}, {real_bounds[1][1]:.4f}], Z[{real_bounds[0][2]:.4f}, {real_bounds[1][2]:.4f}]")
    print(f"   Generated: X[{gen_bounds[0][0]:.4f}, {gen_bounds[1][0]:.4f}], Y[{gen_bounds[0][1]:.4f}, {gen_bounds[1][1]:.4f}], Z[{gen_bounds[0][2]:.4f}, {gen_bounds[1][2]:.4f}]")
    
    # Calculate overlap scores
    overlap_scores = []
    for dim in range(3):
        real_range = real_bounds[1][dim] - real_bounds[0][dim]
        gen_range = gen_bounds[1][dim] - gen_bounds[0][dim]
        
        overlap_start = max(real_bounds[0][dim], gen_bounds[0][dim])
        overlap_end = min(real_bounds[1][dim], gen_bounds[1][dim])
        
        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            union = max(real_bounds[1][dim], gen_bounds[1][dim]) - min(real_bounds[0][dim], gen_bounds[0][dim])
            overlap_score = overlap / union
        else:
            overlap_score = 0
        
        overlap_scores.append(overlap_score)
        print(f"   Dimension {['X', 'Y', 'Z'][dim]}: {overlap_score:.2%} overlap")
    
    spatial_similarity = np.mean(overlap_scores)
    print(f"\n🏆 SPATIAL SIMILARITY: {spatial_similarity:.2%}")
    
    return spatial_similarity

def main():
    print("🧪 FINAL COMPREHENSIVE ACCURACY CHECK")
    print("=" * 60)
    
    # Load real data
    real_tracks = load_real_data()
    if real_tracks is None:
        print("❌ Cannot proceed without real data!")
        return
    
    # Load generated data
    gen_tracks, data_type = load_generated_data()
    if gen_tracks is None:
        print("❌ Cannot proceed without generated data!")
        return
    
    print(f"\n📊 Data Summary:")
    print(f"   Real tracks: {len(real_tracks)}")
    print(f"   Generated tracks: {len(gen_tracks)}")
    print(f"   Generated data type: {data_type}")
    
    # Normalize real tracks for fair comparison
    normalized_real_tracks = normalize_real_tracks(real_tracks)
    
    # Calculate metrics
    print(f"\n📈 Calculating metrics...")
    real_metrics = calculate_track_metrics(normalized_real_tracks)
    gen_metrics = calculate_track_metrics(gen_tracks)
    
    # Compare metrics
    accuracy_score = compare_metrics(real_metrics, gen_metrics)
    
    # Calculate spatial similarity
    spatial_similarity = calculate_spatial_similarity(normalized_real_tracks, gen_tracks)
    
    # Final accuracy assessment
    print(f"\n🎯 FINAL ACCURACY ASSESSMENT")
    print("=" * 50)
    print(f"📊 Metrics Accuracy: {accuracy_score:.2%}")
    print(f"📍 Spatial Similarity: {spatial_similarity:.2%}")
    
    # Overall score (weighted average)
    overall_score = 0.7 * accuracy_score + 0.3 * spatial_similarity
    print(f"🏆 OVERALL ACCURACY SCORE: {overall_score:.2%}")
    
    # Quality assessment
    if overall_score >= 0.8:
        quality = "EXCELLENT"
    elif overall_score >= 0.6:
        quality = "GOOD"
    elif overall_score >= 0.4:
        quality = "FAIR"
    else:
        quality = "NEEDS IMPROVEMENT"
    
    print(f"🎯 Quality Assessment: {quality}")
    
    print(f"\n🎉 Final accuracy check complete!")
    print(f"📁 Generated data files:")
    print(f"   - accurate_neutron_tracks.npy (improved data)")
    print(f"   - accurate_neutron_tracks.csv (improved CSV)")

if __name__ == "__main__":
    main()
