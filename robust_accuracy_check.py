#!/usr/bin/env python3
"""
Robust accuracy check for neutron track data
Handles single real track vs multiple generated tracks
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

def normalize_to_unit_cube(tracks):
    """Normalize tracks to fit in a unit cube [-1,1]^3"""
    if isinstance(tracks, np.ndarray):
        # Generated data format: (n_tracks, n_points, 3)
        all_points = tracks.reshape(-1, 3)
    else:
        # Real data format: list of arrays
        all_points = np.vstack(tracks)
    
    # Find global bounds
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    
    # Normalize to [-1, 1]
    normalized_tracks = []
    
    if isinstance(tracks, np.ndarray):
        # Generated data
        normalized = np.zeros_like(tracks)
        for i in range(tracks.shape[0]):
            normalized[i] = 2 * (tracks[i] - min_vals) / (max_vals - min_vals) - 1
        return normalized
    else:
        # Real data
        for track in tracks:
            normalized_track = 2 * (track - min_vals) / (max_vals - min_vals) - 1
            normalized_tracks.append(normalized_track)
        return normalized_tracks

def calculate_track_statistics(tracks):
    """Calculate robust statistics for tracks"""
    if isinstance(tracks, np.ndarray):
        # Generated data: (n_tracks, n_points, 3)
        track_list = [tracks[i] for i in range(tracks.shape[0])]
    else:
        # Real data: list of arrays
        track_list = tracks
    
    stats = {
        'lengths': [],
        'curvatures': [],
        'smoothness': [],
        'spatial_range': [],
        'point_counts': []
    }
    
    for track in track_list:
        # Track length
        diffs = np.diff(track, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        stats['lengths'].append(length)
        
        # Point count
        stats['point_counts'].append(len(track))
        
        # Spatial range (volume)
        spatial_range = np.max(track, axis=0) - np.min(track, axis=0)
        volume = np.prod(spatial_range)
        stats['spatial_range'].append(volume)
        
        # Average curvature (simplified)
        if len(track) > 2:
            angles = []
            for j in range(1, len(track) - 1):
                v1 = track[j] - track[j-1]
                v2 = track[j+1] - track[j]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
            if angles:
                stats['curvatures'].append(np.mean(angles))
        
        # Smoothness (variance of second derivatives)
        if len(track) > 2:
            second_derivs = np.diff(track, n=2, axis=0)
            smoothness = np.var(second_derivs)
            stats['smoothness'].append(smoothness)
    
    # Convert to numpy arrays
    for key in stats:
        if stats[key]:
            stats[key] = np.array(stats[key])
    
    return stats

def calculate_similarity_score(real_stats, gen_stats):
    """Calculate similarity score between real and generated statistics"""
    print("\n📊 STATISTICS COMPARISON")
    print("=" * 50)
    
    total_score = 0
    n_metrics = 0
    
    for metric_name in ['lengths', 'curvatures', 'smoothness', 'spatial_range', 'point_counts']:
        if metric_name in real_stats and metric_name in gen_stats and len(real_stats[metric_name]) > 0 and len(gen_stats[metric_name]) > 0:
            real_mean = np.mean(real_stats[metric_name])
            gen_mean = np.mean(gen_stats[metric_name])
            
            # Handle zero values
            if abs(real_mean) < 1e-8:
                if abs(gen_mean) < 1e-8:
                    similarity = 1.0  # Both are zero
                else:
                    similarity = 0.0  # Only real is zero
            else:
                # Calculate relative difference
                relative_diff = abs(real_mean - gen_mean) / abs(real_mean)
                similarity = max(0, 1 - relative_diff)
            
            print(f"\n{metric_name.upper()}:")
            print(f"  Real mean:     {real_mean:.6f}")
            print(f"  Generated mean: {gen_mean:.6f}")
            print(f"  Similarity:    {similarity:.2%}")
            
            total_score += similarity
            n_metrics += 1
    
    if n_metrics > 0:
        average_score = total_score / n_metrics
        print(f"\n🏆 AVERAGE SIMILARITY SCORE: {average_score:.2%}")
        return average_score
    
    return 0

def calculate_spatial_coverage(real_tracks, gen_tracks):
    """Calculate spatial coverage similarity"""
    print("\n📍 SPATIAL COVERAGE ANALYSIS")
    print("=" * 50)
    
    # Get all points
    if isinstance(gen_tracks, np.ndarray):
        gen_points = gen_tracks.reshape(-1, 3)
    else:
        gen_points = np.vstack(gen_tracks)
    
    real_points = np.vstack(real_tracks)
    
    print(f"📊 Point distributions:")
    print(f"   Real points: {real_points.shape}")
    print(f"   Generated points: {gen_points.shape}")
    
    # Calculate bounds
    real_bounds = np.min(real_points, axis=0), np.max(real_points, axis=0)
    gen_bounds = np.min(gen_points, axis=0), np.max(gen_points, axis=0)
    
    print(f"\n📏 Spatial bounds:")
    print(f"   Real:     X[{real_bounds[0][0]:.4f}, {real_bounds[1][0]:.4f}], Y[{real_bounds[0][1]:.4f}, {real_bounds[1][1]:.4f}], Z[{real_bounds[0][2]:.4f}, {real_bounds[1][2]:.4f}]")
    print(f"   Generated: X[{gen_bounds[0][0]:.4f}, {gen_bounds[1][0]:.4f}], Y[{gen_bounds[0][1]:.4f}, {gen_bounds[1][1]:.4f}], Z[{gen_bounds[0][2]:.4f}, {gen_bounds[1][2]:.4f}]")
    
    # Calculate overlap for each dimension
    overlap_scores = []
    for dim in range(3):
        real_min, real_max = real_bounds[0][dim], real_bounds[1][dim]
        gen_min, gen_max = gen_bounds[0][dim], gen_bounds[1][dim]
        
        overlap_start = max(real_min, gen_min)
        overlap_end = min(real_max, gen_max)
        
        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            union = max(real_max, gen_max) - min(real_min, gen_min)
            overlap_score = overlap / union
        else:
            overlap_score = 0
        
        overlap_scores.append(overlap_score)
        print(f"   Dimension {['X', 'Y', 'Z'][dim]}: {overlap_score:.2%} overlap")
    
    spatial_score = np.mean(overlap_scores)
    print(f"\n🏆 SPATIAL COVERAGE SCORE: {spatial_score:.2%}")
    
    return spatial_score

def calculate_distribution_similarity(real_tracks, gen_tracks):
    """Calculate distribution similarity using histogram comparison"""
    print("\n📈 DISTRIBUTION SIMILARITY")
    print("=" * 50)
    
    # Get all points
    if isinstance(gen_tracks, np.ndarray):
        gen_points = gen_tracks.reshape(-1, 3)
    else:
        gen_points = np.vstack(gen_tracks)
    
    real_points = np.vstack(real_tracks)
    
    # Calculate histograms for each dimension
    dimension_scores = []
    for dim in range(3):
        real_hist, _ = np.histogram(real_points[:, dim], bins=20, density=True)
        gen_hist, _ = np.histogram(gen_points[:, dim], bins=20, density=True)
        
        # Calculate histogram similarity (cosine similarity)
        dot_product = np.dot(real_hist, gen_hist)
        real_norm = np.linalg.norm(real_hist)
        gen_norm = np.linalg.norm(gen_hist)
        
        if real_norm > 0 and gen_norm > 0:
            similarity = dot_product / (real_norm * gen_norm)
        else:
            similarity = 0
        
        dimension_scores.append(similarity)
        print(f"   Dimension {['X', 'Y', 'Z'][dim]}: {similarity:.2%} similarity")
    
    distribution_score = np.mean(dimension_scores)
    print(f"\n🏆 DISTRIBUTION SIMILARITY: {distribution_score:.2%}")
    
    return distribution_score

def main():
    print("🧪 ROBUST ACCURACY CHECK")
    print("=" * 60)
    
    # Load data
    real_tracks = load_real_data()
    if real_tracks is None:
        print("❌ Cannot proceed without real data!")
        return
    
    gen_tracks, data_type = load_generated_data()
    if gen_tracks is None:
        print("❌ Cannot proceed without generated data!")
        return
    
    print(f"\n📊 Data Summary:")
    print(f"   Real tracks: {len(real_tracks)}")
    print(f"   Generated tracks: {len(gen_tracks)}")
    print(f"   Generated data type: {data_type}")
    
    # Normalize both datasets to unit cube for fair comparison
    print(f"\n🔄 Normalizing data to unit cube...")
    normalized_real = normalize_to_unit_cube(real_tracks)
    normalized_gen = normalize_to_unit_cube(gen_tracks)
    
    # Calculate statistics
    print(f"\n📈 Calculating statistics...")
    real_stats = calculate_track_statistics(normalized_real)
    gen_stats = calculate_track_statistics(normalized_gen)
    
    # Calculate similarity scores
    stats_similarity = calculate_similarity_score(real_stats, gen_stats)
    spatial_similarity = calculate_spatial_coverage(normalized_real, normalized_gen)
    distribution_similarity = calculate_distribution_similarity(normalized_real, normalized_gen)
    
    # Final assessment
    print(f"\n🎯 FINAL ACCURACY ASSESSMENT")
    print("=" * 50)
    print(f"📊 Statistics Similarity: {stats_similarity:.2%}")
    print(f"📍 Spatial Coverage: {spatial_similarity:.2%}")
    print(f"📈 Distribution Similarity: {distribution_similarity:.2%}")
    
    # Overall score (weighted average)
    overall_score = 0.4 * stats_similarity + 0.3 * spatial_similarity + 0.3 * distribution_similarity
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
    
    print(f"\n🎉 Robust accuracy check complete!")
    print(f"📁 Generated data files:")
    print(f"   - accurate_neutron_tracks.npy (improved data)")
    print(f"   - accurate_neutron_tracks.csv (improved CSV)")

if __name__ == "__main__":
    main()
