# load_data.py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.config import SEQ_LEN, BOUNDS, N_TRACKS, SPATIAL_CONSTRAINTS

def analyze_real_data(csv_file=None):
    """Analyze real data characteristics for better training."""
    if csv_file is None:
        import os
        csv_file = os.path.join(os.path.dirname(__file__), 'Sheet.csv')
    
    df = pd.read_csv(csv_file)
    points = df[['x', 'y', 'z']].values
    
    # Calculate real data statistics
    x_range = [df.x.min(), df.x.max()]
    y_range = [df.y.min(), df.y.max()]
    z_range = [df.z.min(), df.z.max()]
    
    # Calculate track characteristics
    track_length = np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))
    track_curvature = calculate_track_curvature(points)
    track_smoothness = calculate_track_smoothness(points)
    
    print(f"📊 Real Data Analysis:")
    print(f"   X range: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
    print(f"   Y range: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
    print(f"   Z range: [{z_range[0]:.3f}, {z_range[1]:.3f}]")
    print(f"   Track length: {track_length:.3f}")
    print(f"   Track curvature: {track_curvature:.3f}")
    print(f"   Track smoothness: {track_smoothness:.3f}")
    
    return {
        'x_range': x_range,
        'y_range': y_range,
        'z_range': z_range,
        'track_length': track_length,
        'track_curvature': track_curvature,
        'track_smoothness': track_smoothness,
        'total_points': len(points)
    }

def calculate_track_curvature(points):
    """Calculate average curvature of the track."""
    if len(points) < 3:
        return 0.0
    
    curvatures = []
    for i in range(1, len(points) - 1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate curvature using cross product
        cross = np.cross(v1, v2)
        curvature = np.linalg.norm(cross) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        curvatures.append(curvature)
    
    return np.mean(curvatures) if curvatures else 0.0

def calculate_track_smoothness(points):
    """Calculate track smoothness (lower is smoother)."""
    if len(points) < 3:
        return 0.0
    
    # Calculate second derivatives
    second_derivatives = []
    for i in range(1, len(points) - 1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        second_deriv = np.linalg.norm(p3 - 2*p2 + p1)
        second_derivatives.append(second_deriv)
    
    return np.mean(second_derivatives) if second_derivatives else 0.0

def load_and_split_tracks(csv_file=None):
    if csv_file is None:
        import os
        csv_file = os.path.join(os.path.dirname(__file__), 'Sheet.csv')
    """
    Load CSV and split into N_TRACKS using large spatial jumps.
    """
    df = pd.read_csv(csv_file)
    points = df[['x', 'y', 'z']].values  # (1087, 3)

    # Compute step sizes
    diffs = np.linalg.norm(points[1:] - points[:-1], axis=1)
    jump_indices = np.argsort(diffs)[- (N_TRACKS - 1):]  # Find 9 largest jumps
    jump_indices.sort()

    # Split tracks
    tracks = []
    start = 0
    for idx in jump_indices:
        end = idx + 1
        tracks.append(points[start:end+1])
        start = end + 1
    tracks.append(points[start:])

    assert len(tracks) == N_TRACKS, f"Expected {N_TRACKS} tracks, got {len(tracks)}"
    print(f"✅ Loaded and split into {N_TRACKS} tracks.")
    return tracks

def preprocess_tracks(tracks, seq_len=SEQ_LEN):
    """
    Resample to fixed length, normalize, extract condition.
    Preserves real data characteristics better.
    """
    def resample(track, n=seq_len):
        if len(track) == 1:
            # If track has only one point, repeat it
            return np.tile(track, (n, 1))
        elif len(track) == 2:
            # If track has only two points, use linear interpolation
            t = np.linspace(0, 1, len(track))
            f = interp1d(t, track, axis=0, kind='linear')
            return f(np.linspace(0, 1, n))
        else:
            # For tracks with more than 2 points, use cubic interpolation
            t = np.linspace(0, 1, len(track))
            f = interp1d(t, track, axis=0, kind='cubic', bounds_error=False, fill_value='extrapolate')
            return f(np.linspace(0, 1, n))

    tracks_fixed = np.array([resample(track) for track in tracks])
    
    # Use real data bounds for normalization
    xyz_min = np.array(BOUNDS['min'])
    xyz_max = np.array(BOUNDS['max'])
    
    # Normalize to [0,1] range
    tracks_norm = (tracks_fixed - xyz_min) / (xyz_max - xyz_min)
    
    # Convert to [-1,1] range for VAE
    tracks_norm = 2 * tracks_norm - 1

    # Condition: start + initial direction (normalized)
    conditions = []
    for track in tracks_fixed:
        start = track[0]
        vel = (track[1] - track[0]) if len(track) > 1 else np.zeros(3)
        # Normalize condition to match input range
        start_norm = 2 * (start - xyz_min) / (xyz_max - xyz_min) - 1
        vel_norm = 2 * vel / (xyz_max - xyz_min)
        conditions.append(np.concatenate([start_norm, vel_norm]))
    conditions = np.array(conditions)

    return tracks_norm, conditions, xyz_min, xyz_max