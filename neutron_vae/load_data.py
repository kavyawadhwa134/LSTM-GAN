# load_data.py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.config import SEQ_LEN, BOUNDS, N_TRACKS

def load_and_split_tracks(csv_file=None):
    if csv_file is None:
        import os
        csv_file = os.path.join(os.path.dirname(__file__), 'Sheet.csv')
    """
    Load CSV and split into N_TRACKS using large spatial jumps.
    """
    df = pd.read_csv(csv_file)
    points = df[['x', 'y', 'z']].values  # (7382, 3)

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
    xyz_min = np.array(BOUNDS['min'])
    xyz_max = np.array(BOUNDS['max'])
    tracks_norm = (tracks_fixed - xyz_min) / (xyz_max - xyz_min)  # [0,1]

    # Condition: start + initial direction
    conditions = []
    for track in tracks_fixed:
        start = track[0]
        vel = (track[1] - track[0])
        conditions.append(np.concatenate([start, vel]))
    conditions = np.array(conditions)

    return tracks_norm, conditions, xyz_min, xyz_max