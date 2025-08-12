"""
Neutron Trajectory Data Preprocessor for LSTM-TrajGAN
Converts neutron trajectory data to format compatible with LSTM-TrajGAN
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

def preprocess_neutron_data(csv_path, max_length=200, train_test_split=0.8):
    """
    Preprocess neutron trajectory data for LSTM-TrajGAN
    
    Args:
        csv_path: Path to neutron trajectory CSV file
        max_length: Maximum sequence length for trajectories
        train_test_split: Ratio for train/test split
    """
    # Load the data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trajectory points")
    
    # Normalize coordinates to [-1, 1] range for better GAN performance
    scaler_xyz = MinMaxScaler(feature_range=(-1, 1))
    df[['x_norm', 'y_norm', 'z_norm']] = scaler_xyz.fit_transform(df[['x', 'y', 'z']])
    
    # Save scaler parameters for later use
    np.save('data/scaler_params.npy', {
        'data_min_': scaler_xyz.data_min_,
        'data_max_': scaler_xyz.data_max_,
        'scale_': scaler_xyz.scale_,
        'min_': scaler_xyz.min_
    })
    
    # Create trajectory segments
    total_points = len(df)
    trajectories = []
    
    # Create overlapping segments of trajectory
    step_size = max_length // 2  # 50% overlap
    for start_idx in range(0, total_points - max_length + 1, step_size):
        end_idx = start_idx + max_length
        segment = df.iloc[start_idx:end_idx].copy()
        segment['tid'] = len(trajectories)  # trajectory ID
        trajectories.append(segment)
    
    # Combine all segments
    all_segments = pd.concat(trajectories, ignore_index=True)
    
    # Split into train and test
    n_trajectories = len(trajectories)
    n_train = int(n_trajectories * train_test_split)
    
    train_tids = list(range(n_train))
    test_tids = list(range(n_train, n_trajectories))
    
    train_df = all_segments[all_segments['tid'].isin(train_tids)].copy()
    test_df = all_segments[all_segments['tid'].isin(test_tids)].copy()
    
    # Save train and test CSV files
    train_df.to_csv('data/neutron_train.csv', index=False)
    test_df.to_csv('data/neutron_test.csv', index=False)
    
    print(f"Created {n_train} training trajectories and {len(test_tids)} test trajectories")
    print(f"Training data: {len(train_df)} points")
    print(f"Test data: {len(test_df)} points")
    
    return train_df, test_df, scaler_xyz

def convert_to_npy_format(df, save_path, max_length=200):
    """
    Convert trajectory DataFrame to numpy format for LSTM-TrajGAN
    """
    # Group by trajectory ID
    trajectories = []
    for tid in df['tid'].unique():
        traj_data = df[df['tid'] == tid].copy()
        
        # Extract features
        xyz_coords = traj_data[['x_norm', 'y_norm', 'z_norm']].values
        arc_lengths = traj_data['arc_length'].values
        
        # Normalize arc_length to [0, 1] for each trajectory
        if len(arc_lengths) > 1:
            arc_lengths = (arc_lengths - arc_lengths.min()) / (arc_lengths.max() - arc_lengths.min())
        else:
            arc_lengths = np.array([0.0])
        
        trajectories.append({
            'coords': xyz_coords,
            'arc_length': arc_lengths,
            'length': len(xyz_coords)
        })
    
    # Convert to format expected by LSTM-TrajGAN
    # Features: [coordinates, arc_length_discretized, mask]
    coords_list = []
    arc_length_list = []
    mask_list = []
    
    for traj in trajectories:
        # Pad/truncate coordinates
        coords = traj['coords']
        if len(coords) < max_length:
            # Pad with zeros
            padded_coords = np.zeros((max_length, 3))
            padded_coords[:len(coords)] = coords
            coords = padded_coords
        else:
            coords = coords[:max_length]
        
        # Create arc length bins (similar to hour discretization)
        arc_bins = np.linspace(0, 1, 10)  # 10 bins for arc length
        arc_length_discrete = np.digitize(traj['arc_length'], arc_bins) - 1
        arc_length_discrete = np.clip(arc_length_discrete, 0, 9)
        
        # Pad arc length
        if len(arc_length_discrete) < max_length:
            padded_arc = np.zeros(max_length, dtype=int)
            padded_arc[:len(arc_length_discrete)] = arc_length_discrete
            arc_length_discrete = padded_arc
        else:
            arc_length_discrete = arc_length_discrete[:max_length]
        
        # Create mask (1 for valid points, 0 for padding)
        mask = np.zeros((max_length, 1))
        actual_length = min(traj['length'], max_length)
        mask[:actual_length] = 1
        
        coords_list.append(coords)
        arc_length_list.append(np.eye(10)[arc_length_discrete])  # One-hot encode
        mask_list.append(mask)
    
    # Convert to numpy arrays
    coords_array = np.array(coords_list)  # Shape: (n_trajectories, max_length, 3)
    arc_length_array = np.array(arc_length_list)  # Shape: (n_trajectories, max_length, 10)
    mask_array = np.array(mask_list)  # Shape: (n_trajectories, max_length, 1)
    
    # Save as numpy arrays separately in a dictionary format
    final_data = {
        'coordinates': coords_array,
        'arc_length': arc_length_array, 
        'mask': mask_array
    }
    
    np.savez(save_path.replace('.npy', '.npz'), **final_data)
    print(f"Saved processed data to {save_path}")
    return final_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/Sheet.csv")
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--train_test_split", type=float, default=0.8)
    args = parser.parse_args()
    
    # Preprocess the data
    train_df, test_df, scaler = preprocess_neutron_data(
        args.input_csv, 
        args.max_length, 
        args.train_test_split
    )
    
    # Convert to numpy format
    convert_to_npy_format(train_df, 'data/neutron_train_final.npy', args.max_length)
    convert_to_npy_format(test_df, 'data/neutron_test_final.npy', args.max_length)
    
    print("Data preprocessing completed!")
