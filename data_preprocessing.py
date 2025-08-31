import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class NeutronDataPreprocessor:
    def __init__(self, csv_file, sequence_length=50):
        self.csv_file = csv_file
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.feature_columns = ['x', 'y', 'z', 'ux', 'uy', 'uz', 'energy']
        
    def load_and_preprocess(self):
        """Load and preprocess the neutron tracking data"""
        print("Loading data...")
        df = pd.read_csv(self.csv_file)
        
        # Group by particle_id to get individual particle tracks
        particle_tracks = []
        for particle_id in df['particle_id'].unique():
            particle_data = df[df['particle_id'] == particle_id][self.feature_columns].values
            if len(particle_data) >= self.sequence_length:
                particle_tracks.append(particle_data)
        
        print(f"Found {len(particle_tracks)} particle tracks with sufficient length")
        
        # Normalize the data
        all_data = np.vstack(particle_tracks)
        normalized_data = self.scaler.fit_transform(all_data)
        
        # Create sequences
        sequences = []
        for track in particle_tracks:
            track_normalized = self.scaler.transform(track)
            for i in range(len(track_normalized) - self.sequence_length + 1):
                sequence = track_normalized[i:i + self.sequence_length]
                sequences.append(sequence)
        
        sequences = np.array(sequences)
        print(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        
        return sequences
    
    def inverse_transform(self, data):
        """Convert normalized data back to original scale"""
        return self.scaler.inverse_transform(data)
    
    def get_data_statistics(self):
        """Get statistics about the original data"""
        df = pd.read_csv(self.csv_file)
        stats = {
            'total_particles': df['particle_id'].nunique(),
            'total_points': len(df),
            'avg_track_length': df.groupby('particle_id').size().mean(),
            'energy_range': (df['energy'].min(), df['energy'].max()),
            'position_ranges': {
                'x': (df['x'].min(), df['x'].max()),
                'y': (df['y'].min(), df['y'].max()),
                'z': (df['z'].min(), df['z'].max())
            }
        }
        return stats

class NeutronDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def create_data_loaders(sequences, batch_size=32, train_split=0.8):
    """Create train and validation data loaders"""
    # Split data
    split_idx = int(len(sequences) * train_split)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    # Create datasets
    train_dataset = NeutronDataset(train_sequences)
    val_dataset = NeutronDataset(val_sequences)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def visualize_particle_tracks(df, num_particles=5):
    """Visualize particle tracks in 3D"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, particle_id in enumerate(df['particle_id'].unique()[:num_particles]):
        particle_data = df[df['particle_id'] == particle_id]
        ax.plot(particle_data['x'], particle_data['y'], particle_data['z'], 
                label=f'Particle {particle_id}', alpha=0.7)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Neutron Particle Tracks')
    ax.legend()
    plt.tight_layout()
    plt.savefig('particle_tracks.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Test the preprocessing
    preprocessor = NeutronDataPreprocessor('Sheet.csv', sequence_length=50)
    sequences = preprocessor.load_and_preprocess()
    
    # Get statistics
    stats = preprocessor.get_data_statistics()
    print("\nData Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(sequences, batch_size=32)
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")
        break
