# visualize.py
import numpy as np
import matplotlib.pyplot as plt

def plot_tracks_simple(real_tracks, generated, xyz_min, xyz_max):
    """Simple 2D visualization using matplotlib"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot real tracks
    for track in real_tracks:
        ax1.plot(track[:, 0], track[:, 1], 'c-', alpha=0.5, linewidth=1)
        ax2.plot(track[:, 0], track[:, 2], 'c-', alpha=0.5, linewidth=1)
        ax3.plot(track[:, 1], track[:, 2], 'c-', alpha=0.5, linewidth=1)
    
    # Plot generated track
    ax1.plot(generated[:, 0], generated[:, 1], 'y-', linewidth=3, label='Generated')
    ax2.plot(generated[:, 0], generated[:, 2], 'y-', linewidth=3, label='Generated')
    ax3.plot(generated[:, 1], generated[:, 2], 'y-', linewidth=3, label='Generated')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('XY Projection')
    ax1.legend()
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('XZ Projection')
    ax2.legend()
    
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_title('YZ Projection')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('track_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'track_visualization.png'")

# Load and plot
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks
    raw_tracks = load_and_split_tracks()
    _, _, xyz_min, xyz_max = preprocess_tracks(raw_tracks)
    real_world = [t * (xyz_max - xyz_min) + xyz_min for t in raw_tracks]
    generated = np.loadtxt('generated_track.csv', delimiter=',', skiprows=1)
    plot_tracks_simple(real_world, generated, xyz_min, xyz_max)