#!/usr/bin/env python3
"""
Simple generation script that works with the trained model
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neutron_vae.model import UltraHighFidelityVAE
from neutron_vae.config import *

def generate_samples(n_samples=10):
    """Generate neutron track samples"""
    print(f"🎲 Generating {n_samples} neutron track samples...")
    
    # Load the trained model
    model = UltraHighFidelityVAE().to(DEVICE)
    
    # Load the checkpoint
    checkpoint_path = 'neutron_vae_best.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from {checkpoint_path}")
            print(f"   Training loss: {checkpoint['loss']:.6f}")
            print(f"   Epoch: {checkpoint['epoch']}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model from {checkpoint_path} (old format)")
    else:
        print(f"❌ Model file {checkpoint_path} not found")
        return
    
    model.eval()
    
    # Generate samples
    generated_tracks = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Sample random latent vector
            z = torch.randn(1, LATENT_DIM, device=DEVICE)
            
            # Create condition with correct dimensions [x0, y0, z0, vx, vy, vz]
            # Using random initial positions and velocities
            condition = torch.randn(1, COND_DIM, device=DEVICE) * 0.1  # Small random values
            
            # Generate track
            recon = model.decode(z, condition)
            
            # Convert to numpy
            track = recon[0].cpu().numpy()
            generated_tracks.append(track)
            
            print(f"   Generated track {i+1}: shape {track.shape}")
    
    # Save generated tracks
    generated_tracks = np.array(generated_tracks)
    np.save('generated_neutron_tracks.npy', generated_tracks)
    
    print(f"\n✅ Generated {n_samples} tracks successfully!")
    print(f"📁 Saved as: generated_neutron_tracks.npy")
    print(f"📊 Shape: {generated_tracks.shape}")
    
    # Print some statistics
    print(f"\n📈 Generated Track Statistics:")
    print(f"   Mean track length: {np.mean([len(track) for track in generated_tracks]):.2f}")
    print(f"   Track shape: {generated_tracks[0].shape}")
    print(f"   Value range: [{generated_tracks.min():.4f}, {generated_tracks.max():.4f}]")
    
    return generated_tracks

def visualize_tracks(tracks, n_tracks=5):
    """Simple visualization of generated tracks"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot first n_tracks
        for i in range(min(n_tracks, len(tracks))):
            track = tracks[i]
            
            # 3D plot
            ax = fig.add_subplot(1, 3, 1, projection='3d')
            ax.plot(track[:, 0], track[:, 1], track[:, 2], alpha=0.7, linewidth=2)
            ax.set_title('3D Neutron Tracks')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # 2D projections
            axes[1].plot(track[:, 0], track[:, 1], alpha=0.7, linewidth=2)
            axes[1].set_title('XY Projection')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].grid(True)
            
            axes[2].plot(track[:, 0], track[:, 2], alpha=0.7, linewidth=2)
            axes[2].set_title('XZ Projection')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Z')
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('generated_tracks_visualization.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as: generated_tracks_visualization.png")
        
    except ImportError:
        print("📊 Matplotlib not available for visualization")
        print("📁 Generated tracks saved as: generated_neutron_tracks.npy")

if __name__ == "__main__":
    print("🚀 Neutron Track Generation")
    print("=" * 40)
    
    # Generate samples
    tracks = generate_samples(n_samples=20)
    
    if tracks is not None:
        # Visualize tracks
        visualize_tracks(tracks)
        
        print(f"\n🎉 Generation complete!")
        print(f"📁 Files created:")
        print(f"   - generated_neutron_tracks.npy (raw data)")
        print(f"   - generated_tracks_visualization.png (visualization)")
    else:
        print("❌ Generation failed!")
