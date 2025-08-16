#!/usr/bin/env python3
"""
GPU-Optimized Generation Script for Neutron VAE
Designed for Binder environments with GPU support
"""

import torch
import numpy as np
import os
import sys
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.model import UltraHighFidelityVAE
from neutron_vae.config import *

def setup_gpu():
    """Setup GPU for generation"""
    print("🔧 GPU SETUP FOR GENERATION")
    print("=" * 50)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🚀 GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        return device, True
    else:
        print("⚠️ CUDA not available, using CPU")
        return torch.device("cpu"), False

def load_model_gpu(model_path, device):
    """Load model with GPU support"""
    print(f"\n📥 Loading model from {model_path}")
    
    if os.path.exists(model_path):
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model state
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            # Direct model state dict
            model_state_dict = checkpoint
        
        # Create model
        model = UltraHighFidelityVAE().to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")
        
        return model
    else:
        print(f"❌ Model file not found: {model_path}")
        return None

def generate_tracks_gpu(model, device, n_tracks=20, seq_len=200, temperature=1.0):
    """Generate tracks using GPU acceleration"""
    print(f"\n🎲 GENERATING TRACKS WITH GPU")
    print("=" * 50)
    print(f"   Number of tracks: {n_tracks}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Temperature: {temperature}")
    print(f"   Device: {device}")
    
    # Generate conditions (initial positions and velocities)
    conditions = torch.randn(n_tracks, COND_DIM, device=device) * 0.1
    
    # Generate latent vectors with temperature scaling
    latent_vectors = torch.randn(n_tracks, LATENT_DIM, device=device) * temperature
    
    start_time = time.time()
    
    with torch.no_grad():
        # Generate tracks
        generated_tracks = model.decode(latent_vectors, conditions)
        
        # Apply temperature scaling to output
        if temperature != 1.0:
            generated_tracks = generated_tracks * temperature
    
    generation_time = time.time() - start_time
    
    print(f"✅ Generation completed in {generation_time:.3f} seconds")
    print(f"   Average time per track: {generation_time/n_tracks:.3f} seconds")
    
    if device.type == "cuda":
        memory_used = torch.cuda.memory_allocated() / 1024**2
        print(f"   GPU Memory used: {memory_used:.1f} MB")
    
    return generated_tracks.cpu().numpy()

def save_generated_data(tracks, filename_prefix="gpu_generated"):
    """Save generated tracks to files"""
    print(f"\n💾 SAVING GENERATED DATA")
    print("=" * 50)
    
    # Save as numpy array
    npy_filename = f"{filename_prefix}_neutron_tracks.npy"
    np.save(npy_filename, tracks)
    print(f"✅ Saved as numpy array: {npy_filename}")
    print(f"   Shape: {tracks.shape}")
    print(f"   Data type: {tracks.dtype}")
    
    # Save as CSV
    csv_filename = f"{filename_prefix}_neutron_tracks.csv"
    try:
        import pandas as pd
        
        # Flatten tracks for CSV
        csv_data = []
        for track_id, track in enumerate(tracks):
            for point_id, point in enumerate(track):
                csv_data.append({
                    'track_id': track_id,
                    'point_id': point_id,
                    'x': point[0],
                    'y': point[1],
                    'z': point[2]
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_filename, index=False)
        print(f"✅ Saved as CSV: {csv_filename}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
    except ImportError:
        print("⚠️ Pandas not available, skipping CSV export")
    
    return npy_filename, csv_filename

def analyze_generated_tracks(tracks):
    """Analyze the generated tracks"""
    print(f"\n📊 GENERATED TRACKS ANALYSIS")
    print("=" * 50)
    
    print(f"📈 Track Statistics:")
    print(f"   Number of tracks: {tracks.shape[0]}")
    print(f"   Points per track: {tracks.shape[1]}")
    print(f"   Dimensions: {tracks.shape[2]}")
    
    # Calculate track lengths
    track_lengths = []
    for track in tracks:
        diffs = np.diff(track, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        track_lengths.append(length)
    
    track_lengths = np.array(track_lengths)
    print(f"\n📏 Length Statistics:")
    print(f"   Mean length: {np.mean(track_lengths):.4f}")
    print(f"   Std length: {np.std(track_lengths):.4f}")
    print(f"   Min length: {np.min(track_lengths):.4f}")
    print(f"   Max length: {np.max(track_lengths):.4f}")
    
    # Calculate spatial bounds
    all_points = tracks.reshape(-1, 3)
    bounds = {
        'x': (np.min(all_points[:, 0]), np.max(all_points[:, 0])),
        'y': (np.min(all_points[:, 1]), np.max(all_points[:, 1])),
        'z': (np.min(all_points[:, 2]), np.max(all_points[:, 2]))
    }
    
    print(f"\n📍 Spatial Bounds:")
    for dim, (min_val, max_val) in bounds.items():
        print(f"   {dim.upper()}: [{min_val:.4f}, {max_val:.4f}]")
    
    # Calculate smoothness (variance of second derivatives)
    smoothness_scores = []
    for track in tracks:
        if len(track) > 2:
            second_derivs = np.diff(track, n=2, axis=0)
            smoothness = np.var(second_derivs)
            smoothness_scores.append(smoothness)
    
    if smoothness_scores:
        smoothness_scores = np.array(smoothness_scores)
        print(f"\n🔄 Smoothness Statistics:")
        print(f"   Mean smoothness: {np.mean(smoothness_scores):.6f}")
        print(f"   Std smoothness: {np.std(smoothness_scores):.6f}")
    
    return {
        'lengths': track_lengths,
        'bounds': bounds,
        'smoothness': smoothness_scores if smoothness_scores else None
    }

def main():
    """Main generation function"""
    print("🎲 GPU-OPTIMIZED NEUTRON TRACK GENERATION")
    print("=" * 60)
    
    # Setup GPU
    device, gpu_available = setup_gpu()
    
    # Try to load the best available model
    model_paths = [
        'neutron_vae_gpu_best.pth',
        'neutron_vae_best.pth',
        'neutron_vae_gpu_final.pth',
        'neutron_vae_final.pth'
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            model = load_model_gpu(path, device)
            if model is not None:
                break
    
    if model is None:
        print("❌ No trained model found!")
        print("   Please train the model first using train_gpu.py")
        return
    
    # Generation parameters
    n_tracks = 20
    seq_len = 200
    temperature = 1.0
    
    # Generate tracks
    generated_tracks = generate_tracks_gpu(
        model, device, 
        n_tracks=n_tracks, 
        seq_len=seq_len, 
        temperature=temperature
    )
    
    # Analyze generated tracks
    analysis = analyze_generated_tracks(generated_tracks)
    
    # Save generated data
    npy_file, csv_file = save_generated_data(generated_tracks)
    
    print(f"\n🎉 GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"✅ Generated {n_tracks} tracks with {seq_len} points each")
    print(f"✅ Files saved:")
    print(f"   - {npy_file}")
    print(f"   - {csv_file}")
    
    if gpu_available:
        print(f"✅ GPU acceleration used: {torch.cuda.get_device_name(0)}")
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"   Peak GPU memory: {peak_memory:.1f} MB")
    
    print(f"\n📊 Generation Summary:")
    print(f"   Average track length: {np.mean(analysis['lengths']):.4f}")
    print(f"   Spatial coverage: {analysis['bounds']}")
    
    # Clean up GPU memory
    if gpu_available:
        torch.cuda.empty_cache()
        print(f"🧹 GPU memory cleared")

if __name__ == "__main__":
    main()
