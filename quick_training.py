#!/usr/bin/env python3
"""
Quick Training Script for Neutron VAE with Spatial Constraints
Optimized for GPU acceleration and realistic track generation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neutron_vae.model import UltraHighFidelityVAE, vae_loss
from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks, analyze_real_data
from neutron_vae.config import *

def setup_quick_gpu():
    """Setup GPU for maximum performance."""
    if torch.cuda.is_available():
        print("⚡ QUICK GPU SETUP ==================================================")
        print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        print(f"📊 Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(MAX_GPU_MEMORY_FRACTION)
        
        print("✅ Quick optimizations enabled")
        return True
    else:
        print("❌ CUDA not available, using CPU")
        return False

def train_quick():
    """Quick training with spatial constraints."""
    print("⚡ QUICK TRAINING FOR NEUTRON VAE ============================================================")
    
    # Setup GPU
    use_gpu = setup_quick_gpu()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Analyze real data first
    print("\n📊 Analyzing real data characteristics...")
    real_data_stats = analyze_real_data()
    
    # Load and preprocess data
    print("\n📊 Loading data...")
    tracks = load_and_split_tracks()
    tracks_norm, conditions, xyz_min, xyz_max = preprocess_tracks(tracks)
    
    # Convert to tensors
    data = torch.FloatTensor(tracks_norm).to(device)
    conditions = torch.FloatTensor(conditions).to(device)
    
    print(f"✅ Data loaded: {data.shape}")
    
    # Create model
    model = UltraHighFidelityVAE().to(device)
    print(f"🤖 Model created: Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {device}")
    
    # Optimize batch size for GPU
    if use_gpu:
        # Find optimal batch size
        optimal_batch_size = find_optimal_batch_size(model, data, conditions, device)
        print(f"Quick batch size: {optimal_batch_size}")
    else:
        optimal_batch_size = BATCH_SIZE
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
    
    # Training configuration
    print(f"\n⚡ Quick Training Configuration:")
    print(f"   Epochs: {N_EPOCHS}")
    print(f"   Batch size: {optimal_batch_size}")
    print(f"   Learning rate: {LR}")
    print(f"   Device: {device}")
    
    # Training loop
    print(f"\n🔄 Starting quick training...")
    print("=" * 80)
    
    best_loss = float('inf')
    losses = []
    start_time = time.time()
    
    for epoch in range(N_EPOCHS):
        model.train()
        
        # Forward pass
        recon, mu, logvar = model(data, conditions)
        
        # Calculate loss with spatial constraints
        loss, loss_components = vae_loss(
            recon, data, mu, logvar, 
            beta=BETA, alpha=ALPHA, 
            gamma=0.05, delta=0.01,
            spatial_weight=SPATIAL_CONSTRAINTS['volume_weight']
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        
        optimizer.step()
        scheduler.step()
        
        # Track progress
        losses.append(loss_components['total'])
        
        if loss_components['total'] < best_loss:
            best_loss = loss_components['total']
            torch.save(model.state_dict(), 'neutron_vae_quick_best.pth')
        
        # Progress reporting
        if epoch % 25 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (N_EPOCHS - epoch - 1)
            
            # GPU memory info
            if use_gpu:
                gpu_memory = torch.cuda.memory_allocated() / 1e6
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e6
                gpu_util = gpu_memory / gpu_total * 100
                gpu_info = f"GPU Memory: {gpu_memory:.1f}MB / {gpu_total:.1f}MB ({gpu_util:.1f}%)"
            else:
                gpu_info = "CPU Training"
            
            print(f"Epoch {epoch}/{N_EPOCHS} | Loss: {loss_components['total']:.6f} | "
                  f"Best: {best_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"ETA: {eta/60:.1f}min | {gpu_info}")
            
            # Print spatial constraint losses
            if epoch % 100 == 0:
                print(f"   Spatial Losses - Volume: {loss_components['volume']:.6f}, "
                      f"Smoothness: {loss_components['smoothness']:.6f}, "
                      f"Length: {loss_components['length']:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), 'neutron_vae_quick_final.pth')
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\n✅ Quick training completed! Total time: {total_time/60:.1f} minutes")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Performance summary
    if use_gpu:
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e6
        memory_util = peak_memory / total_memory * 100
        print(f"\n🚀 Quick Performance Summary:")
        print(f"   Peak memory used: {peak_memory:.1f} MB")
        print(f"   Total GPU memory: {total_memory:.1f} MB")
        print(f"   Memory utilization: {memory_util:.1f}%")
        print(f"   Quick batch size: {optimal_batch_size}")
    
    print(f"\n💾 Models saved:")
    print(f"   - neutron_vae_quick_best.pth")
    print(f"   - neutron_vae_quick_final.pth")
    
    # Plot training progress
    plot_training_progress(losses, real_data_stats)
    
    print(f"\n🎉 Quick training completed successfully!")
    print(f"✅ Quick batch size achieved: {optimal_batch_size}")
    print(f"⚡ Fast training optimization complete!")
    print(f"📊 Training plot saved as 'quick_training.png'")

def find_optimal_batch_size(model, data, conditions, device):
    """Find optimal batch size for GPU memory."""
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    
    for batch_size in batch_sizes:
        try:
            # Test with this batch size
            test_data = data[:batch_size]
            test_conditions = conditions[:batch_size]
            
            model.train()
            recon, mu, logvar = model(test_data, test_conditions)
            loss, _ = vae_loss(recon, test_data, mu, logvar)
            loss.backward()
            
            # Clear gradients
            model.zero_grad()
            
            print(f"✅ Quick batch size achieved: {batch_size}")
            return batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Batch size {batch_size} exceeds GPU memory")
                continue
            else:
                raise e
    
    # Fallback to smaller batch size
    return 32

def plot_training_progress(losses, real_data_stats):
    """Plot training progress with real data comparison."""
    plt.figure(figsize=(12, 8))
    
    # Training loss
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Real data statistics
    plt.subplot(2, 2, 2)
    stats_names = ['Track Length', 'Curvature', 'Smoothness']
    stats_values = [
        real_data_stats['track_length'],
        real_data_stats['track_curvature'],
        real_data_stats['track_smoothness']
    ]
    plt.bar(stats_names, stats_values)
    plt.title('Real Data Characteristics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Spatial ranges
    plt.subplot(2, 2, 3)
    ranges = [
        real_data_stats['x_range'][1] - real_data_stats['x_range'][0],
        real_data_stats['y_range'][1] - real_data_stats['y_range'][0],
        real_data_stats['z_range'][1] - real_data_stats['z_range'][0]
    ]
    plt.bar(['X', 'Y', 'Z'], ranges)
    plt.title('Spatial Ranges')
    plt.ylabel('Range')
    
    # Loss components (last epoch)
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.9, f"Final Loss: {losses[-1]:.6f}", transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f"Best Loss: {min(losses):.6f}", transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"Total Points: {real_data_stats['total_points']}", transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Training Time: {len(losses)} epochs", transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_training.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train_quick()
