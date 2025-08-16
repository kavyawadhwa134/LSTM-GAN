#!/usr/bin/env python3
"""
Optimized Training Script for Neutron VAE with Spatial Constraints
Features: Batch size 2048, checkpoint save/load, GPU optimization
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
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neutron_vae.model import UltraHighFidelityVAE, vae_loss
from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks, analyze_real_data
from neutron_vae.config import *

# Fixed optimal batch size for maximum GPU utilization
OPTIMAL_BATCH_SIZE = 2048
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 500  # Save checkpoint every 500 epochs

def setup_gpu():
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

def create_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"📁 Created checkpoint directory: {CHECKPOINT_DIR}")

def save_checkpoint(model, optimizer, scheduler, epoch, loss, losses, real_data_stats, filename=None):
    """Save training checkpoint."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
    
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'losses': losses,
        'real_data_stats': real_data_stats,
        'config': {
            'LATENT_DIM': LATENT_DIM,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'NUM_LAYERS': NUM_LAYERS,
            'COND_DIM': COND_DIM,
            'BATCH_SIZE': OPTIMAL_BATCH_SIZE,
            'LR': LR,
            'BETA': BETA,
            'ALPHA': ALPHA
        },
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")
    return filepath

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, 0, []
    
    print(f"📥 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    losses = checkpoint.get('losses', [])
    real_data_stats = checkpoint.get('real_data_stats', {})
    
    print(f"✅ Checkpoint loaded successfully")
    print(f"   Epoch: {epoch}")
    print(f"   Loss: {checkpoint['loss']:.6f}")
    print(f"   Config: {checkpoint['config']}")
    
    return checkpoint, epoch, losses

def find_latest_checkpoint():
    """Find the latest checkpoint file."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)
    return os.path.join(CHECKPOINT_DIR, checkpoint_files[0])

def train_with_checkpoints():
    """Main training function with checkpoint support."""
    print("⚡ QUICK TRAINING FOR NEUTRON VAE ============================================================")
    
    # Setup GPU
    use_gpu = setup_gpu()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Create checkpoint directory
    create_checkpoint_dir()
    
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
    
    # Use optimal batch size
    optimal_batch_size = OPTIMAL_BATCH_SIZE
    print(f"🚀 Using optimal batch size: {optimal_batch_size}")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
    
    # Training configuration
    print(f"\n⚡ Quick Training Configuration:")
    print(f"   Epochs: {N_EPOCHS}")
    print(f"   Batch size: {optimal_batch_size}")
    print(f"   Learning rate: {LR}")
    print(f"   Device: {device}")
    print(f"   Checkpoint interval: {CHECKPOINT_INTERVAL} epochs")
    
    # Check for existing checkpoint
    start_epoch = 0
    losses = []
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"\n🔄 Found existing checkpoint: {latest_checkpoint}")
        response = input("Do you want to resume training? (y/n): ").lower().strip()
        if response == 'y':
            checkpoint, start_epoch, losses = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
            if checkpoint is None:
                start_epoch = 0
                losses = []
        else:
            print("Starting fresh training...")
    
    # Training loop
    print(f"\n🔄 Starting quick training...")
    print("=" * 80)
    
    best_loss = float('inf') if not losses else min(losses)
    start_time = time.time()
    
    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        
        # Use mini-batches for faster training
        total_loss = 0
        num_batches = 0
        
        # Create mini-batches
        batch_size = min(optimal_batch_size, len(data))
        indices = torch.randperm(len(data))
        
        for i in range(0, len(data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_data = data[batch_indices]
            batch_conditions = conditions[batch_indices]
            
            # Forward pass
            recon, mu, logvar = model(batch_data, batch_conditions)
            
            # Calculate loss with spatial constraints
            loss, loss_components = vae_loss(
                recon, batch_data, mu, logvar, 
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
            
            total_loss += loss_components['total']
            num_batches += 1
        
        # Average loss for the epoch
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        # Track progress
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'neutron_vae_quick_best.pth')
        
        # Progress reporting
        if epoch % 25 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch - start_epoch + 1)) * (N_EPOCHS - epoch - 1)
            
            # GPU memory info
            if use_gpu:
                gpu_memory = torch.cuda.memory_allocated() / 1e6
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e6
                gpu_util = gpu_memory / gpu_total * 100
                gpu_info = f"GPU Memory: {gpu_memory:.1f}MB / {gpu_total:.1f}MB ({gpu_util:.1f}%)"
            else:
                gpu_info = "CPU Training"
            
            print(f"Epoch {epoch}/{N_EPOCHS} | Loss: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"ETA: {eta/60:.1f}min | {gpu_info}")
            
            # Print spatial constraint losses
            if epoch % 100 == 0:
                print(f"   Spatial Losses - Volume: {loss_components['volume']:.6f}, "
                      f"Smoothness: {loss_components['smoothness']:.6f}, "
                      f"Length: {loss_components['length']:.6f}")
        
        # Save checkpoint periodically
        if epoch % CHECKPOINT_INTERVAL == 0 and epoch > 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, losses, real_data_stats)
    
    # Save final model and checkpoint
    torch.save(model.state_dict(), 'neutron_vae_quick_final.pth')
    save_checkpoint(model, optimizer, scheduler, N_EPOCHS-1, avg_loss, losses, real_data_stats, "final_checkpoint.pth")
    
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
        print(f"   Optimal batch size: {optimal_batch_size}")
    
    print(f"\n💾 Models saved:")
    print(f"   - neutron_vae_quick_best.pth")
    print(f"   - neutron_vae_quick_final.pth")
    print(f"   - checkpoints/ (checkpoint files)")
    
    # Plot training progress
    plot_training_progress(losses, real_data_stats)
    
    print(f"\n🎉 Quick training completed successfully!")
    print(f"✅ Optimal batch size achieved: {optimal_batch_size}")
    print(f"⚡ Fast training optimization complete!")
    print(f"📊 Training plot saved as 'quick_training.png'")

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
    plt.text(0.1, 0.5, f"Batch Size: {OPTIMAL_BATCH_SIZE}", transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_training.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train_with_checkpoints()
