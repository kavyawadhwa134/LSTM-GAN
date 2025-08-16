#!/usr/bin/env python3
"""
Quick Training Script for Neutron VAE
Fast training with reduced epochs and aggressive optimization
"""

import torch
import torch.nn as nn
import os
import sys
import time
import gc

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.model import UltraHighFidelityVAE, vae_loss
from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks
from neutron_vae.config import *

def setup_quick_gpu():
    """Setup GPU for quick training"""
    print("⚡ QUICK GPU SETUP")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return torch.device("cpu"), False
    
    device = torch.device("cuda")
    
    # Get GPU information
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / 1024**3
    
    print(f"🎯 GPU: {gpu_name}")
    print(f"📊 Total Memory: {total_memory_gb:.2f} GB")
    
    # Quick optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Set memory fraction
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✅ Quick optimizations enabled")
    
    return device, True

def train_quick():
    """Quick training with reduced epochs"""
    print("⚡ QUICK NEUTRON VAE TRAINING")
    print("=" * 60)
    
    # Setup GPU
    device, gpu_available = setup_quick_gpu()
    
    # Load and preprocess data
    print("\n📊 Loading data...")
    raw_tracks = load_and_split_tracks(CSV_FILE)
    tracks_norm, conditions, xyz_min, xyz_max = preprocess_tracks(raw_tracks)
    
    # Convert to torch tensors
    tracks_norm = torch.tensor(tracks_norm, dtype=torch.float32)
    conditions = torch.tensor(conditions, dtype=torch.float32)
    tracks_norm = 2 * tracks_norm - 1  # [0,1] → [-1,1]
    
    print(f"✅ Data loaded: {tracks_norm.shape}")
    
    # Use large batch size for speed
    quick_batch_size = 2048  # Large batch for speed
    
    # Move data to device
    tracks_norm = tracks_norm.to(device)
    conditions = conditions.to(device)
    
    # Create model
    model = UltraHighFidelityVAE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n🤖 Model created:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Device: {device}")
    print(f"   Quick batch size: {quick_batch_size}")
    
    # Optimizer with aggressive settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR * 3,  # Very high learning rate for fast convergence
        weight_decay=WEIGHT_DECAY, 
        betas=(0.9, 0.999)
    )
    
    # Fast learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, eta_min=LR/50  # Very fast scheduling
    )
    
    # Quick training settings
    quick_epochs = 5000  # 5000 epochs for better quality
    print(f"\n⚡ Quick Training Configuration:")
    print(f"   Epochs: {quick_epochs}")
    print(f"   Batch size: {quick_batch_size}")
    print(f"   Learning rate: {LR * 3}")
    print(f"   Device: {device}")
    
    # Training loop
    print(f"\n🔄 Starting quick training...")
    print("=" * 80)
    
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    # Early stopping
    patience = 500  # Increased patience for 5000 epochs
    no_improvement = 0
    
    for epoch in range(quick_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Use quick batch size
        idx = torch.randperm(len(tracks_norm))[:quick_batch_size]
        x_batch = tracks_norm[idx]
        c_batch = conditions[idx]
        
        # Forward pass
        recon, mu, logvar = model(x_batch, c_batch)
        loss, loss_components = vae_loss(recon, x_batch, mu, logvar, beta=BETA, alpha=ALPHA)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
        
        # Record loss
        loss_history.append(loss.item())
        
        # Update best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improvement = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'config': {
                    'LATENT_DIM': LATENT_DIM,
                    'HIDDEN_SIZE': HIDDEN_SIZE,
                    'NUM_LAYERS': NUM_LAYERS,
                    'COND_DIM': COND_DIM,
                    'QUICK_BATCH_SIZE': quick_batch_size
                }
            }, 'neutron_vae_quick_best.pth')
        else:
            no_improvement += 1
        
        # Progress reporting
        if epoch % 25 == 0:  # More frequent updates
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            eta = avg_time_per_epoch * (quick_epochs - epoch - 1)
            throughput = quick_batch_size / avg_time_per_epoch
            
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            memory_utilization = memory_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**2) * 100
            
            print(f"Epoch {epoch:4d}/{quick_epochs} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Best: {best_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"ETA: {eta/60:.1f}min | "
                  f"Speed: {throughput:.0f} samples/s")
            
            if gpu_available:
                print(f"   GPU Memory: {memory_allocated:.1f}MB / {memory_reserved:.1f}MB ({memory_utilization:.1f}%)")
        
        # Early stopping
        if no_improvement >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
        
        # Save checkpoint every 500 epochs
        if epoch % 500 == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'loss_history': loss_history,
                'quick_batch_size': quick_batch_size
            }, f'neutron_vae_quick_checkpoint_epoch_{epoch}.pth')
            print(f"💾 Checkpoint saved at epoch {epoch}")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n✅ Quick training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Final loss: {loss_history[-1]:.6f}")
    print(f"   Average speed: {quick_batch_size / (total_time/quick_epochs):.0f} samples/s")
    
    # Memory usage summary
    if gpu_available:
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        utilization = peak_memory / total_memory * 100
        
        print(f"\n🚀 Quick Performance Summary:")
        print(f"   Peak memory used: {peak_memory:.1f} MB")
        print(f"   Total GPU memory: {total_memory:.1f} MB")
        print(f"   Memory utilization: {utilization:.1f}%")
        print(f"   Quick batch size: {quick_batch_size}")
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_history[-1],
        'loss_history': loss_history,
        'quick_batch_size': quick_batch_size,
        'config': {
            'LATENT_DIM': LATENT_DIM,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'NUM_LAYERS': NUM_LAYERS,
            'COND_DIM': COND_DIM,
            'QUICK_BATCH_SIZE': quick_batch_size
        }
    }, 'neutron_vae_quick_final.pth')
    
    print(f"💾 Models saved:")
    print(f"   - neutron_vae_quick_best.pth")
    print(f"   - neutron_vae_quick_final.pth")
    
    return model, loss_history, quick_batch_size

def main():
    """Main function"""
    try:
        print("⚡ QUICK TRAINING FOR NEUTRON VAE")
        print("=" * 60)
        
        model, loss_history, quick_batch_size = train_quick()
        
        print(f"\n🎉 Quick training completed successfully!")
        print(f"✅ Quick batch size achieved: {quick_batch_size}")
        print(f"⚡ Fast training optimization complete!")
        
        # Plot training progress if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history)
            plt.title('Quick Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('quick_training.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Training plot saved as 'quick_training.png'")
        except ImportError:
            print("📊 Matplotlib not available, skipping plot generation")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
