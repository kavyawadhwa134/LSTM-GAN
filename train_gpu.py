#!/usr/bin/env python3
"""
GPU-Optimized Training Script for Neutron VAE
Designed for Binder environments with GPU support
"""

import torch
import torch.nn as nn
import os
import sys
import time
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.model import UltraHighFidelityVAE, vae_loss
from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks
from neutron_vae.config import *

def setup_gpu():
    """Setup GPU optimization and check availability with maximum memory usage"""
    print("🔧 GPU SETUP WITH MAXIMUM MEMORY")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Get GPU info
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = total_memory / 1024**3
        
        print(f"🚀 GPU Available: {gpu_name}")
        print(f"   Total Memory: {gpu_memory_gb:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        # Set maximum memory fraction (95% of available memory)
        max_memory_fraction = 0.95
        torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
        print(f"   Memory fraction set to: {max_memory_fraction*100:.0f}%")
        
        # Clear any existing allocations
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Test GPU memory with maximum allocation
        try:
            # Test with larger tensor to verify maximum memory
            test_size = int(total_memory * 0.8 / 4)  # Use 80% of total memory
            test_tensor = torch.randn(test_size, device=device)
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
            print(f"   ✅ Memory test successful")
            print(f"   Allocated: {allocated_mb:.1f} MB")
            print(f"   Reserved: {reserved_mb:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ⚠️ GPU memory test failed: {e}")
            # Reduce memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
            print(f"   🔄 Reduced memory fraction to 80%")
        
        return device, True
    else:
        print("⚠️ CUDA not available, using CPU")
        return torch.device("cpu"), False

def optimize_batch_size(device, tracks_norm, conditions):
    """Find optimal batch size for GPU memory with maximum utilization"""
    print("\n🔍 OPTIMIZING BATCH SIZE FOR MAXIMUM MEMORY")
    print("=" * 50)
    
    if device.type == "cuda":
        # Test increasingly larger batch sizes for maximum memory usage
        batch_sizes = [32, 64, 128, 256, 512, 1024]
        optimal_batch_size = 32  # Default
        max_memory_used = 0
        
        for batch_size in batch_sizes:
            try:
                print(f"Testing batch size: {batch_size}")
                
                # Clear memory before test
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # Create test batch
                idx = torch.randperm(len(tracks_norm))[:batch_size]
                x_batch = tracks_norm[idx].to(device)
                c_batch = conditions[idx].to(device)
                
                # Create model
                model = UltraHighFidelityVAE().to(device)
                
                # Forward pass
                recon, mu, logvar = model(x_batch, c_batch)
                loss, _ = vae_loss(recon, x_batch, mu, logvar, beta=BETA, alpha=ALPHA)
                
                # Backward pass
                loss.backward()
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
                utilization = memory_used / total_memory * 100
                
                print(f"   ✅ Success - Allocated: {memory_used:.1f} MB, Reserved: {memory_reserved:.1f} MB ({utilization:.1f}%)")
                optimal_batch_size = batch_size
                max_memory_used = memory_used
                
                # Clean up
                del model, x_batch, c_batch, recon, mu, logvar, loss
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ❌ Out of memory at batch size {batch_size}")
                    break
                else:
                    print(f"   ❌ Error: {e}")
                    break
        
        print(f"🎯 Maximum batch size: {optimal_batch_size}")
        print(f"📊 Peak memory usage: {max_memory_used:.1f} MB")
        
        return optimal_batch_size
    else:
        print("Using default batch size for CPU")
        return BATCH_SIZE

def train_with_gpu():
    """Main training function with GPU optimization"""
    print("🧪 GPU-OPTIMIZED NEUTRON VAE TRAINING")
    print("=" * 60)
    
    # Setup GPU
    device, gpu_available = setup_gpu()
    
    # Load and preprocess data
    print("\n📊 Loading data...")
    raw_tracks = load_and_split_tracks(CSV_FILE)
    tracks_norm, conditions, xyz_min, xyz_max = preprocess_tracks(raw_tracks)
    
    # Convert to torch tensors
    tracks_norm = torch.tensor(tracks_norm, dtype=torch.float32)
    conditions = torch.tensor(conditions, dtype=torch.float32)
    tracks_norm = 2 * tracks_norm - 1  # [0,1] → [-1,1]
    
    print(f"✅ Data loaded: {tracks_norm.shape}")
    print(f"   Tracks: {len(tracks_norm)}")
    print(f"   Sequence length: {tracks_norm.shape[1]}")
    print(f"   Features: {tracks_norm.shape[2]}")
    
    # Optimize batch size for GPU
    optimal_batch_size = optimize_batch_size(device, tracks_norm, conditions)
    
    # Move data to device
    tracks_norm = tracks_norm.to(device)
    conditions = conditions.to(device)
    
    # Create model
    model = UltraHighFidelityVAE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🤖 Model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {device}")
    
    # Optimizer with GPU-optimized settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY, 
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=2, eta_min=LR/1000
    )
    
    # Training settings
    print(f"\n🚀 Training configuration:")
    print(f"   Epochs: {N_EPOCHS}")
    print(f"   Batch size: {optimal_batch_size}")
    print(f"   Learning rate: {LR}")
    print(f"   Beta: {BETA}")
    print(f"   Alpha: {ALPHA}")
    print(f"   Device: {device}")
    
    # Training loop
    print(f"\n🔄 Starting training...")
    print("=" * 80)
    
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Random batch
        idx = torch.randperm(len(tracks_norm))[:optimal_batch_size]
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
                    'COND_DIM': COND_DIM
                }
            }, 'neutron_vae_gpu_best.pth')
        
        # Progress reporting
        if epoch % 100 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            eta = avg_time_per_epoch * (N_EPOCHS - epoch - 1)
            
            print(f"Epoch {epoch:4d}/{N_EPOCHS} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Best: {best_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"ETA: {eta/60:.1f}min")
            
            if gpu_available:
                memory_used = torch.cuda.memory_allocated() / 1024**2
                print(f"   GPU Memory: {memory_used:.1f} MB")
        
        # Save checkpoint every 1000 epochs
        if epoch % 1000 == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'loss_history': loss_history
            }, f'neutron_vae_gpu_checkpoint_epoch_{epoch}.pth')
            print(f"💾 Checkpoint saved at epoch {epoch}")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Final loss: {loss_history[-1]:.6f}")
    
    # Save final model
    torch.save({
        'epoch': N_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_history[-1],
        'loss_history': loss_history,
        'config': {
            'LATENT_DIM': LATENT_DIM,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'NUM_LAYERS': NUM_LAYERS,
            'COND_DIM': COND_DIM
        }
    }, 'neutron_vae_gpu_final.pth')
    
    print(f"💾 Models saved:")
    print(f"   - neutron_vae_gpu_best.pth (best model)")
    print(f"   - neutron_vae_gpu_final.pth (final model)")
    
    # Performance summary
    if gpu_available:
        print(f"\n🚀 GPU Performance Summary:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        print(f"   Average time per epoch: {total_time/N_EPOCHS:.3f} seconds")
    
    return model, loss_history

def main():
    """Main function"""
    try:
        model, loss_history = train_with_gpu()
        print(f"\n🎉 GPU training completed successfully!")
        
        # Plot training progress if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history)
            plt.title('Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('gpu_training_loss.png')
            plt.close()
            print(f"📊 Training plot saved as 'gpu_training_loss.png'")
        except ImportError:
            print("📊 Matplotlib not available, skipping plot generation")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
