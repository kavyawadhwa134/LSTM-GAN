#!/usr/bin/env python3
"""
Maximize CUDA Memory Allocation for Neutron VAE
Optimizes GPU memory usage to the maximum available capacity
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

def setup_maximum_cuda_memory():
    """Setup CUDA to use maximum available memory"""
    print("🚀 MAXIMIZING CUDA MEMORY ALLOCATION")
    print("=" * 60)
    
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
    
    # Set maximum memory fraction (use 95% of available memory)
    max_memory_fraction = MAX_GPU_MEMORY_FRACTION
    max_memory_bytes = int(total_memory * max_memory_fraction)
    max_memory_gb = max_memory_bytes / 1024**3
    
    print(f"🎯 Target Memory Usage: {max_memory_gb:.2f} GB ({max_memory_fraction*100:.0f}%)")
    
    # Clear any existing allocations
    torch.cuda.empty_cache()
    gc.collect()
    
    # Set memory fraction
    torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
    
    # Enable memory optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Test memory allocation
    try:
        # Allocate a large tensor to test memory
        test_size = int(max_memory_bytes * 0.8 / 4)  # Use 80% of target memory
        test_tensor = torch.randn(test_size, device=device)
        allocated_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"✅ Memory test successful: {allocated_mb:.1f} MB allocated")
        del test_tensor
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"⚠️ Memory test failed: {e}")
        # Reduce memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)
        print("🔄 Reduced memory fraction to 80%")
    
    return device, True

def find_optimal_batch_size_max_memory(device, tracks_norm, conditions):
    """Find the maximum possible batch size for GPU memory"""
    print("\n🔍 FINDING MAXIMUM BATCH SIZE")
    print("=" * 50)
    
    if device.type != "cuda":
        return BATCH_SIZE
    
    # Test increasingly larger batch sizes
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    optimal_batch_size = 32
    max_memory_used = 0
    
    for batch_size in batch_sizes:
        try:
            print(f"Testing batch size: {batch_size}")
            
            # Clear memory
            torch.cuda.empty_cache()
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
            
            print(f"   ✅ Success - Allocated: {memory_used:.1f} MB, Reserved: {memory_reserved:.1f} MB")
            
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

def train_with_maximum_memory():
    """Train with maximum GPU memory utilization"""
    print("🧪 MAXIMUM MEMORY NEUTRON VAE TRAINING")
    print("=" * 60)
    
    # Setup maximum CUDA memory
    device, gpu_available = setup_maximum_cuda_memory()
    
    # Load and preprocess data
    print("\n📊 Loading data...")
    raw_tracks = load_and_split_tracks(CSV_FILE)
    tracks_norm, conditions, xyz_min, xyz_max = preprocess_tracks(raw_tracks)
    
    # Convert to torch tensors
    tracks_norm = torch.tensor(tracks_norm, dtype=torch.float32)
    conditions = torch.tensor(conditions, dtype=torch.float32)
    tracks_norm = 2 * tracks_norm - 1  # [0,1] → [-1,1]
    
    print(f"✅ Data loaded: {tracks_norm.shape}")
    
    # Find maximum batch size
    max_batch_size = find_optimal_batch_size_max_memory(device, tracks_norm, conditions)
    
    # Move data to device
    tracks_norm = tracks_norm.to(device)
    conditions = conditions.to(device)
    
    # Create model with maximum memory optimization
    model = UltraHighFidelityVAE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n🤖 Model created:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Device: {device}")
    print(f"   Maximum batch size: {max_batch_size}")
    
    # Optimizer with maximum memory settings
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
    print(f"\n🚀 Maximum Memory Training Configuration:")
    print(f"   Epochs: {N_EPOCHS}")
    print(f"   Batch size: {max_batch_size}")
    print(f"   Learning rate: {LR}")
    print(f"   Memory fraction: {MAX_GPU_MEMORY_FRACTION*100:.0f}%")
    print(f"   Device: {device}")
    
    # Training loop with maximum memory usage
    print(f"\n🔄 Starting maximum memory training...")
    print("=" * 80)
    
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Use maximum batch size
        idx = torch.randperm(len(tracks_norm))[:max_batch_size]
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
                    'COND_DIM': COND_DIM,
                    'MAX_BATCH_SIZE': max_batch_size
                }
            }, 'neutron_vae_max_memory_best.pth')
        
        # Progress reporting with memory monitoring
        if epoch % 100 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            eta = avg_time_per_epoch * (N_EPOCHS - epoch - 1)
            
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            memory_utilization = memory_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**2) * 100
            
            print(f"Epoch {epoch:4d}/{N_EPOCHS} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Best: {best_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"ETA: {eta/60:.1f}min")
            
            if gpu_available:
                print(f"   GPU Memory: {memory_allocated:.1f}MB / {memory_reserved:.1f}MB ({memory_utilization:.1f}%)")
        
        # Save checkpoint every 1000 epochs
        if epoch % 1000 == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'loss_history': loss_history,
                'max_batch_size': max_batch_size
            }, f'neutron_vae_max_memory_checkpoint_epoch_{epoch}.pth')
            print(f"💾 Checkpoint saved at epoch {epoch}")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n✅ Maximum memory training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Final loss: {loss_history[-1]:.6f}")
    
    # Memory usage summary
    if gpu_available:
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        utilization = peak_memory / total_memory * 100
        
        print(f"\n🚀 Maximum Memory Usage Summary:")
        print(f"   Peak memory used: {peak_memory:.1f} MB")
        print(f"   Total GPU memory: {total_memory:.1f} MB")
        print(f"   Memory utilization: {utilization:.1f}%")
        print(f"   Maximum batch size achieved: {max_batch_size}")
    
    # Save final model
    torch.save({
        'epoch': N_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_history[-1],
        'loss_history': loss_history,
        'max_batch_size': max_batch_size,
        'config': {
            'LATENT_DIM': LATENT_DIM,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'NUM_LAYERS': NUM_LAYERS,
            'COND_DIM': COND_DIM,
            'MAX_BATCH_SIZE': max_batch_size
        }
    }, 'neutron_vae_max_memory_final.pth')
    
    print(f"💾 Models saved:")
    print(f"   - neutron_vae_max_memory_best.pth")
    print(f"   - neutron_vae_max_memory_final.pth")
    
    return model, loss_history, max_batch_size

def main():
    """Main function"""
    try:
        print("🎯 MAXIMIZING CUDA MEMORY FOR NEUTRON VAE")
        print("=" * 60)
        
        model, loss_history, max_batch_size = train_with_maximum_memory()
        
        print(f"\n🎉 Maximum memory training completed successfully!")
        print(f"✅ Maximum batch size achieved: {max_batch_size}")
        
        # Plot training progress if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            # Plot loss
            plt.subplot(2, 1, 1)
            plt.plot(loss_history)
            plt.title('Training Loss Over Time (Maximum Memory)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Plot memory usage if available
            if torch.cuda.is_available():
                plt.subplot(2, 1, 2)
                # This would need to be tracked during training
                plt.title('GPU Memory Usage')
                plt.xlabel('Epoch')
                plt.ylabel('Memory (MB)')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('max_memory_training.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Training plot saved as 'max_memory_training.png'")
        except ImportError:
            print("📊 Matplotlib not available, skipping plot generation")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
