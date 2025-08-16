#!/usr/bin/env python3
"""
Ultra-Fast Training Script for Neutron VAE
Optimized for maximum speed on powerful GPUs like NVIDIA A10
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

def setup_ultra_fast_gpu():
    """Setup GPU for ultra-fast training"""
    print("🚀 ULTRA-FAST GPU SETUP")
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
    
    # Ultra-fast optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True  # Enable TensorFloat-32 for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Set maximum memory fraction
    torch.cuda.set_per_process_memory_fraction(0.98)  # Use 98% of memory
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✅ Ultra-fast optimizations enabled")
    print(f"   cuDNN Benchmark: Enabled")
    print(f"   TF32: Enabled")
    print(f"   Memory Fraction: 98%")
    
    return device, True

def find_ultra_fast_batch_size(device, tracks_norm, conditions):
    """Find the fastest batch size for maximum throughput"""
    print("\n🔍 FINDING ULTRA-FAST BATCH SIZE")
    print("=" * 50)
    
    if device.type != "cuda":
        return BATCH_SIZE
    
    # Test very large batch sizes for maximum throughput
    batch_sizes = [512, 1024, 2048, 4096, 8192]
    optimal_batch_size = 512
    fastest_time = float('inf')
    
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
            
            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    _ = model(x_batch, c_batch)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):  # Multiple iterations for accurate timing
                recon, mu, logvar = model(x_batch, c_batch)
                loss, _ = vae_loss(recon, x_batch, mu, logvar, beta=BETA, alpha=ALPHA)
                loss.backward()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            throughput = batch_size / avg_time
            
            memory_used = torch.cuda.memory_allocated() / 1024**2
            
            print(f"   ✅ Success - Time: {avg_time:.4f}s, Throughput: {throughput:.0f} samples/s, Memory: {memory_used:.1f} MB")
            
            if throughput > fastest_time:
                optimal_batch_size = batch_size
                fastest_time = throughput
            
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
    
    print(f"🎯 Optimal batch size: {optimal_batch_size}")
    print(f"📊 Maximum throughput: {fastest_time:.0f} samples/s")
    
    return optimal_batch_size

def train_ultra_fast():
    """Ultra-fast training with maximum GPU utilization"""
    print("⚡ ULTRA-FAST NEUTRON VAE TRAINING")
    print("=" * 60)
    
    # Setup ultra-fast GPU
    device, gpu_available = setup_ultra_fast_gpu()
    
    # Load and preprocess data
    print("\n📊 Loading data...")
    raw_tracks = load_and_split_tracks(CSV_FILE)
    tracks_norm, conditions, xyz_min, xyz_max = preprocess_tracks(raw_tracks)
    
    # Convert to torch tensors
    tracks_norm = torch.tensor(tracks_norm, dtype=torch.float32)
    conditions = torch.tensor(conditions, dtype=torch.float32)
    tracks_norm = 2 * tracks_norm - 1  # [0,1] → [-1,1]
    
    print(f"✅ Data loaded: {tracks_norm.shape}")
    
    # Find ultra-fast batch size
    ultra_batch_size = find_ultra_fast_batch_size(device, tracks_norm, conditions)
    
    # Move data to device
    tracks_norm = tracks_norm.to(device)
    conditions = conditions.to(device)
    
    # Create model with mixed precision for speed
    model = UltraHighFidelityVAE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n🤖 Model created:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Device: {device}")
    print(f"   Ultra-fast batch size: {ultra_batch_size}")
    
    # Optimizer with ultra-fast settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR * 2,  # Higher learning rate for faster convergence
        weight_decay=WEIGHT_DECAY, 
        betas=(0.9, 0.999)
    )
    
    # Fast learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, T_mult=2, eta_min=LR/100  # Faster scheduling
    )
    
    # Training settings
    print(f"\n⚡ Ultra-Fast Training Configuration:")
    print(f"   Epochs: {N_EPOCHS}")
    print(f"   Batch size: {ultra_batch_size}")
    print(f"   Learning rate: {LR * 2}")
    print(f"   Memory fraction: 98%")
    print(f"   Device: {device}")
    
    # Training loop with ultra-fast optimization
    print(f"\n🔄 Starting ultra-fast training...")
    print("=" * 80)
    
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    # Early stopping
    patience = 500
    no_improvement = 0
    
    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Use ultra-fast batch size
        idx = torch.randperm(len(tracks_norm))[:ultra_batch_size]
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
                    'ULTRA_BATCH_SIZE': ultra_batch_size
                }
            }, 'neutron_vae_ultra_fast_best.pth')
        else:
            no_improvement += 1
        
        # Progress reporting with speed metrics
        if epoch % 50 == 0:  # More frequent updates
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            eta = avg_time_per_epoch * (N_EPOCHS - epoch - 1)
            throughput = ultra_batch_size / avg_time_per_epoch
            
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            memory_utilization = memory_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**2) * 100
            
            print(f"Epoch {epoch:4d}/{N_EPOCHS} | "
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
                'ultra_batch_size': ultra_batch_size
            }, f'neutron_vae_ultra_fast_checkpoint_epoch_{epoch}.pth')
            print(f"💾 Checkpoint saved at epoch {epoch}")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n✅ Ultra-fast training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Final loss: {loss_history[-1]:.6f}")
    print(f"   Average speed: {ultra_batch_size / (total_time/N_EPOCHS):.0f} samples/s")
    
    # Memory usage summary
    if gpu_available:
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        utilization = peak_memory / total_memory * 100
        
        print(f"\n🚀 Ultra-Fast Performance Summary:")
        print(f"   Peak memory used: {peak_memory:.1f} MB")
        print(f"   Total GPU memory: {total_memory:.1f} MB")
        print(f"   Memory utilization: {utilization:.1f}%")
        print(f"   Ultra-fast batch size: {ultra_batch_size}")
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_history[-1],
        'loss_history': loss_history,
        'ultra_batch_size': ultra_batch_size,
        'config': {
            'LATENT_DIM': LATENT_DIM,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'NUM_LAYERS': NUM_LAYERS,
            'COND_DIM': COND_DIM,
            'ULTRA_BATCH_SIZE': ultra_batch_size
        }
    }, 'neutron_vae_ultra_fast_final.pth')
    
    print(f"💾 Models saved:")
    print(f"   - neutron_vae_ultra_fast_best.pth")
    print(f"   - neutron_vae_ultra_fast_final.pth")
    
    return model, loss_history, ultra_batch_size

def main():
    """Main function"""
    try:
        print("⚡ ULTRA-FAST TRAINING FOR NEUTRON VAE")
        print("=" * 60)
        
        model, loss_history, ultra_batch_size = train_ultra_fast()
        
        print(f"\n🎉 Ultra-fast training completed successfully!")
        print(f"✅ Ultra-fast batch size achieved: {ultra_batch_size}")
        print(f"⚡ Maximum speed optimization complete!")
        
        # Plot training progress if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            # Plot loss
            plt.subplot(2, 1, 1)
            plt.plot(loss_history)
            plt.title('Ultra-Fast Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Plot speed
            plt.subplot(2, 1, 2)
            speeds = [ultra_batch_size / (time.time() / len(loss_history))] * len(loss_history)
            plt.plot(speeds)
            plt.title('Training Speed (samples/second)')
            plt.xlabel('Epoch')
            plt.ylabel('Speed')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('ultra_fast_training.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Training plot saved as 'ultra_fast_training.png'")
        except ImportError:
            print("📊 Matplotlib not available, skipping plot generation")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
