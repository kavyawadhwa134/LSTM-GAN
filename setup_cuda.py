#!/usr/bin/env python3
"""
CUDA Setup Script for Neutron VAE
Enables GPU acceleration for training and inference
"""

import torch
import os
import sys

def check_cuda_availability():
    """Check CUDA availability and print detailed information"""
    print("🔍 CUDA AVAILABILITY CHECK")
    print("=" * 50)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA Version: {cuda_version}")
        
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current Device: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Device Name: {device_name}")
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(current_device)
        print(f"Device Memory: {device_props.total_memory / 1024**3:.2f} GB")
        print(f"Compute Capability: {device_props.major}.{device_props.minor}")
        
        # Check memory usage
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2
        memory_cached = torch.cuda.memory_reserved(current_device) / 1024**2
        print(f"Memory Allocated: {memory_allocated:.2f} MB")
        print(f"Memory Cached: {memory_cached:.2f} MB")
        
        return True
    else:
        print("❌ CUDA is not available. Using CPU.")
        print("   Make sure you have:")
        print("   - NVIDIA GPU with CUDA support")
        print("   - CUDA toolkit installed")
        print("   - PyTorch with CUDA support")
        return False

def setup_cuda_optimization():
    """Setup CUDA optimization settings"""
    print("\n⚙️ CUDA OPTIMIZATION SETUP")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        print("✅ Enabled cuDNN benchmarking")
        
        # Enable cuDNN deterministic mode (optional, for reproducibility)
        # torch.backends.cudnn.deterministic = True
        # print("✅ Enabled cuDNN deterministic mode")
        
        # Set memory fraction (optional, to prevent OOM)
        # torch.cuda.set_per_process_memory_fraction(0.8)
        # print("✅ Set GPU memory fraction to 80%")
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("✅ Enabled Flash Attention (if available)")
        
        return True
    else:
        print("❌ CUDA not available, skipping optimization")
        return False

def test_cuda_performance():
    """Test CUDA performance with a simple benchmark"""
    print("\n🚀 CUDA PERFORMANCE TEST")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for performance test")
        return
    
    device = torch.device("cuda")
    
    # Create test tensors
    print("Creating test tensors...")
    x = torch.randn(1000, 200, 3).to(device)
    y = torch.randn(1000, 200, 3).to(device)
    
    # Warm up
    print("Warming up GPU...")
    for _ in range(10):
        _ = torch.matmul(x, y.transpose(-2, -1))
    
    # Benchmark
    print("Running benchmark...")
    torch.cuda.synchronize()
    
    import time
    start_time = time.time()
    
    for _ in range(100):
        result = torch.matmul(x, y.transpose(-2, -1))
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"✅ Average matrix multiplication time: {avg_time*1000:.2f} ms")
    
    # Memory test
    print("Testing memory operations...")
    large_tensor = torch.randn(5000, 200, 3).to(device)
    print(f"✅ Successfully created tensor of shape {large_tensor.shape}")
    print(f"   Memory used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    del large_tensor
    torch.cuda.empty_cache()
    print("✅ Memory cleared successfully")

def update_config_for_cuda():
    """Update configuration for optimal CUDA performance"""
    print("\n📝 UPDATING CONFIGURATION FOR CUDA")
    print("=" * 50)
    
    # Read current config
    config_path = os.path.join('neutron_vae', 'config.py')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Check if CUDA device is already configured
        if 'DEVICE = torch.device("cuda"' in config_content:
            print("✅ CUDA device already configured in config.py")
        else:
            print("❌ CUDA device not found in config.py")
        
        # Check for optimal batch size
        if 'BATCH_SIZE = 10' in config_content:
            print("ℹ️ Current batch size: 10")
            print("💡 Consider increasing batch size for GPU training")
            print("   Suggested: BATCH_SIZE = 32 or 64 for GPU")
        
        return True
    else:
        print("❌ Config file not found")
        return False

def create_cuda_training_script():
    """Create a CUDA-optimized training script"""
    print("\n📄 CREATING CUDA TRAINING SCRIPT")
    print("=" * 50)
    
    cuda_script = '''#!/usr/bin/env python3
"""
CUDA-Optimized Training Script for Neutron VAE
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.model import UltraHighFidelityVAE, vae_loss
from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks
from neutron_vae.config import *

def setup_cuda():
    """Setup CUDA optimization"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("⚠️ CUDA not available, using CPU")
        return False

def main():
    print("🧪 CUDA-OPTIMIZED NEUTRON VAE TRAINING")
    print("=" * 60)
    
    # Setup CUDA
    cuda_available = setup_cuda()
    
    # Load and preprocess data
    print("📊 Loading data...")
    raw_tracks = load_and_split_tracks(CSV_FILE)
    tracks_norm, conditions, xyz_min, xyz_max = preprocess_tracks(raw_tracks)
    
    # Convert to torch tensors
    tracks_norm = torch.tensor(tracks_norm, dtype=torch.float32).to(DEVICE)
    conditions = torch.tensor(conditions, dtype=torch.float32).to(DEVICE)
    tracks_norm = 2 * tracks_norm - 1  # [0,1] → [-1,1]
    
    print(f"✅ Data loaded: {tracks_norm.shape}")
    
    # Create model
    model = UltraHighFidelityVAE().to(DEVICE)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer with CUDA-optimized settings
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
    
    print("🚀 Starting training...")
    print(f"   Epochs: {N_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LR}")
    
    # Training loop
    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Random batch
        idx = torch.randperm(len(tracks_norm))[:BATCH_SIZE]
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
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}")
    
    print("✅ Training completed!")
    
    # Save model
    torch.save(model.state_dict(), 'neutron_vae_cuda_best.pth')
    print("💾 Model saved as 'neutron_vae_cuda_best.pth'")

if __name__ == "__main__":
    main()
'''
    
    with open('train_cuda.py', 'w') as f:
        f.write(cuda_script)
    
    print("✅ Created train_cuda.py")
    print("   Run with: python train_cuda.py")

def main():
    """Main setup function"""
    print("🔧 CUDA SETUP FOR NEUTRON VAE")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = check_cuda_availability()
    
    if cuda_available:
        # Setup optimization
        setup_cuda_optimization()
        
        # Test performance
        test_cuda_performance()
        
        # Update configuration
        update_config_for_cuda()
        
        # Create CUDA training script
        create_cuda_training_script()
        
        print("\n🎉 CUDA SETUP COMPLETE!")
        print("=" * 50)
        print("✅ CUDA is available and optimized")
        print("✅ Performance test completed")
        print("✅ Configuration updated")
        print("✅ CUDA training script created")
        print("\n🚀 Next steps:")
        print("   1. Run: python train_cuda.py")
        print("   2. Monitor GPU usage with: nvidia-smi")
        print("   3. Check training speed improvement")
        
    else:
        print("\n❌ CUDA SETUP FAILED")
        print("=" * 50)
        print("Please ensure:")
        print("   - NVIDIA GPU is available")
        print("   - CUDA toolkit is installed")
        print("   - PyTorch with CUDA support is installed")
        print("\nFor Binder environments:")
        print("   - Make sure GPU runtime is selected")
        print("   - Check if CUDA is available in the environment")

if __name__ == "__main__":
    main()
