# 🚀 GPU-Accelerated Neutron VAE

This guide explains how to use GPU acceleration for the Neutron VAE project, specifically designed for Binder environments with GPU support.

## 🎯 Overview

The project now includes GPU-optimized scripts that can significantly speed up training and generation when CUDA is available. The scripts automatically detect GPU availability and fall back to CPU if needed.

## 📋 Prerequisites

### For Binder Environments:
1. **Select GPU Runtime**: When launching your Binder environment, make sure to select a GPU runtime
2. **CUDA Support**: The environment should have PyTorch with CUDA support installed
3. **NVIDIA GPU**: The underlying infrastructure should have NVIDIA GPUs available

### For Local Development:
1. **NVIDIA GPU**: Your machine needs an NVIDIA GPU with CUDA support
2. **CUDA Toolkit**: Install CUDA toolkit (version 11.0 or higher recommended)
3. **PyTorch with CUDA**: Install PyTorch with CUDA support

## 🔧 Setup

### 1. Check GPU Availability
```bash
python setup_cuda.py
```

This script will:
- Check if CUDA is available
- Display GPU information (name, memory, CUDA version)
- Run performance benchmarks
- Create optimized training scripts

### 2. Verify Installation
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

## 🚀 GPU-Optimized Scripts

### 1. Training with GPU (`train_gpu.py`)

**Features:**
- Automatic GPU detection and optimization
- Dynamic batch size optimization based on GPU memory
- Real-time GPU memory monitoring
- Automatic checkpoint saving
- Performance metrics tracking

**Usage:**
```bash
python train_gpu.py
```

**Key Optimizations:**
- **cuDNN Benchmarking**: Enabled for faster convolutions
- **Memory Management**: Automatic memory cleanup
- **Batch Size Optimization**: Tests different batch sizes to find optimal one
- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: Cosine annealing with warm restarts

### 2. Generation with GPU (`generate_gpu.py`)

**Features:**
- GPU-accelerated track generation
- Temperature-controlled generation
- Real-time performance monitoring
- Multiple output formats (NPY, CSV)

**Usage:**
```bash
python generate_gpu.py
```

**Parameters:**
- `n_tracks`: Number of tracks to generate (default: 20)
- `seq_len`: Sequence length per track (default: 200)
- `temperature`: Generation temperature (default: 1.0)

## 📊 Performance Benefits

### Training Speedup
- **GPU vs CPU**: 10-50x faster training depending on GPU
- **Memory Efficiency**: Optimized batch sizes for your GPU
- **Real-time Monitoring**: Track GPU memory and performance

### Generation Speedup
- **Batch Generation**: Generate multiple tracks simultaneously
- **Memory Optimization**: Efficient tensor operations
- **Temperature Control**: Adjust generation creativity

## 🔍 Monitoring GPU Usage

### During Training
The training script automatically displays:
- GPU memory usage
- Training progress
- Estimated time to completion
- Performance metrics

### Manual Monitoring
```bash
# Check GPU usage (if nvidia-smi is available)
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi
```

## ⚙️ Configuration

### GPU-Specific Settings
The configuration automatically adjusts for GPU:

```python
# In neutron_vae/config.py
BATCH_SIZE = 32 if torch.cuda.is_available() else 10  # Larger batch for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Memory Optimization
- **Automatic Batch Size**: Scripts test different batch sizes to find optimal one
- **Memory Cleanup**: Automatic cache clearing
- **Gradient Accumulation**: For very large models

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Symptoms:** `RuntimeError: CUDA out of memory`
**Solutions:**
- Reduce batch size in config
- Use gradient accumulation
- Clear GPU cache: `torch.cuda.empty_cache()`

#### 2. CUDA Not Available
**Symptoms:** Script falls back to CPU
**Solutions:**
- Check if GPU runtime is selected in Binder
- Verify PyTorch CUDA installation
- Check NVIDIA drivers

#### 3. Slow Performance
**Solutions:**
- Enable cuDNN benchmarking
- Increase batch size if memory allows
- Use mixed precision training (if available)

### Debug Commands
```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

# Check GPU memory
if torch.cuda.is_available():
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
```

## 📈 Expected Performance

### Training Times (Approximate)
| Hardware | Epochs | Time |
|----------|--------|------|
| CPU (Intel i7) | 1000 | ~30 minutes |
| GPU (RTX 3080) | 1000 | ~2 minutes |
| GPU (V100) | 1000 | ~1 minute |

### Generation Times
| Hardware | 20 tracks | Time |
|----------|-----------|------|
| CPU | 200 points each | ~10 seconds |
| GPU | 200 points each | ~0.5 seconds |

## 🎯 Best Practices

### 1. Memory Management
- Monitor GPU memory usage
- Use appropriate batch sizes
- Clear cache when needed

### 2. Training Optimization
- Start with smaller batch sizes
- Gradually increase if memory allows
- Use learning rate scheduling

### 3. Generation Optimization
- Use batch generation for multiple tracks
- Adjust temperature for creativity vs consistency
- Monitor generation quality

## 📁 File Structure

```
neutron-vae/
├── train_gpu.py          # GPU-optimized training
├── generate_gpu.py       # GPU-optimized generation
├── setup_cuda.py         # CUDA setup and testing
├── neutron_vae/
│   ├── config.py         # GPU-aware configuration
│   ├── model.py          # GPU-compatible model
│   └── train.py          # Original training script
└── GPU_README.md         # This file
```

## 🚀 Quick Start

1. **Setup GPU environment:**
   ```bash
   python setup_cuda.py
   ```

2. **Train with GPU:**
   ```bash
   python train_gpu.py
   ```

3. **Generate with GPU:**
   ```bash
   python generate_gpu.py
   ```

4. **Check accuracy:**
   ```bash
   python robust_accuracy_check.py
   ```

## 🎉 Success Indicators

- GPU memory usage is stable
- Training loss decreases consistently
- Generation completes quickly
- Generated tracks show good quality
- Accuracy scores are high (>60%)

## 📞 Support

If you encounter issues:
1. Check GPU availability with `setup_cuda.py`
2. Verify PyTorch CUDA installation
3. Monitor GPU memory usage
4. Try reducing batch size if out of memory
5. Check Binder GPU runtime selection

---

**Happy GPU-accelerated training! 🚀**
