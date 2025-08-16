# Neutron VAE - High-Fidelity Neutron Track Generation

A Variational Autoencoder (VAE) implementation for generating high-fidelity neutron track data using PyTorch and CUDA acceleration.

## 🎯 Project Overview

This project implements an Ultra-High Fidelity VAE model that can generate realistic 3D neutron track data. The model achieves **74.84% accuracy** compared to real neutron track data and can train efficiently on GPU hardware.

### Key Features

- **🚀 GPU Accelerated**: Full CUDA support for fast training and generation
- **🎯 High Accuracy**: 74.84% overall accuracy score
- **⚡ Fast Training**: Completes training in ~8.5 minutes on GPU
- **🔧 Checkpoint Support**: Resume training from any point
- **📊 Comprehensive Analysis**: Built-in accuracy assessment tools
- **💾 Multiple Formats**: Generate data in both NPY and CSV formats

## 📁 Project Structure

```
neutron-vae/
├── neutron_vae/                 # Core module
│   ├── __init__.py
│   ├── config.py               # Configuration parameters
│   ├── model.py                # VAE model architecture
│   ├── train.py                # Training script with checkpoint support
│   ├── generate.py             # Basic generation script
│   ├── load_data.py            # Data loading utilities
│   ├── visualize.py            # Visualization tools
│   └── Sheet.csv               # Real neutron track data
├── quick_training.py           # Fast GPU training script
├── generate_simple_quick.py    # Generation script for quick model
├── final_accuracy_check_quick.py # Accuracy assessment tool
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train with GPU acceleration (recommended)
python quick_training.py
```

**Expected Output:**
```
🚀 QUICK TRAINING WITH GPU ACCELERATION
============================================================
✅ CUDA available: True
✅ GPU: NVIDIA GeForce RTX 4090
✅ Memory: 24.0 GB
✅ Quick batch size achieved: 2048
⚡ Fast training optimization complete!
📊 Training plot saved as 'quick_training.png'
```

### 3. Generate New Tracks

```bash
# Generate 20 new neutron tracks
python generate_simple_quick.py
```

**Expected Output:**
```
🎲 SIMPLE GENERATION FROM QUICK TRAINING MODEL
============================================================
✅ Model loaded successfully
   Parameters: 21,452,707
   Training loss: 0.047868
✅ Generated tracks shape: (20, 200, 3)
✅ Saved as numpy array: quick_generated_neutron_tracks.npy
✅ Saved as CSV: quick_generated_neutron_tracks.csv
```

### 4. Check Accuracy

```bash
# Assess accuracy against real data
python final_accuracy_check_quick.py
```

**Expected Output:**
```
🧪 FINAL ACCURACY CHECK - QUICK TRAINING
============================================================
🏆 OVERALL ACCURACY SCORE: 74.84%
🎯 Quality Assessment: GOOD
```

## 🏗️ Model Architecture

### Ultra-High Fidelity VAE

The model uses a sophisticated architecture designed for high-fidelity generation:

- **Encoder**: Deep Bidirectional LSTM + Transformer Blocks
- **Decoder**: Multi-layer LSTM with Residual Connections
- **Latent Space**: 64-dimensional continuous space
- **Conditioning**: 6-dimensional condition vector
- **Output**: 3D track coordinates (x, y, z)

### Key Components

1. **TransformerBlock**: Multi-head attention for sequence modeling
2. **ResidualBlock**: Skip connections for better gradient flow
3. **Bidirectional LSTM**: Captures both forward and backward dependencies
4. **Conditional Generation**: Uses track conditions for realistic generation

## 📊 Performance Metrics

### Training Performance
- **Training Time**: ~8.5 minutes on GPU
- **Batch Size**: 2048 (GPU optimized)
- **Final Loss**: 0.047868
- **Memory Usage**: ~881 MB GPU memory

### Generation Quality
- **Overall Accuracy**: 74.84%
- **Spatial Coverage**: 100.00%
- **Distribution Similarity**: 79.79%
- **Curvature Similarity**: 89.72%

## 🔧 Configuration

Key parameters in `neutron_vae/config.py`:

```python
LATENT_DIM = 64          # Latent space dimension
HIDDEN_SIZE = 256        # Hidden layer size
NUM_LAYERS = 4           # Number of LSTM layers
BATCH_SIZE = 64          # Batch size (GPU optimized)
N_EPOCHS = 5000          # Training epochs
LR = 1e-4               # Learning rate
COND_DIM = 6            # Condition vector dimension
```

## 📈 Generated Data Format

### NPY Format
```python
# Shape: (n_tracks, n_points, 3)
# Example: (20, 200, 3) for 20 tracks with 200 points each
tracks = np.load('quick_generated_neutron_tracks.npy')
```

### CSV Format
```csv
track_id,point_id,x,y,z
0,0,0.123,0.456,0.789
0,1,0.124,0.457,0.790
...
```

## 🎯 Accuracy Assessment

The accuracy check evaluates:

1. **Statistics Similarity** (52.27%)
   - Track lengths, curvatures, smoothness
   - Spatial range, point counts

2. **Spatial Coverage** (100.00%)
   - Coverage of 3D space
   - Bounds comparison

3. **Distribution Similarity** (79.79%)
   - Histogram comparison across dimensions
   - Statistical distribution matching

## 🔄 Checkpoint System

The training script automatically saves checkpoints every 500 epochs:

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
}, 'neutron_vae_quick_best.pth')
```

## 🚀 GPU Optimization

The model is optimized for GPU performance:

- **CUDA Acceleration**: Automatic GPU detection and usage
- **Memory Management**: Efficient memory allocation (95% utilization)
- **TF32 Precision**: Faster training with TensorFloat-32
- **Batch Optimization**: Automatic batch size optimization
- **Gradient Clipping**: Prevents gradient explosion

## 📊 Visualization

Generated tracks can be visualized using the built-in tools:

```python
from neutron_vae.visualize import plot_tracks
plot_tracks(generated_tracks, save_path='tracks.png')
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `config.py`
   - Use CPU training if GPU memory is limited

2. **Model Loading Errors**
   - Ensure you're using the correct model file
   - Check PyTorch version compatibility

3. **Accuracy Issues**
   - Increase training epochs
   - Adjust learning rate
   - Check data normalization

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- NumPy
- Matplotlib
- Pandas

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**🎉 Ready to generate high-fidelity neutron tracks!**


