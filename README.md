# LSTM-MC: CUDA-Accelerated LSTM Trajectory GAN

A high-performance, CUDA-enabled implementation of LSTM-based Trajectory GAN for neutron trajectory generation, optimized for GPU clusters.

## Features

- **CUDA Acceleration**: Full GPU support with mixed precision training
- **LSTM Architecture**: Advanced recurrent neural networks for trajectory modeling
- **GAN Training**: Generative Adversarial Network for synthetic trajectory generation
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **Automatic Checkpointing**: Regular model saves during training

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.8+
- TensorFlow 2.x with CUDA support
- CUDA Toolkit 11.0+

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LSTM-MC.git
cd LSTM-MC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install "tensorflow[and-cuda]" numpy pandas scikit-learn scipy matplotlib seaborn
```

## Quick Start

### 1. Data Preparation
```bash
# Preprocess your neutron trajectory data
python neutron_data_preprocessor.py --input_csv data/Sheet.csv --max_length 200
```

### 2. Training
```bash
# Train with CUDA acceleration
python cuda_neutron_train.py <epochs> <batch_size> <save_interval> [mixed_precision]

# Example: 1000 epochs, batch size 32, save every 50 epochs, with mixed precision
python cuda_neutron_train.py 1000 32 50 True
```

### 3. Checkpoints
Training automatically saves checkpoints to `training_params/`:
- `cuda_neutron_C_model_{epoch}.h5` - Combined model
- `cuda_neutron_G_model_{epoch}.h5` - Generator
- `cuda_neutron_D_model_{epoch}.h5` - Discriminator

## Architecture

- **Generator**: LSTM-based trajectory generator with noise injection
- **Discriminator**: LSTM-based discriminator for trajectory authenticity
- **Training**: Adversarial training with gradient scaling and mixed precision

## Data Format

Input: CSV with columns `x`, `y`, `z`, `arc_length`
Output: Normalized trajectory segments in NPZ format

## Performance

- **GPU Memory**: Optimized with memory growth and mixed precision
- **Multi-GPU**: Automatic distribution strategy detection
- **Checkpointing**: Regular saves prevent training loss

## Cluster Usage

### SLURM Example
```bash
#!/bin/bash
#SBATCH -J lstm-mc
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00

module load cuda/12.2
python -m venv venv
source venv/bin/activate
pip install "tensorflow[and-cuda]" numpy pandas scikit-learn scipy

python neutron_data_preprocessor.py --input_csv data/Sheet.csv --max_length 200
python cuda_neutron_train.py 1000 32 50 True
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lstm_mc,
  title={LSTM-MC: CUDA-Accelerated LSTM Trajectory GAN},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/LSTM-MC}
}
```
