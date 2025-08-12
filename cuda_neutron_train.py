import sys
import pandas as pd
import numpy as np
import os

from cuda_neutron_model import CUDA_Neutron_LSTM_TrajGAN

def main():
    if len(sys.argv) < 4:
        print("Usage: python cuda_neutron_train.py <epochs> <batch_size> <sample_interval> [mixed_precision]")
        print("Example: python cuda_neutron_train.py 1000 32 50 True")
        return
    
    n_epochs = int(sys.argv[1])
    n_batch_size = int(sys.argv[2])
    n_sample_interval = int(sys.argv[3])
    use_mixed_precision = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True
    
    # Neutron trajectory parameters
    latent_dim = 100
    max_length = 200
    coordinate_bounds = {'x': [-10, 10], 'y': [-10, 10], 'z': [-10, 10]}  # Adjust based on your data
    
    print("=" * 60)
    print("CUDA-Accelerated Neutron Trajectory GAN Training")
    print("=" * 60)
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {n_batch_size}")
    print(f"Sample interval: {n_sample_interval}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"Latent dim: {latent_dim}")
    print(f"Max length: {max_length}")
    
    # Check if training data exists
    if not os.path.exists('data/neutron_train_final.npz'):
        print("ERROR: Training data 'data/neutron_train_final.npz' not found!")
        print("Please ensure you have preprocessed neutron data available.")
        return
    
    # Initialize CUDA-enabled GAN
    try:
        gan = CUDA_Neutron_LSTM_TrajGAN(
            latent_dim=latent_dim,
            max_length=max_length,
            coordinate_bounds=coordinate_bounds,
            use_mixed_precision=use_mixed_precision
        )
        
        # Display device information
        print("\n" + "=" * 60)
        print("Device Information:")
        print("=" * 60)
        gan.get_device_info()
        
        # Start training
        print("\n" + "=" * 60)
        print("Starting Training:")
        print("=" * 60)
        
        gan.train(
            epochs=n_epochs,
            batch_size=n_batch_size,
            sample_interval=n_sample_interval
        )
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
        # Save final model
        gan.save_checkpoint('final')
        print("Final model saved.")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
