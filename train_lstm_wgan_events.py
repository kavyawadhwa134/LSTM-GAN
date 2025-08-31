import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from data_preprocessing import NeutronDataPreprocessor, create_data_loaders
from lstm_wgan_with_events import LSTMWGANWithEvents

def train_lstm_wgan_with_events(csv_file, sequence_length=50, hidden_dim=256, num_layers=3, 
                              batch_size=64, iterations=4000, save_interval=500, 
                              checkpoint_dir='lstm_wgan_events_checkpoints'):
    """
    Train the LSTM-WGAN with event prediction for 4000 iterations
    """
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for CUDA availability and optimize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("CUDA optimizations enabled: cudnn.benchmark=True")
    else:
        print("CUDA not available, using CPU")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = NeutronDataPreprocessor(csv_file, sequence_length)
    sequences = preprocessor.load_and_preprocess()
    
    # Get data statistics
    stats = preprocessor.get_data_statistics()
    print(f"Data statistics: {stats}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(sequences, batch_size=batch_size)
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Initialize LSTM-WGAN with events
    input_dim = sequences.shape[2]  # Number of features
    wgan = LSTMWGANWithEvents(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sequence_length=sequence_length,
        device=device
    )
    
    # Print model info
    model_info = wgan.get_model_info()
    print(f"LSTM-WGAN with events model info: {model_info}")
    print(f"Learning rate: 0.00005 (RMSprop)")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {iterations}")
    print(f"Event prediction: {wgan.event_names}")
    print(f"WGAN parameters: lambda_gp=10.0, n_critic=5")
    
    # Training loop
    print("Starting LSTM-WGAN with events training...")
    start_time = time.time()
    
    iteration = 0
    best_accuracy = 0.0
    
    while iteration < iterations:
        for batch_idx, real_sequences in enumerate(train_loader):
            if iteration >= iterations:
                break
                
            real_sequences = real_sequences.to(device)
            
            # Train step
            g_loss, d_loss, gradient_penalty = wgan.train_step(real_sequences)
            
            wgan.g_losses.append(g_loss)
            wgan.d_losses.append(d_loss)
            wgan.gradient_penalties.append(gradient_penalty)
            
            iteration += 1
            
            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{iterations}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, GP: {gradient_penalty:.4f}")
            
            # Save checkpoint and evaluate
            if iteration % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'lstm_wgan_events_checkpoint_iter_{iteration}.pth')
                wgan.save_checkpoint(checkpoint_path)
                
                # Generate sample sequences with events
                sample_sequences, sample_events = wgan.generate_sequences_with_events(100)
                print(f"Generated sample sequences shape: {sample_sequences.shape}")
                print(f"Generated sample events shape: {sample_events.shape}")
                
                # Quick accuracy check
                if len(sample_sequences) > 0:
                    energy_values = sample_sequences[:, :, 6].flatten()
                    positive_energy = np.sum(energy_values > 0)
                    print(f"Sample energy check: {positive_energy}/{len(energy_values)} positive ({positive_energy/len(energy_values)*100:.1f}%)")
                    
                    # Event analysis
                    print("Event prediction analysis:")
                    for i, event_name in enumerate(wgan.event_names):
                        event_probs = sample_events[:, :, i].flatten()
                        avg_prob = np.mean(event_probs)
                        print(f"  {event_name}: {avg_prob:.4f} average probability")
                    
                    # Quick accuracy assessment
                    accuracy = quick_accuracy_check(sequences[:100], sample_sequences)
                    print(f"Quick accuracy assessment: {accuracy:.1f}%")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        # Save best model
                        best_checkpoint_path = os.path.join(checkpoint_dir, 'lstm_wgan_events_best_checkpoint.pth')
                        wgan.save_checkpoint(best_checkpoint_path)
                        print(f"New best accuracy: {accuracy:.1f}% - Model saved!")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'lstm_wgan_events_final_checkpoint.pth')
    wgan.save_checkpoint(final_checkpoint_path)
    
    training_time = time.time() - start_time
    print(f"LSTM-WGAN with events training completed in {training_time:.2f} seconds")
    print(f"Best accuracy achieved: {best_accuracy:.1f}%")
    
    return wgan, preprocessor

def quick_accuracy_check(original_sequences, generated_sequences):
    """Quick accuracy assessment for monitoring during training"""
    try:
        # Flatten sequences
        orig_flat = original_sequences.reshape(-1, 7)
        gen_flat = generated_sequences.reshape(-1, 7)
        
        # Calculate feature-wise accuracy
        feature_scores = []
        for i in range(7):
            orig_feature = orig_flat[:, i]
            gen_feature = gen_flat[:, i]
            
            # Statistical comparison
            orig_mean = np.mean(orig_feature)
            gen_mean = np.mean(gen_feature)
            orig_std = np.std(orig_feature)
            gen_std = np.std(gen_feature)
            
            mean_diff = abs(orig_mean - gen_mean) / abs(orig_mean) * 100 if orig_mean != 0 else 0
            std_diff = abs(orig_std - gen_std) / orig_std * 100 if orig_std != 0 else 0
            
            # Feature score (0-100)
            mean_score = max(0, 100 - mean_diff)
            std_score = max(0, 100 - std_diff)
            feature_score = (mean_score + std_score) / 2
            feature_scores.append(feature_score)
        
        return np.mean(feature_scores)
    except:
        return 0.0

def generate_and_save_lstm_wgan_sequences_with_events(wgan, preprocessor, num_sequences=1000, 
                                                    save_path='lstm_wgan_events_generated_sequences.npy'):
    """Generate and save synthetic sequences with event predictions in both NPY and CSV formats"""
    print(f"Generating {num_sequences} LSTM-WGAN synthetic sequences with events...")
    
    # Generate sequences with events
    synthetic_sequences, synthetic_events = wgan.generate_sequences_with_events(num_sequences)
    
    # Convert back to original scale
    original_scale_sequences = []
    for sequence in synthetic_sequences:
        original_scale = preprocessor.inverse_transform(sequence)
        original_scale_sequences.append(original_scale)
    
    original_scale_sequences = np.array(original_scale_sequences)
    
    # Save sequences to NPY file
    np.save(save_path, original_scale_sequences)
    print(f"LSTM-WGAN generated sequences saved to {save_path}")
    print(f"Generated sequences shape: {original_scale_sequences.shape}")
    
    # Save events to NPY file
    events_path = save_path.replace('.npy', '_events.npy')
    np.save(events_path, synthetic_events)
    print(f"LSTM-WGAN generated events saved to {events_path}")
    print(f"Generated events shape: {synthetic_events.shape}")
    
    # Convert to CSV format
    csv_path = save_path.replace('.npy', '_data.csv')
    convert_sequences_with_events_to_csv(original_scale_sequences, synthetic_events, csv_path, wgan.event_names)
    
    return original_scale_sequences, synthetic_events

def convert_sequences_with_events_to_csv(sequences, events, csv_path, event_names):
    """Convert sequences and events to CSV format"""
    
    # Reshape to 2D for CSV (flatten sequences)
    num_sequences, seq_length, features = sequences.shape
    flattened_data = sequences.reshape(-1, features)
    flattened_events = events.reshape(-1, len(event_names))
    
    # Create column names for sequences
    sequence_columns = ['x_position', 'y_position', 'z_position', 'vx_velocity', 'vy_velocity', 'vz_velocity', 'energy']
    
    # Create column names for events
    event_columns = [f'event_{event_name}' for event_name in event_names]
    
    # Combine all columns
    all_columns = sequence_columns + event_columns
    
    # Create DataFrame
    combined_data = np.concatenate([flattened_data, flattened_events], axis=1)
    df = pd.DataFrame(combined_data, columns=all_columns)
    
    # Add sequence ID column
    sequence_ids = np.repeat(np.arange(num_sequences), seq_length)
    df.insert(0, 'sequence_id', sequence_ids)
    
    # Add time step column
    time_steps = np.tile(np.arange(seq_length), num_sequences)
    df.insert(1, 'time_step', time_steps)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Generated data with events saved to CSV: {csv_path}")
    print(f"CSV DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show sample of the data
    print('\nSample of generated data with events:')
    print(df.head(10))
    
    return df

def plot_training_progress(wgan, save_path='lstm_wgan_events_training_progress.png'):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator and Discriminator losses
    axes[0,0].plot(wgan.g_losses, label='Generator Loss', alpha=0.7)
    axes[0,0].plot(wgan.d_losses, label='Discriminator Loss', alpha=0.7)
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Generator vs Discriminator Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Gradient penalty
    axes[0,1].plot(wgan.gradient_penalties, label='Gradient Penalty', color='red', alpha=0.7)
    axes[0,1].set_xlabel('Iteration')
    axes[0,1].set_ylabel('Gradient Penalty')
    axes[0,1].set_title('WGAN Gradient Penalty')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Loss ratio
    if len(wgan.g_losses) > 0 and len(wgan.d_losses) > 0:
        loss_ratio = [g/d if d > 0 else 0 for g, d in zip(wgan.g_losses, wgan.d_losses)]
        axes[1,0].plot(loss_ratio, label='G/D Loss Ratio', color='green', alpha=0.7)
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Loss Ratio')
        axes[1,0].set_title('Generator/Discriminator Loss Ratio')
        axes[1,0].legend()
        axes[1,0].grid(True)
    
    # Combined loss
    if len(wgan.g_losses) > 0 and len(wgan.gradient_penalties) > 0:
        combined_loss = [g + gp for g, gp in zip(wgan.g_losses, wgan.gradient_penalties)]
        axes[1,1].plot(combined_loss, label='Combined Loss', color='purple', alpha=0.7)
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Combined Loss')
        axes[1,1].set_title('Generator + Gradient Penalty Loss')
        axes[1,1].legend()
        axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training progress plot saved to {save_path}")

if __name__ == "__main__":
    # LSTM-WGAN with events training parameters
    CSV_FILE = 'Sheet.csv'
    SEQUENCE_LENGTH = 50
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    BATCH_SIZE = 64
    ITERATIONS = 4000
    SAVE_INTERVAL = 500
    
    print("LSTM-WGAN WITH EVENT PREDICTION TRAINING")
    print("="*60)
    print(f"Hidden dimensions: {HIDDEN_DIM}")
    print(f"Number of layers: {NUM_LAYERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Event prediction: scattering, absorption, fission, leakage, capture")
    print(f"WGAN with gradient penalty and attention mechanism")
    print("="*60)
    
    # Train the LSTM-WGAN with events
    wgan, preprocessor = train_lstm_wgan_with_events(
        csv_file=CSV_FILE,
        sequence_length=SEQUENCE_LENGTH,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE,
        iterations=ITERATIONS,
        save_interval=SAVE_INTERVAL
    )
    
    # Plot training progress
    plot_training_progress(wgan)
    
    # Generate synthetic sequences with events
    synthetic_sequences, synthetic_events = generate_and_save_lstm_wgan_sequences_with_events(wgan, preprocessor)
    
    print("LSTM-WGAN with events training completed successfully!")
