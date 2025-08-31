#!/usr/bin/env python3
"""
GEANT4-Style Data Generation with LSTM-WGAN
===========================================
Generate neutron physics data using the trained LSTM-WGAN model
with professional GEANT4-style terminal output.
"""

import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
from lstm_wgan_with_events import LSTMWGANWithEvents
from data_preprocessing import NeutronDataPreprocessor
from geant4_style_output import Geant4Output, Colors

def main():
    # Initialize GEANT4-style output
    g4_output = Geant4Output(width=100)
    
    # Clear screen and print header
    g4_output.clear_screen()
    g4_output.print_header("LSTM-WGAN Neutron Physics Simulation", "2.0.0")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g4_output.print_section("SYSTEM INITIALIZATION", 1)
    g4_output.print_info(f"Using device: {device}")
    g4_output.print_info(f"PyTorch version: {torch.__version__}")
    
    # Initialize model
    g4_output.print_section("MODEL LOADING", 1)
    g4_output.print_info("Initializing LSTM-WGAN model...")
    
    try:
        wgan = LSTMWGANWithEvents(device=device)
        g4_output.print_success("Model architecture initialized")
        
        # Load checkpoint
        checkpoint_path = 'lstm_wgan_events_checkpoints/lstm_wgan_events_best_checkpoint.pth'
        g4_output.print_info(f"Loading checkpoint: {checkpoint_path}")
        
        wgan.load_checkpoint(checkpoint_path)
        g4_output.print_success("Best LSTM-WGAN model loaded successfully!")
        
    except Exception as e:
        g4_output.print_error(f"Failed to load model: {str(e)}")
        return
    
    # Data generation parameters
    num_sequences = 1000
    g4_output.print_section("DATA GENERATION", 1)
    g4_output.print_info(f"Generating {num_sequences:,} neutron sequences")
    g4_output.print_info("Each sequence contains 50 time steps with 7 physics features")
    g4_output.print_info("Event prediction for: scattering, absorption, fission, leakage, capture")
    
    # Generate data with progress tracking
    g4_output.print_section("GENERATION PROGRESS", 1)
    start_time = time.time()
    
    print(f"\n{Colors.BRIGHT_CYAN}Generating sequences...{Colors.RESET}")
    
    # Generate sequences in batches for progress tracking
    batch_size = 100
    all_sequences = []
    all_events = []
    
    for batch_idx in range(0, num_sequences, batch_size):
        current_batch_size = min(batch_size, num_sequences - batch_idx)
        
        # Generate batch
        batch_sequences, batch_events = wgan.generate_sequences_with_events(current_batch_size)
        all_sequences.append(batch_sequences)
        all_events.append(batch_events)
        
        # Update progress
        progress = batch_idx + current_batch_size
        g4_output.print_progress_bar(progress, num_sequences, "Generation", 
                                   f"Batch {batch_idx//batch_size + 1}", 50)
        
        # Show real-time statistics
        if progress % 200 == 0:
            elapsed = time.time() - start_time
            rate = progress / elapsed if elapsed > 0 else 0
            print(f"\n{Colors.CYAN}Generated: {progress:,}/{num_sequences:,} sequences | "
                  f"Rate: {rate:.1f} seq/s | Elapsed: {elapsed:.1f}s{Colors.RESET}")
    
    # Combine all batches
    all_sequences = np.concatenate(all_sequences, axis=0)
    all_events = np.concatenate(all_events, axis=0)
    
    generation_time = time.time() - start_time
    g4_output.print_success(f"Data generation completed in {generation_time:.2f} seconds")
    
    # Analyze generated data
    g4_output.print_section("DATA ANALYSIS", 1)
    g4_output.print_info("Analyzing generated sequences...")
    
    # Event analysis
    total_time_steps = all_sequences.shape[0] * all_sequences.shape[1]
    event_names = wgan.event_names
    
    # Method 1: Dominant events
    dominant_events = np.argmax(all_events, axis=2)
    dominant_events_flat = dominant_events.flatten()
    
    event_counts = {}
    for i, event_name in enumerate(event_names):
        count = np.sum(dominant_events_flat == i)
        percentage = (count / total_time_steps) * 100
        event_counts[event_name] = {'count': count, 'percentage': percentage}
    
    # Print event summary
    g4_output.print_event_summary(event_counts, total_time_steps)
    
    # Physics analysis
    g4_output.print_section("PHYSICS VALIDATION", 1)
    
    # Energy analysis
    energy_values = all_sequences[:, :, 6].flatten()
    energy_stats = {
        'mean': np.mean(energy_values),
        'std': np.std(energy_values),
        'min': np.min(energy_values),
        'max': np.max(energy_values)
    }
    
    # Position analysis
    x_values = all_sequences[:, :, 0].flatten()
    y_values = all_sequences[:, :, 1].flatten()
    z_values = all_sequences[:, :, 2].flatten()
    
    position_stats = {
        'x_range': np.max(x_values) - np.min(x_values),
        'y_range': np.max(y_values) - np.min(y_values),
        'z_range': np.max(z_values) - np.min(z_values)
    }
    
    physics_data = {
        'energy_stats': energy_stats,
        'position_stats': position_stats
    }
    
    g4_output.print_physics_summary(physics_data)
    
    # Real-time event display (GEANT4 style)
    g4_output.print_section("REAL-TIME EVENT DISPLAY", 1)
    g4_output.print_info("Displaying neutron events in GEANT4 style...")
    
    print(f"\n{Colors.BRIGHT_WHITE}{'Event ID':<8} {'Time':<6} {'Position (x,y,z)':<20} {'Energy':<8} {'Event Type':<12} {'Probability':<12}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}{'─'*80}{Colors.RESET}")
    
    # Show first 50 events from first sequence
    sequence_idx = 0
    event_counter = 0
    
    for time_step in range(min(50, all_sequences.shape[1])):
        # Get physics data
        x, y, z = all_sequences[sequence_idx, time_step, 0:3]
        energy = all_sequences[sequence_idx, time_step, 6]
        
        # Get event data
        event_probs = all_events[sequence_idx, time_step, :]
        dominant_event_idx = np.argmax(event_probs)
        dominant_event = event_names[dominant_event_idx]
        dominant_prob = event_probs[dominant_event_idx]
        
        # Color coding for events
        event_colors = {
            'scattering': Colors.BLUE,
            'absorption': Colors.RED,
            'fission': Colors.YELLOW,
            'leakage': Colors.MAGENTA,
            'capture': Colors.GREEN
        }
        
        event_color = event_colors.get(dominant_event, Colors.WHITE)
        
        # Format output
        event_id = f"E{event_counter:04d}"
        position = f"({x:.3f},{y:.3f},{z:.3f})"
        energy_str = f"{energy:.4f}"
        event_type = f"{event_color}{dominant_event.upper()}{Colors.RESET}"
        probability = f"{dominant_prob:.3f}"
        
        print(f"{event_id:<8} {time_step:<6} {position:<20} {energy_str:<8} {event_type:<20} {probability:<12}")
        
        event_counter += 1
        
        # Add small delay for visual effect
        time.sleep(0.05)
    
    print(f"{Colors.BRIGHT_CYAN}{'─'*80}{Colors.RESET}")
    g4_output.print_success("Real-time event display completed")
    
    # Save data to CSV
    g4_output.print_section("DATA EXPORT", 1)
    g4_output.print_info("Converting generated data to CSV format...")
    
    # Create detailed CSV with events
    detailed_data = []
    for seq_idx in range(all_sequences.shape[0]):
        for time_idx in range(all_sequences.shape[1]):
            # Physics data
            x, y, z = all_sequences[seq_idx, time_idx, 0:3]
            vx, vy, vz = all_sequences[seq_idx, time_idx, 3:6]
            energy = all_sequences[seq_idx, time_idx, 6]
            
            # Event data
            event_probs = all_events[seq_idx, time_idx, :]
            dominant_event_idx = np.argmax(event_probs)
            dominant_event = event_names[dominant_event_idx]
            dominant_prob = event_probs[dominant_event_idx]
            
            row = {
                'sequence_id': seq_idx,
                'time_step': time_idx,
                'position_x': x,
                'position_y': y,
                'position_z': z,
                'velocity_x': vx,
                'velocity_y': vy,
                'velocity_z': vz,
                'energy': energy,
                'dominant_event': dominant_event,
                'dominant_probability': dominant_prob
            }
            
            # Add individual event probabilities
            for i, event_name in enumerate(event_names):
                row[f'prob_{event_name}'] = event_probs[i]
            
            detailed_data.append(row)
    
    # Save to CSV
    df_detailed = pd.DataFrame(detailed_data)
    csv_filename = 'lstm_wgan_geant4_style_output.csv'
    df_detailed.to_csv(csv_filename, index=False)
    
    g4_output.print_success(f"Detailed data saved to: {csv_filename}")
    g4_output.print_info(f"CSV contains {len(detailed_data):,} rows with {len(detailed_data[0])} columns")
    
    # Calculate accuracy (simplified)
    g4_output.print_section("ACCURACY ASSESSMENT", 1)
    g4_output.print_info("Performing quick accuracy assessment...")
    
    # Simple accuracy based on energy positivity and physics constraints
    positive_energy = np.sum(energy_values > 0)
    energy_accuracy = (positive_energy / len(energy_values)) * 100
    
    # Event probability sum validation (should be close to 1.0)
    event_prob_sums = np.sum(all_events, axis=2)
    valid_prob_sums = np.sum(np.abs(event_prob_sums - 1.0) < 0.1)
    prob_accuracy = (valid_prob_sums / event_prob_sums.size) * 100
    
    overall_accuracy = (energy_accuracy + prob_accuracy) / 2
    
    g4_output.print_info(f"Energy positivity: {energy_accuracy:.1f}%")
    g4_output.print_info(f"Event probability validation: {prob_accuracy:.1f}%")
    g4_output.print_success(f"Overall accuracy: {overall_accuracy:.1f}%")
    
    # Final summary
    model_info = {
        'type': 'LSTM-WGAN with Events',
        'device': str(device),
        'sequences': num_sequences,
        'time_steps': all_sequences.shape[1],
        'features': all_sequences.shape[2]
    }
    
    g4_output.print_final_summary(
        total_time=generation_time,
        total_events=total_time_steps,
        accuracy=overall_accuracy,
        model_info=model_info
    )
    
    # Show file information
    g4_output.print_section("OUTPUT FILES", 1)
    g4_output.print_info(f"Generated CSV file: {csv_filename}")
    g4_output.print_info(f"File size: {len(detailed_data):,} rows × {len(detailed_data[0])} columns")
    g4_output.print_info("File contains: positions, velocities, energies, and event probabilities")
    
    print(f"\n{Colors.BRIGHT_GREEN}✓ GEANT4-style data generation completed successfully!{Colors.RESET}")

if __name__ == "__main__":
    main()
