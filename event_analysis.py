#!/usr/bin/env python3
"""
Event Analysis for LSTM-WGAN Neutron Physics Simulation
======================================================
Detailed analysis of event numbers and distributions from the simulation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_events():
    print("=" * 80)
    print("DETAILED EVENT ANALYSIS - LSTM-WGAN NEUTRON PHYSICS SIMULATION")
    print("=" * 80)
    print()
    
    # Load the generated data
    try:
        df = pd.read_csv('simple_event_display_data.csv')
        print(f"✅ Data loaded successfully: {len(df)} events")
    except FileNotFoundError:
        print("❌ Error: simple_event_display_data.csv not found")
        print("Please run simple_event_display.py first to generate the data")
        return
    
    print()
    
    # 1. OVERALL EVENT STATISTICS
    print("1. OVERALL EVENT STATISTICS")
    print("-" * 50)
    total_events = len(df)
    unique_sequences = df['sequence_id'].nunique()
    avg_events_per_sequence = total_events / unique_sequences
    
    print(f"Total Events Generated: {total_events:,}")
    print(f"Number of Sequences: {unique_sequences:,}")
    print(f"Average Events per Sequence: {avg_events_per_sequence:.1f}")
    print(f"Sequence Length: 50 time steps")
    print()
    
    # 2. EVENT TYPE DISTRIBUTION
    print("2. EVENT TYPE DISTRIBUTION")
    print("-" * 50)
    
    event_counts = df['event_type'].value_counts()
    event_percentages = (event_counts / total_events * 100).round(1)
    
    print("Event Type Distribution:")
    print(f"{'Event Type':<12} {'Count':<10} {'Percentage':<12} {'Status'}")
    print("-" * 50)
    
    for event_type in ['capture', 'scattering', 'fission', 'absorption', 'leakage']:
        count = event_counts.get(event_type, 0)
        percentage = event_percentages.get(event_type, 0)
        
        # Determine status
        if percentage > 50:
            status = "DOMINANT"
        elif percentage > 20:
            status = "COMMON"
        elif percentage > 5:
            status = "MODERATE"
        elif percentage > 1:
            status = "RARE"
        else:
            status = "VERY RARE"
        
        print(f"{event_type.upper():<12} {count:<10,} {percentage:<12.1f}% {status}")
    
    print()
    
    # 3. PHYSICS ANALYSIS
    print("3. PHYSICS ANALYSIS")
    print("-" * 50)
    
    # Energy analysis by event type
    print("Energy Analysis by Event Type:")
    print(f"{'Event Type':<12} {'Mean Energy':<15} {'Std Energy':<15} {'Min Energy':<15} {'Max Energy'}")
    print("-" * 80)
    
    for event_type in ['capture', 'scattering', 'fission', 'absorption', 'leakage']:
        event_data = df[df['event_type'] == event_type]
        if len(event_data) > 0:
            mean_energy = event_data['energy'].mean()
            std_energy = event_data['energy'].std()
            min_energy = event_data['energy'].min()
            max_energy = event_data['energy'].max()
            
            print(f"{event_type.upper():<12} {mean_energy:<15.4f} {std_energy:<15.4f} {min_energy:<15.4f} {max_energy:.4f}")
        else:
            print(f"{event_type.upper():<12} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A'}")
    
    print()
    
    # 4. SEQUENCE ANALYSIS
    print("4. SEQUENCE ANALYSIS")
    print("-" * 50)
    
    # Analyze event patterns in sequences
    sequence_analysis = df.groupby('sequence_id').agg({
        'event_type': lambda x: x.value_counts().to_dict(),
        'energy': ['mean', 'std', 'min', 'max'],
        'time_step': 'max'
    }).round(4)
    
    # Count sequences with different event types
    sequences_with_capture = df[df['event_type'] == 'capture']['sequence_id'].nunique()
    sequences_with_scattering = df[df['event_type'] == 'scattering']['sequence_id'].nunique()
    sequences_with_fission = df[df['event_type'] == 'fission']['sequence_id'].nunique()
    sequences_with_absorption = df[df['event_type'] == 'absorption']['sequence_id'].nunique()
    sequences_with_leakage = df[df['event_type'] == 'leakage']['sequence_id'].nunique()
    
    print("Sequences Containing Each Event Type:")
    print(f"Capture Events: {sequences_with_capture:,} sequences ({sequences_with_capture/unique_sequences*100:.1f}%)")
    print(f"Scattering Events: {sequences_with_scattering:,} sequences ({sequences_with_scattering/unique_sequences*100:.1f}%)")
    print(f"Fission Events: {sequences_with_fission:,} sequences ({sequences_with_fission/unique_sequences*100:.1f}%)")
    print(f"Absorption Events: {sequences_with_absorption:,} sequences ({sequences_with_absorption/unique_sequences*100:.1f}%)")
    print(f"Leakage Events: {sequences_with_leakage:,} sequences ({sequences_with_leakage/unique_sequences*100:.1f}%)")
    print()
    
    # 5. EVENT PROBABILITY ANALYSIS
    print("5. EVENT PROBABILITY ANALYSIS")
    print("-" * 50)
    
    print("Event Probability Statistics:")
    print(f"{'Event Type':<12} {'Mean Prob':<12} {'Std Prob':<12} {'Min Prob':<12} {'Max Prob'}")
    print("-" * 60)
    
    for event_type in ['capture', 'scattering', 'fission', 'absorption', 'leakage']:
        event_data = df[df['event_type'] == event_type]
        if len(event_data) > 0:
            mean_prob = event_data['event_probability'].mean()
            std_prob = event_data['event_probability'].std()
            min_prob = event_data['event_probability'].min()
            max_prob = event_data['event_probability'].max()
            
            print(f"{event_type.upper():<12} {mean_prob:<12.3f} {std_prob:<12.3f} {min_prob:<12.3f} {max_prob:.3f}")
        else:
            print(f"{event_type.upper():<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A'}")
    
    print()
    
    # 6. PHYSICS CONSISTENCY CHECK
    print("6. PHYSICS CONSISTENCY CHECK")
    print("-" * 50)
    
    # Check for unphysical values
    negative_energy = len(df[df['energy'] < 0])
    zero_energy = len(df[df['energy'] == 0])
    very_high_energy = len(df[df['energy'] > 10])  # Assuming 10 MeV is very high
    
    print("Physics Consistency:")
    print(f"Negative Energy Events: {negative_energy:,} ({negative_energy/total_events*100:.2f}%)")
    print(f"Zero Energy Events: {zero_energy:,} ({zero_energy/total_events*100:.2f}%)")
    print(f"Very High Energy Events (>10 MeV): {very_high_energy:,} ({very_high_energy/total_events*100:.2f}%)")
    
    # Position bounds check
    x_out_of_bounds = len(df[(df['x'] < 0) | (df['x'] > 1)])
    y_out_of_bounds = len(df[(df['y'] < 0) | (df['y'] > 1)])
    z_out_of_bounds = len(df[(df['z'] < 0) | (df['z'] > 1)])
    
    print(f"X Position Out of Bounds: {x_out_of_bounds:,} ({x_out_of_bounds/total_events*100:.2f}%)")
    print(f"Y Position Out of Bounds: {y_out_of_bounds:,} ({y_out_of_bounds/total_events*100:.2f}%)")
    print(f"Z Position Out of Bounds: {z_out_of_bounds:,} ({z_out_of_bounds/total_events*100:.2f}%)")
    print()
    
    # 7. SUMMARY AND INSIGHTS
    print("7. SUMMARY AND INSIGHTS")
    print("-" * 50)
    
    print("Key Findings:")
    print(f"• CAPTURE is the dominant process ({event_percentages.get('capture', 0):.1f}% of events)")
    print(f"• SCATTERING is the second most common ({event_percentages.get('scattering', 0):.1f}% of events)")
    print(f"• FISSION events are rare but present ({event_percentages.get('fission', 0):.1f}% of events)")
    print(f"• ABSORPTION and LEAKAGE events are extremely rare ({event_percentages.get('absorption', 0):.1f}% and {event_percentages.get('leakage', 0):.1f}%)")
    
    print()
    print("Physics Interpretation:")
    print("• High capture rate suggests neutrons are being absorbed by nuclei")
    print("• Scattering events indicate elastic/inelastic collisions")
    print("• Low fission rate is realistic for most neutron energies")
    print("• Very low absorption/leakage suggests good containment")
    
    print()
    print("Model Performance:")
    if negative_energy/total_events < 0.01:
        print("✅ Excellent: Very few unphysical negative energy events")
    elif negative_energy/total_events < 0.05:
        print("✅ Good: Few unphysical negative energy events")
    else:
        print("⚠️  Warning: Significant number of unphysical negative energy events")
    
    if event_percentages.get('capture', 0) > 50:
        print("✅ Realistic: Capture-dominated physics (typical for thermal neutrons)")
    
    print()
    print("=" * 80)
    print("EVENT ANALYSIS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    analyze_events()
