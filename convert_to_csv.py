#!/usr/bin/env python3
"""
Convert generated neutron tracks to CSV format
"""

import numpy as np
import pandas as pd
import os

def convert_to_csv():
    """Convert generated neutron tracks to CSV format"""
    print("🔄 Converting generated tracks to CSV format...")
    
    # Load the generated tracks
    if os.path.exists('generated_neutron_tracks.npy'):
        tracks = np.load('generated_neutron_tracks.npy')
        print(f"✅ Loaded {len(tracks)} tracks from generated_neutron_tracks.npy")
    else:
        print("❌ generated_neutron_tracks.npy not found!")
        return
    
    # Create CSV data
    csv_data = []
    
    for track_idx, track in enumerate(tracks):
        for point_idx, point in enumerate(track):
            csv_data.append({
                'track_id': track_idx + 1,
                'point_id': point_idx + 1,
                'x': point[0],
                'y': point[1], 
                'z': point[2]
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    
    # Save to CSV
    csv_filename = 'generated_neutron_tracks.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"✅ Converted to CSV: {csv_filename}")
    print(f"📊 CSV Statistics:")
    print(f"   Total tracks: {len(tracks)}")
    print(f"   Total points: {len(csv_data)}")
    print(f"   Points per track: {len(tracks[0])}")
    print(f"   Columns: {list(df.columns)}")
    
    # Show first few rows
    print(f"\n📋 First 10 rows:")
    print(df.head(10))
    
    return df

if __name__ == "__main__":
    convert_to_csv()
