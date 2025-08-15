#!/usr/bin/env python3
"""
Test script to verify the complete neutron-vae pipeline works.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🧪 Testing Neutron VAE Pipeline")
    print("=" * 40)
    
    # Test 1: Training
    success1 = run_command("python -m neutron_vae.train", "Training VAE model")
    
    # Test 2: Generation
    success2 = run_command("python -m neutron_vae.generate", "Generating new track")
    
    # Test 3: Visualization
    success3 = run_command("python -m neutron_vae.visualize", "Creating visualization")
    
    # Check if output files exist
    print(f"\n📁 Checking output files...")
    files_to_check = [
        "neutron_vae.pth",
        "generated_track.csv", 
        "track_visualization.png"
    ]
    
    all_files_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} exists ({size} bytes)")
        else:
            print(f"❌ {file} missing")
            all_files_exist = False
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"Training: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Generation: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"Visualization: {'✅ PASS' if success3 else '❌ FAIL'}")
    print(f"Output files: {'✅ PASS' if all_files_exist else '❌ FAIL'}")
    
    if all([success1, success2, success3, all_files_exist]):
        print(f"\n🎉 All tests passed! The pipeline is working correctly.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
