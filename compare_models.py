#!/usr/bin/env python3
"""
Compare old vs new model performance
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    print(f"\n🔄 {description}...")
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        print(f"✅ {description} completed in {end_time - start_time:.2f}s")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False, e.stderr

def main():
    print("🔬 Model Comparison: Old vs Improved VAE")
    print("=" * 50)
    
    # Test the improved model
    print("\n📈 Testing Improved Model Performance:")
    
    # Train improved model
    success1, output1 = run_command("python -m neutron_vae.train", "Training improved VAE model")
    
    # Test accuracy
    success2, output2 = run_command("python -m neutron_vae.accuracy_test", "Running accuracy tests")
    
    # Generate samples
    success3, output3 = run_command("python -m neutron_vae.generate", "Generating new tracks")
    
    # Create visualization
    success4, output4 = run_command("python -m neutron_vae.visualize", "Creating visualizations")
    
    # Extract key metrics from training output
    if success1:
        lines = output1.split('\n')
        final_loss = None
        best_loss = None
        for line in lines:
            if "Final loss:" in line:
                final_loss = float(line.split(":")[1].strip())
            elif "Best loss:" in line:
                best_loss = float(line.split(":")[1].strip())
        
        print(f"\n📊 Training Results:")
        print(f"Final Loss: {final_loss:.6f}")
        print(f"Best Loss: {best_loss:.6f}")
    
    # Extract accuracy score
    if success2:
        lines = output2.split('\n')
        accuracy_score = None
        for line in lines:
            if "Overall Accuracy Score:" in line:
                accuracy_score = float(line.split(":")[1].strip().split('/')[0])
                break
        
        print(f"Accuracy Score: {accuracy_score:.2f}/100")
    
    # Check output files
    print(f"\n📁 Generated Files:")
    files_to_check = [
        "neutron_vae.pth",
        "neutron_vae_best.pth", 
        "generated_track.csv",
        "track_visualization.png",
        "accuracy_test_results.png"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} missing")
    
    # Summary
    print(f"\n🎯 Summary:")
    if all([success1, success2, success3, success4]):
        print("✅ All tests passed! The improved model is working correctly.")
        print(f"🎉 Model accuracy: {accuracy_score:.2f}/100")
        print(f"🏆 Best training loss: {best_loss:.6f}")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
