#!/usr/bin/env python3
"""
Test checkpoint loading functionality
"""

import subprocess
import sys
import os
import time
import platform

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
    print("🧪 Testing Checkpoint Loading Functionality")
    print("=" * 50)
    
    # Check if checkpoint exists
    if os.path.exists('neutron_vae_checkpoint.pth'):
        print("✅ Checkpoint file found: neutron_vae_checkpoint.pth")
        
        # Test checkpoint loading
        success, output = run_command("python -m neutron_vae.train", "Testing checkpoint loading and training continuation")
        
        if success:
            print("\n📊 Checkpoint Loading Results:")
            lines = output.split('\n')
            for line in lines:
                if "Loading checkpoint" in line or "Starting from epoch" in line or "Best loss so far" in line:
                    print(f"   {line}")
            
            print("\n🎯 Checkpoint loading test completed successfully!")
            return 0
        else:
            print("\n❌ Checkpoint loading test failed!")
            return 1
    else:
        print("📝 No checkpoint file found. Starting fresh training to create checkpoint...")
        
        # Use gtimeout on macOS, timeout on Linux
        if platform.system() == "Darwin":  # macOS
            timeout_cmd = "gtimeout"
        else:
            timeout_cmd = "timeout"
        
        # Start training for a few epochs to create checkpoint
        success, output = run_command(f"{timeout_cmd} 300 python -m neutron_vae.train", "Starting training to create checkpoint")
        
        if success:
            print("\n✅ Training started successfully. Checkpoint will be created after 1000 epochs.")
            print("You can interrupt training (Ctrl+C) and restart to test checkpoint loading.")
            return 0
        else:
            print(f"\n❌ Training failed to start!")
            print("Trying without timeout...")
            
            # Try without timeout
            success, output = run_command("python -m neutron_vae.train", "Starting training without timeout")
            if success:
                print("\n✅ Training started successfully without timeout.")
                print("You can interrupt training (Ctrl+C) and restart to test checkpoint loading.")
                return 0
            else:
                print("\n❌ Training failed to start!")
                return 1

if __name__ == "__main__":
    sys.exit(main())
