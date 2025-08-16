#!/usr/bin/env python3
"""
GPU Memory Monitoring Script
Real-time monitoring of CUDA memory usage for the Neutron VAE project
"""

import torch
import time
import psutil
import os
import sys
from datetime import datetime

def get_gpu_memory_info():
    """Get detailed GPU memory information"""
    if not torch.cuda.is_available():
        return None
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    memory_info = {
        'device_name': props.name,
        'total_memory': props.total_memory,
        'total_memory_gb': props.total_memory / 1024**3,
        'allocated_memory': torch.cuda.memory_allocated(device),
        'allocated_memory_mb': torch.cuda.memory_allocated(device) / 1024**2,
        'reserved_memory': torch.cuda.memory_reserved(device),
        'reserved_memory_mb': torch.cuda.memory_reserved(device) / 1024**2,
        'max_memory_allocated': torch.cuda.max_memory_allocated(device),
        'max_memory_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2,
        'memory_fraction': torch.cuda.memory_reserved(device) / props.total_memory,
        'utilization_percent': (torch.cuda.memory_allocated(device) / props.total_memory) * 100
    }
    
    return memory_info

def get_system_memory_info():
    """Get system memory information"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / 1024**3,
        'available_gb': memory.available / 1024**3,
        'used_gb': memory.used / 1024**3,
        'percent_used': memory.percent
    }

def print_memory_status(gpu_info, system_info):
    """Print current memory status"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n📊 MEMORY STATUS - {timestamp}")
    print("=" * 60)
    
    if gpu_info:
        print(f"🎯 GPU: {gpu_info['device_name']}")
        print(f"   Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
        print(f"   Allocated: {gpu_info['allocated_memory_mb']:.1f} MB")
        print(f"   Reserved: {gpu_info['reserved_memory_mb']:.1f} MB")
        print(f"   Peak Allocated: {gpu_info['max_memory_allocated_mb']:.1f} MB")
        print(f"   Utilization: {gpu_info['utilization_percent']:.1f}%")
        print(f"   Memory Fraction: {gpu_info['memory_fraction']:.2%}")
    else:
        print("❌ GPU not available")
    
    if system_info:
        print(f"\n💻 System Memory:")
        print(f"   Total: {system_info['total_gb']:.2f} GB")
        print(f"   Used: {system_info['used_gb']:.2f} GB")
        print(f"   Available: {system_info['available_gb']:.2f} GB")
        print(f"   Usage: {system_info['percent_used']:.1f}%")

def monitor_memory_continuously(interval=5, duration=None):
    """Monitor memory usage continuously"""
    print("🔍 STARTING GPU MEMORY MONITORING")
    print("=" * 60)
    print(f"Monitoring interval: {interval} seconds")
    if duration:
        print(f"Duration: {duration} seconds")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    start_time = time.time()
    peak_gpu_memory = 0
    peak_system_memory = 0
    
    try:
        while True:
            gpu_info = get_gpu_memory_info()
            system_info = get_system_memory_info()
            
            # Update peak values
            if gpu_info:
                peak_gpu_memory = max(peak_gpu_memory, gpu_info['allocated_memory_mb'])
            if system_info:
                peak_system_memory = max(peak_system_memory, system_info['used_gb'])
            
            print_memory_status(gpu_info, system_info)
            
            # Check if duration exceeded
            if duration and (time.time() - start_time) > duration:
                break
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Monitoring stopped by user")
    
    # Print summary
    print(f"\n📈 MONITORING SUMMARY")
    print("=" * 60)
    print(f"Duration: {(time.time() - start_time):.1f} seconds")
    if gpu_info:
        print(f"Peak GPU Memory: {peak_gpu_memory:.1f} MB")
    if system_info:
        print(f"Peak System Memory: {peak_system_memory:.2f} GB")

def test_memory_allocation():
    """Test memory allocation with different tensor sizes"""
    print("🧪 TESTING MEMORY ALLOCATION")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for testing")
        return
    
    device = torch.device("cuda")
    
    # Test different tensor sizes
    tensor_sizes = [
        (1000, 200, 3),    # Small
        (5000, 200, 3),    # Medium
        (10000, 200, 3),   # Large
        (20000, 200, 3),   # Very Large
    ]
    
    for size in tensor_sizes:
        try:
            print(f"\nTesting tensor size: {size}")
            
            # Clear memory
            torch.cuda.empty_cache()
            
            # Create tensor
            tensor = torch.randn(*size, device=device)
            memory_used = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            
            print(f"   ✅ Success - Allocated: {memory_used:.1f} MB, Reserved: {memory_reserved:.1f} MB")
            
            # Calculate tensor size in MB
            tensor_size_mb = tensor.numel() * 4 / 1024**2  # 4 bytes per float32
            print(f"   Tensor size: {tensor_size_mb:.1f} MB")
            
            del tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ❌ Out of memory")
                break
            else:
                print(f"   ❌ Error: {e}")
                break

def optimize_memory_settings():
    """Optimize memory settings for maximum usage"""
    print("⚙️ OPTIMIZING MEMORY SETTINGS")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_memory_gb = props.total_memory / 1024**3
    
    print(f"🎯 GPU: {props.name}")
    print(f"📊 Total Memory: {total_memory_gb:.2f} GB")
    
    # Test different memory fractions
    memory_fractions = [0.8, 0.85, 0.9, 0.95, 0.98]
    
    for fraction in memory_fractions:
        try:
            print(f"\nTesting memory fraction: {fraction*100:.0f}%")
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(fraction)
            
            # Clear memory
            torch.cuda.empty_cache()
            
            # Test allocation
            test_size = int(props.total_memory * fraction * 0.8 / 4)  # Use 80% of allocated memory
            test_tensor = torch.randn(test_size, device=device)
            
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
            utilization = allocated_mb / (total_memory_gb * 1024) * 100
            
            print(f"   ✅ Success - Allocated: {allocated_mb:.1f} MB ({utilization:.1f}%)")
            
            del test_tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"   ❌ Failed: {e}")
            break
    
    # Reset to default
    torch.cuda.set_per_process_memory_fraction(1.0)
    print(f"\n✅ Memory settings reset to default")

def main():
    """Main function"""
    print("🔍 GPU MEMORY MONITORING TOOL")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "monitor":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else None
            monitor_memory_continuously(interval, duration)
            
        elif command == "test":
            test_memory_allocation()
            
        elif command == "optimize":
            optimize_memory_settings()
            
        elif command == "status":
            gpu_info = get_gpu_memory_info()
            system_info = get_system_memory_info()
            print_memory_status(gpu_info, system_info)
            
        else:
            print("❌ Unknown command. Available commands:")
            print("   monitor [interval] [duration] - Monitor memory continuously")
            print("   test - Test memory allocation")
            print("   optimize - Optimize memory settings")
            print("   status - Show current memory status")
    else:
        # Default: show current status
        gpu_info = get_gpu_memory_info()
        system_info = get_system_memory_info()
        print_memory_status(gpu_info, system_info)
        
        print(f"\n💡 Usage:")
        print(f"   python monitor_gpu_memory.py monitor 5 60  # Monitor for 60 seconds")
        print(f"   python monitor_gpu_memory.py test          # Test memory allocation")
        print(f"   python monitor_gpu_memory.py optimize      # Optimize settings")
        print(f"   python monitor_gpu_memory.py status        # Show current status")

if __name__ == "__main__":
    main()
