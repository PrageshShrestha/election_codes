import time
import subprocess
import psutil
import os
from pathlib import Path

def get_gpu_utilization():
    """Get GPU utilization using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except:
        pass
    return 0

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_script(script_name, duration=30):
    """Benchmark a script for given duration"""
    print(f"Testing {script_name} for {duration} seconds...")
    
    start_time = time.time()
    gpu_samples = []
    memory_samples = []
    
    # Start the script in background
    process = subprocess.Popen(['python', script_name], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
    
    # Monitor for duration
    while time.time() - start_time < duration:
        gpu_util = get_gpu_utilization()
        memory_mb = get_memory_usage()
        gpu_samples.append(gpu_util)
        memory_samples.append(memory_mb)
        time.sleep(1)
        
        # Check if process finished early
        if process.poll() is not None:
            break
    
    # Terminate process
    process.terminate()
    process.wait()
    
    # Calculate averages
    avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
    avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
    max_memory = max(memory_samples) if memory_samples else 0
    
    return {
        'avg_gpu_utilization': avg_gpu,
        'avg_memory_mb': avg_memory,
        'max_memory_mb': max_memory,
        'samples': len(gpu_samples)
    }

def main():
    print("OCR Performance Benchmark")
    print("=" * 50)
    
    scripts = [
        ('main.py', 'Sequential Processing'),
        ('main_multithreaded.py', 'Multithreaded Processing'),
        ('main_ultra_fast.py', 'Ultra-Fast Processing')
    ]
    
    results = {}
    
    for script, description in scripts:
        if Path(script).exists():
            print(f"\nBenchmarking: {description}")
            results[script] = benchmark_script(script, duration=30)
            print(f"Average GPU Utilization: {results[script]['avg_gpu_utilization']:.1f}%")
            print(f"Average Memory Usage: {results[script]['avg_memory_mb']:.1f} MB")
            print(f"Peak Memory Usage: {results[script]['max_memory_mb']:.1f} MB")
        else:
            print(f"\n{script} not found, skipping...")
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 50)
    
    for script, description in scripts:
        if script in results:
            result = results[script]
            print(f"{description}:")
            print(f"  GPU Utilization: {result['avg_gpu_utilization']:.1f}%")
            print(f"  Memory Usage: {result['avg_memory_mb']:.1f} MB (avg), {result['max_memory_mb']:.1f} MB (peak)")

if __name__ == "__main__":
    main()
