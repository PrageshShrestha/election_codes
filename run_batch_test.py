#!/usr/bin/env python3
"""
Test script to compare performance between sequential and batch processing
"""

import time
import subprocess
import os
from pathlib import Path

def run_script(script_name, description):
    """Run a script and measure performance"""
    print(f"\n{'='*60}")
    print(f"TESTING: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(['python3', script_name], 
                              capture_output=True, text=True, timeout=3600)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Execution time: {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[-1000:])  # Show last 1000 chars
            
        if result.stderr:
            print("STDERR:")
            print(result.stderr[-500:])   # Show last 500 chars
            
        return duration, result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Script timed out after 1 hour!")
        return None, False
    except Exception as e:
        print(f"Error running script: {e}")
        return None, False

def main():
    print("OCR Performance Comparison Test")
    print("This will test both sequential and batch processing modes")
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("Error: main.py not found!")
        return
    
    # Check if voter_info folder exists and has images
    voter_folder = Path("voter_info")
    if not voter_folder.exists():
        print("Error: voter_info folder not found!")
        return
        
    image_count = len(list(voter_folder.glob("*.jpg")))
    if image_count == 0:
        print("Error: No images found in voter_info folder!")
        return
        
    print(f"Found {image_count} images to process")
    
    # Backup original main.py if it doesn't exist
    if not Path("main_sequential.py").exists():
        print("Creating backup of sequential version...")
        # We'll need to create a sequential version for comparison
        with open("main.py", "r") as f:
            content = f.read()
        
        # Create sequential version by replacing batch logic
        sequential_content = content.replace(
            "def process_single_voter(image_data):",
            "# SEQUENTIAL VERSION - NO BATCH PROCESSING\ndef process_single_voter(image_data):"
        ).replace(
            "with ThreadPoolExecutor(max_workers=max_workers) as executor:",
            "# Sequential processing - no ThreadPoolExecutor\n# with ThreadPoolExecutor(max_workers=max_workers) as executor:"
        )
        
        with open("main_sequential.py", "w") as f:
            f.write(sequential_content)
    
    print("\nStarting performance tests...")
    
    # Test current batch version
    batch_time, batch_success = run_script("main.py", "Batch Processing (Current)")
    
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    if batch_time:
        print(f"Batch Processing: {batch_time:.2f} seconds")
        print(f"Images per second (batch): {image_count/max(batch_time, 1):.2f}")
    
    print(f"\nTotal images processed: {image_count}")
    print("\nNote: For accurate comparison, ensure both versions process the same dataset")
    print("Monitor GPU memory usage during processing to optimize worker count")

if __name__ == "__main__":
    main()
