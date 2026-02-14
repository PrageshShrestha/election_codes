#!/usr/bin/env python3
import os
import urllib.request
import zipfile
import torch
from pytorch_cnn_gru_model import main

def download_dhcd_dataset():
    """Download DHCD dataset if not available"""
    data_dir = "./DevanagariHandwrittenCharacterDataset"
    
    if os.path.exists(data_dir):
        print(f"Dataset already exists at {data_dir}")
        return True
    
    print("Downloading DHCD dataset...")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download URL for DHCD dataset (you may need to update this URL)
    # Using a known source for the dataset
    try:
        # Try downloading from a reliable source
        url = "https://github.com/amir-bdz/Devanagari-Handwritten-Character-Dataset/raw/master/DevanagariHandwrittenCharacterDataset.zip"
        zip_path = "./dhcd_dataset.zip"
        
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up
        os.remove(zip_path)
        
        print("Dataset downloaded and extracted successfully!")
        return True
        
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Please manually download the DHCD dataset and place it in the current directory.")
        return False

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("GPU is not available. Using CPU.")
        return False

def create_requirements():
    """Create requirements.txt file"""
    requirements = """torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pillow>=8.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("requirements.txt created successfully!")

def main_setup():
    """Main setup function"""
    print("=== DHCD Devanagari Digits Training Setup ===")
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Create requirements file
    create_requirements()
    
    # Download dataset
    if not download_dhcd_dataset():
        print("Failed to set up dataset. Please download manually.")
        return
    
    print("\n=== Setup Complete ===")
    print("To install dependencies: pip install -r requirements.txt")
    print("To start training: python pytorch_cnn_gru_model.py")
    
    if gpu_available:
        print("Training will use GPU acceleration.")
    else:
        print("Training will use CPU (consider using GPU for faster training).")
    
    # Ask user if they want to start training immediately
    response = input("\nDo you want to start training now? (y/n): ").lower().strip()
    if response == 'y':
        print("\nStarting training...")
        main()

if __name__ == "__main__":
    main_setup()
