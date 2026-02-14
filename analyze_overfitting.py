#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Overfitting Issue
Investigates why the model overfits to digit 8
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# ────────────────────────────────────────────────
#   Load Model and Dataset for Analysis
# ────────────────────────────────────────────────

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SqueezeExcitation(torch.nn.Module):
    def __init__(self, channels, ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // ratio, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channels // ratio, channels, bias=False),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return F.relu(out)

class OptimizedNumbersModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(OptimizedNumbersModel, self).__init__()
        self.initial_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )
        self.layer1 = torch.nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
            torch.nn.Dropout2d(0.2)
        )
        self.layer2 = torch.nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
            torch.nn.Dropout2d(0.3)
        )
        self.layer3 = torch.nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            torch.nn.Dropout2d(0.3)
        )
        self.se = SqueezeExcitation(128, ratio=16)
        self.dw_conv = torch.nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=1),
            torch.nn.Dropout2d(0.4)
        )
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(512, 256, bias=False),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.se(x)
        x = self.dw_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ────────────────────────────────────────────────
#   Analysis Functions
# ────────────────────────────────────────────────

def load_model():
    """Load trained model"""
    model = OptimizedNumbersModel(num_classes=10)
    checkpoint = torch.load('dhcd_numbers_best.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def analyze_dataset_imbalance():
    """Analyze class distribution in training data"""
    print("🔍 Analyzing Dataset Class Distribution...")
    
    # Load DHCD dataset
    DHCD_PATH = "/home/pragesh-shrestha/Desktop/nishant_sir/DHCD_Dataset/dataset/dataset.npz"
    data = np.load(DHCD_PATH)
    train_images = data['arr_0']
    train_labels = data['arr_1']
    
    # Filter only number classes
    NUMBER_CLASSES = {35: '०', 36: '१', 37: '२', 38: '३', 39: '४', 40: '५', 41: '६', 42: '७', 43: '८', 44: '९'}
    
    number_indices = [i for i, label in enumerate(train_labels) if label in NUMBER_CLASSES.keys()]
    number_labels = train_labels[number_indices]
    
    # Count distribution
    unique, counts = np.unique(number_labels, return_counts=True)
    
    print("\n📊 Training Data Class Distribution:")
    print("-" * 40)
    for label, count in zip(unique, counts):
        devanagari = NUMBER_CLASSES[label]
        new_class = list(NUMBER_CLASSES.keys()).index(label)
        print(f"   {devanagari} (Class {new_class}): {count} samples")
    
    # Check for imbalance
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\n⚖️  Imbalance Analysis:")
    print(f"   Max samples: {max_count}")
    print(f"   Min samples: {min_count}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 1.5:
        print("   ⚠️  Dataset is IMBALANCED!")
    else:
        print("   ✅ Dataset is relatively balanced")
    
    return unique, counts

def analyze_model_confidence(model):
    """Analyze model confidence patterns"""
    print("\n🔍 Analyzing Model Confidence Patterns...")
    
    # Test on a few sample images
    test_images = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png"]
    
    print("\n📊 Individual Image Analysis:")
    print("-" * 60)
    
    for img_name in test_images:
        if os.path.exists(img_name):
            # Load and preprocess
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(np.expand_dims(img, 0), 0)
            img_tensor = torch.FloatTensor(img).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                probs = probabilities.cpu().numpy()[0]
                
                # Get top 3
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3_probs = probs[top3_idx]
                
                print(f"\n📸 {img_name}:")
                for i, (idx, prob) in enumerate(zip(top3_idx, top3_probs)):
                    devanagari = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९'][idx]
                    print(f"   {i+1}. {devanagari} (Class {idx}): {prob*100:.2f}%")

def analyze_feature_similarity(model):
    """Analyze if images are too similar to digit 8"""
    print("\n🔍 Analyzing Feature Similarity to Digit 8...")
    
    # Load some digit 8 samples from dataset to see what the model learned
    DHCD_PATH = "/home/pragesh-shrestha/Desktop/nishant_sir/DHCD_Dataset/dataset/dataset.npz"
    data = np.load(DHCD_PATH)
    train_images = data['arr_0']
    train_labels = data['arr_1']
    
    # Find digit 8 samples (label 43)
    digit_8_indices = np.where(train_labels == 43)[0]
    
    print(f"\n📊 Found {len(digit_8_indices)} digit ८ samples in training data")
    
    # Show a few digit 8 samples
    print("\n🔍 Sample Digit ८ Images from Training Data:")
    print("-" * 50)
    
    for i in range(min(5, len(digit_8_indices))):
        idx = digit_8_indices[i]
        img = train_images[idx]
        
        # Basic statistics
        mean_val = np.mean(img)
        std_val = np.std(img)
        non_zero_ratio = np.count_nonzero(img) / img.size
        
        print(f"   Sample {i+1}: Mean={mean_val:.2f}, Std={std_val:.2f}, Non-zero={non_zero_ratio*100:.1f}%")

def analyze_preprocessing_bias():
    """Analyze if preprocessing creates bias towards digit 8"""
    print("\n🔍 Analyzing Preprocessing Bias...")
    
    test_images = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png"]
    
    print("\n📊 Preprocessed Image Statistics:")
    print("-" * 60)
    
    for img_name in test_images:
        if os.path.exists(img_name):
            # Load and preprocess with enhanced method
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            
            # Original stats
            orig_mean = np.mean(img)
            orig_std = np.std(img)
            
            # Enhanced preprocessing
            if img.mean() > 127:
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
            
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            kernel = np.ones((2,2), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            img = img.astype('float32') / 255.0
            img = cv2.equalizeHist((img * 255).astype(np.uint8)).astype('float32') / 255.0
            
            # Preprocessed stats
            proc_mean = np.mean(img)
            proc_std = np.std(img)
            
            print(f"\n📸 {img_name}:")
            print(f"   Original: Mean={orig_mean:.2f}, Std={orig_std:.2f}")
            print(f"   Processed: Mean={proc_mean:.2f}, Std={proc_std:.2f}")

def main():
    """Main analysis function"""
    print("=" * 80)
    print("🔍 OVERFITTING ANALYSIS - Why Digit ८?")
    print("=" * 80)
    
    # Load model
    model = load_model()
    
    # Run analyses
    analyze_dataset_imbalance()
    analyze_model_confidence(model)
    analyze_feature_similarity(model)
    analyze_preprocessing_bias()
    
    print("\n" + "=" * 80)
    print("🎯 POTENTIAL CAUSES OF OVERFITTING TO DIGIT ८:")
    print("=" * 80)
    print("1. 📊 Dataset Imbalance - More digit ८ samples")
    print("2. 🧠 Model Bias - Learned features favor digit ८")
    print("3. 🔧 Preprocessing Bias - Enhances digit २-like features")
    print("4. 📸 Image Similarity - Test images resemble digit ८")
    print("5. 🎯 Training Bias - Overexposure to digit ८ patterns")
    print("=" * 80)

if __name__ == "__main__":
    main()
