#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Full DHCD Model to Numbers Only Model
Extracts and converts the full model to work with numbers only
"""

import torch
import torch.nn as nn

# Load the same model architecture from training
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return nn.functional.relu(out)

class OptimizedNumbersModel(nn.Module):
    def __init__(self, num_classes=10):
        super(OptimizedNumbersModel, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
            nn.Dropout2d(0.2)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
            nn.Dropout2d(0.3)
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            nn.Dropout2d(0.3)
        )
        
        self.se = SqueezeExcitation(128, ratio=16)
        
        self.dw_conv = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=1),
            nn.Dropout2d(0.4)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
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

def convert_full_model_to_numbers():
    """Convert full DHCD model to numbers only model"""
    print("🔄 Converting full DHCD model to numbers only...")
    
    # Load full model
    full_model = OptimizedNumbersModel(num_classes=46)
    full_checkpoint = torch.load('dhcd_pytorch_best.pth', map_location='cpu')
    full_model.load_state_dict(full_checkpoint)
    
    # Create numbers model
    numbers_model = OptimizedNumbersModel(num_classes=10)
    
    # Copy all layers except the final classification layer
    numbers_dict = numbers_model.state_dict()
    full_dict = full_model.state_dict()
    
    # Copy all weights except the final layer
    for key in full_dict:
        if 'fc.8' not in key:  # Skip the final 46-class layer
            if key in numbers_dict:
                numbers_dict[key] = full_dict[key].clone()
    
    # Initialize the final 10-class layer and copy relevant weights
    # We'll use the first 10 classes (numbers ०-९ correspond to classes 35-44 in original)
    original_fc_weight = full_dict['fc.8.weight']  # Shape: [46, 256]
    original_fc_bias = full_dict['fc.8.bias']      # Shape: [46]
    
    # Extract weights for number classes (35-44 in original, 0-9 in new)
    number_indices = list(range(35, 45))  # Classes ०-९
    numbers_dict['fc.8.weight'] = original_fc_weight[number_indices].clone()  # Shape: [10, 256]
    numbers_dict['fc.8.bias'] = original_fc_bias[number_indices].clone()      # Shape: [10]
    
    # Load the converted weights
    numbers_model.load_state_dict(numbers_dict)
    
    # Save the numbers model
    torch.save(numbers_model.state_dict(), 'dhcd_numbers_best.pth')
    print("✅ Numbers model saved as 'dhcd_numbers_best.pth'")
    
    # Test the conversion
    print("🧪 Testing conversion...")
    numbers_model.eval()  # Set to eval mode
    test_input = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        output = numbers_model(test_input)
        print(f"📊 Output shape: {output.shape}")  # Should be [1, 10]
        print(f"✅ Conversion successful!")

if __name__ == "__main__":
    convert_full_model_to_numbers()
