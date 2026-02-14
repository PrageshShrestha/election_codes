#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os
from pytorch_cnn_gru_model import CNN_GRU_Model

def test_single_image(image_path, model_path='pytorch_cnn_gru_devanagari_digits.pth'):
    """
    Test single image and check if it's Devanagari digit '1'
    Returns: True if predicted as digit 1, False otherwise
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = CNN_GRU_Model(input_shape=(3, 64, 64), num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Get results
        predicted_class_idx = predicted_class.item()
        confidence_score = confidence.item() * 100
        is_digit_one = (predicted_class_idx == 1)
        
        # Devanagari digits
        devanagari_digits = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
        predicted_digit = devanagari_digits[predicted_class_idx]
        
        # Print results
        print(f"🖼️  Testing: {os.path.basename(image_path)}")
        print(f"🔢 Predicted: Digit {predicted_class_idx} ({predicted_digit})")
        print(f"📊 Confidence: {confidence_score:.2f}%")
        print(f"🎯 Is Devanagari '१' (Digit 1): {'✅ YES' if is_digit_one else '❌ NO'}")
        
        return is_digit_one
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_digit_model.py <image_path>")
        print("Example: python test_digit_model.py test_digit_1.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = test_single_image(image_path)
    
    # Exit code: 0 if digit 1, 1 if not digit 1
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()
