#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pytorch_cnn_gru_model import CNN_GRU_Model

class DevanagariDigitTester:
    def __init__(self, model_path='pytorch_cnn_gru_devanagari_digits.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Devanagari digit class names
        self.class_names = [f'Digit_{i}' for i in range(10)]
        self.devanagari_digits = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
        
        print(f"Model loaded on {self.device}")
        print(f"Available classes: {self.class_names}")
    
    def load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = CNN_GRU_Model(input_shape=(3, 64, 64), num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess single image for prediction"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        
        # Apply transforms
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
        
        return input_batch.to(self.device), original_image
    
    def predict_single_image(self, image_path, show_probabilities=True):
        """Predict single image and return results"""
        try:
            # Preprocess image
            input_batch, original_image = self.preprocess_image(image_path)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_batch)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                # Get all probabilities
                all_probs = probabilities.cpu().numpy().flatten()
                
            # Convert to numpy for easier handling
            predicted_class_idx = predicted_class.item()
            confidence_score = confidence.item() * 100
            
            # Results
            is_digit_one = (predicted_class_idx == 1)
            predicted_digit = self.devanagari_digits[predicted_class_idx]
            
            print(f"\n{'='*50}")
            print(f"TEST RESULTS FOR: {os.path.basename(image_path)}")
            print(f"{'='*50}")
            print(f"Predicted Class: {self.class_names[predicted_class_idx]}")
            print(f"Predicted Digit: {predicted_digit}")
            print(f"Confidence: {confidence_score:.2f}%")
            print(f"Is Devanagari '१' (Digit 1): {'✅ YES' if is_digit_one else '❌ NO'}")
            
            if show_probabilities:
                print(f"\nAll Class Probabilities:")
                print("-" * 30)
                for i, (class_name, prob, digit) in enumerate(zip(self.class_names, all_probs, self.devanagari_digits)):
                    marker = "👉" if i == predicted_class_idx else "  "
                    print(f"{marker} {class_name:8} ({digit}): {prob*100:5.2f}%")
            
            # Display the image
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title(f'Input Image\n{os.path.basename(image_path)}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            bars = plt.bar(range(10), all_probs * 100)
            bars[predicted_class_idx].set_color('red')
            plt.title('Class Probabilities')
            plt.xlabel('Devanagari Digits')
            plt.ylabel('Probability (%)')
            plt.xticks(range(10), self.devanagari_digits)
            plt.ylim(0, 100)
            
            # Add probability values on bars
            for i, (bar, prob) in enumerate(zip(bars, all_probs)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'prediction_result_{os.path.basename(image_path).split(".")[0]}.png')
            print(f"\nVisualization saved as: prediction_result_{os.path.basename(image_path).split('.')[0]}.png")
            plt.show()
            
            return {
                'predicted_class': predicted_class_idx,
                'predicted_digit': predicted_digit,
                'confidence': confidence_score,
                'is_digit_one': is_digit_one,
                'all_probabilities': all_probs.tolist()
            }
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def test_multiple_images(self, image_paths):
        """Test multiple images and show summary"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"TESTING {len(image_paths)} IMAGES FOR DEVANAGARI DIGIT '१'")
        print(f"{'='*60}")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n--- Test {i}/{len(image_paths)} ---")
            result = self.predict_single_image(image_path, show_probabilities=False)
            if result:
                results.append(result)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        digit_one_count = sum(1 for r in results if r['is_digit_one'])
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        
        print(f"Total Images Tested: {len(results)}")
        print(f"Images Predicted as '१' (Digit 1): {digit_one_count}")
        print(f"Images NOT '१': {len(results) - digit_one_count}")
        print(f"Average Confidence: {avg_confidence:.2f}%")
        
        if results:
            most_common_digit = max(set([r['predicted_digit'] for r in results]), 
                                  key=[r['predicted_digit'] for r in results].count)
            print(f"Most Common Prediction: {most_common_digit}")
        
        return results

def create_test_digit_images():
    """Create simple test images with Devanagari digit patterns"""
    print("Creating test images with Devanagari digit patterns...")
    
    # Create simple test images
    test_images = []
    
    for digit in range(10):
        # Create a simple pattern for each digit
        img_array = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        
        # Add some structure to make it look more like a digit
        if digit == 1:  # Make digit 1 more distinctive
            # Add vertical line pattern
            img_array[20:44, 30:34] = [255, 255, 255]
            # Add some noise
            noise = np.random.normal(0, 20, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Save test image
        image = Image.fromarray(img_array)
        filename = f'test_digit_{digit}.png'
        image.save(filename)
        test_images.append(filename)
        print(f"Created: {filename}")
    
    return test_images

def main():
    parser = argparse.ArgumentParser(description='Test Devanagari Digit Recognition Model')
    parser.add_argument('--image', type=str, help='Single image path to test')
    parser.add_argument('--model', type=str, default='pytorch_cnn_gru_devanagari_digits.pth', 
                       help='Model path')
    parser.add_argument('--create-test', action='store_true', 
                       help='Create test images and test them')
    
    args = parser.parse_args()
    
    # Initialize tester
    try:
        tester = DevanagariDigitTester(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the model file exists.")
        return
    
    if args.create_test:
        # Create test images and test them
        test_images = create_test_digit_images()
        results = tester.test_multiple_images(test_images)
        
        # Specifically check digit 1
        digit_1_result = next((r for r in results if r['predicted_class'] == 1), None)
        if digit_1_result:
            print(f"\n🎯 FOCUS ON DIGIT '१' (1):")
            print(f"Prediction: {'✅ CORRECT' if digit_1_result['is_digit_one'] else '❌ INCORRECT'}")
            print(f"Confidence: {digit_1_result['confidence']:.2f}%")
        
    elif args.image:
        # Test single image
        result = tester.predict_single_image(args.image)
        if result:
            print(f"\n🎯 FINAL VERDICT:")
            print(f"Is this Devanagari '१' (Digit 1)? {'✅ YES' if result['is_digit_one'] else '❌ NO'}")
            print(f"Confidence: {result['confidence']:.2f}%")
    
    else:
        # Interactive mode
        print("\nInteractive Mode - Enter image path to test:")
        print("Type 'quit' to exit, 'test' to create and test sample images")
        
        while True:
            user_input = input("\nEnter image path (or 'quit'/'test'): ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'test':
                test_images = create_test_digit_images()
                tester.test_multiple_images(test_images)
            elif user_input:
                result = tester.predict_single_image(user_input)
                if result:
                    print(f"\n🎯 FINAL VERDICT:")
                    print(f"Is this Devanagari '१' (Digit 1)? {'✅ YES' if result['is_digit_one'] else '❌ NO'}")
                    print(f"Confidence: {result['confidence']:.2f}%")

if __name__ == "__main__":
    main()
