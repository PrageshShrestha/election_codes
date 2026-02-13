#!/usr/bin/env python3
"""
Client script to test the OCR server
"""

import base64
import requests
import json
import time
from pathlib import Path

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_single_image(server_url: str = "http://localhost:8000", image_path: str = "test.png"):
    """Test OCR on a single image"""
    print(f"Testing single image: {image_path}")
    
    try:
        # Encode image
        image_base64 = encode_image_to_base64(image_path)
        
        # Prepare request
        payload = {
            "image_base64": image_base64,
            "prompt": "Extract the text from this image exactly as it appears."
        }
        
        # Send request
        start_time = time.time()
        response = requests.post(f"{server_url}/ocr", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success (took {end_time - start_time:.2f}s)")
            print(f"Extracted text: {result['text']}")
            return result['text']
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return None

def test_batch_images(server_url: str = "http://localhost:8000", image_paths: list = None):
    """Test OCR on multiple images"""
    if image_paths is None:
        image_paths = ["test.png"]
    
    print(f"Testing batch processing with {len(image_paths)} images")
    
    try:
        # Encode all images
        images_base64 = []
        for image_path in image_paths:
            if Path(image_path).exists():
                images_base64.append(encode_image_to_base64(image_path))
            else:
                print(f"⚠️  Image not found: {image_path}")
        
        if not images_base64:
            print("❌ No valid images found")
            return
        
        # Prepare request
        payload = {
            "images_base64": images_base64,
            "prompt": "Extract the text from this image exactly as it appears."
        }
        
        # Send request
        start_time = time.time()
        response = requests.post(f"{server_url}/ocr/batch", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch success (took {end_time - start_time:.2f}s)")
            print(f"Success rate: {result['success_count']}/{result['total_count']}")
            
            for i, res in enumerate(result['results']):
                print(f"\n--- Image {i+1} ---")
                if res['success']:
                    print(f"Text: {res['text']}")
                else:
                    print(f"Error: {res['error']}")
            
            return result['results']
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return None

def test_file_endpoint(server_url: str = "http://localhost:8000", image_path: str = "test.png"):
    """Test OCR using file path endpoint"""
    print(f"Testing file endpoint: {image_path}")
    
    try:
        payload = {
            "image_path": image_path,
            "prompt": "Extract the text from this image exactly as it appears."
        }
        
        start_time = time.time()
        response = requests.post(f"{server_url}/ocr/file", params=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ File endpoint success (took {end_time - start_time:.2f}s)")
            print(f"Extracted text: {result['text']}")
            return result['text']
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return None

def check_server_health(server_url: str = "http://localhost:8000"):
    """Check if server is running and model is loaded"""
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server is healthy")
            print(f"Model loaded: {health.get('model_loaded', False)}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {str(e)}")
        return False

def main():
    server_url = "http://localhost:8000"
    
    print("🚀 OCR Client Test Suite")
    print("=" * 50)
    
    # Check server health
    if not check_server_health(server_url):
        print("❌ Server is not running. Please start the server first:")
        print("python ocr_server.py")
        return
    
    print("\n" + "=" * 50)
    
    # Test 1: Single image
    print("\n📝 Test 1: Single Image OCR")
    test_single_image(server_url, "test.png")
    
    print("\n" + "=" * 50)
    
    # Test 2: File endpoint
    print("\n📁 Test 2: File Endpoint OCR")
    test_file_endpoint(server_url, "test.png")
    
    print("\n" + "=" * 50)
    
    # Test 3: Batch processing (if multiple images exist)
    print("\n📚 Test 3: Batch Processing")
    test_images = []
    
    # Look for common test images
    common_images = ["test.png", "test.jpg", "test.jpeg", "sample.png", "sample.jpg"]
    for img in common_images:
        if Path(img).exists():
            test_images.append(img)
    
    if len(test_images) > 1:
        test_batch_images(server_url, test_images)
    else:
        print("Only one test image found, skipping batch test")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()
