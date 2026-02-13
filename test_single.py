import requests
import base64
import cv2
import numpy as np

def test_single_image():
    # Test with first available image
    import os
    voter_folder = "voter_info"
    if not os.path.exists(voter_folder):
        print("No voter_info folder")
        return
    
    images = [f for f in os.listdir(voter_folder) if f.endswith('.jpg')]
    if not images:
        print("No images found")
        return
    
    first_image = os.path.join(voter_folder, images[0])
    print(f"Testing with: {first_image}")
    
    # Load and crop a small region
    img = cv2.imread(first_image)
    if img is None:
        print("Could not load image")
        return
    
    # Crop voter_id region
    crop = img[0:107, 0:512]
    
    # Convert to base64
    rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', rgb_image)
    img_b64 = base64.b64encode(buffer).decode()
    
    # Test OCR
    prompt = """You are an OCR transcription engine for Nepali text.
Output ONLY the characters visible.
Do not explain.
Do not translate.
Preserve spacing and line breaks."""
    
    payload = {
        "model": "allenai_olmOCR-2-7B-1025-Q4_K_M.gguf",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }
    
    print("Sending request...")
    try:
        response = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=payload, timeout=60)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"].strip()
            print(f"Result: {text}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_single_image()
