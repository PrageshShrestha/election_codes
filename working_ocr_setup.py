#!/usr/bin/env python3
"""
Working OCR Setup for Devanagari Text
Uses models that actually work with current environment
"""

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def setup_working_ocr():
    """Setup OCR models that actually work"""
    
    print("🔧 Setting up working OCR solutions...")
    
    working_models = {}
    
    # 1. Try EasyOCR in CPU mode (GPU has issues)
    try:
        print("📥 Setting up EasyOCR (CPU mode)...")
        import easyocr
        reader = easyocr.Reader(['hi', 'en'], gpu=False)  # Force CPU mode
        working_models['easyocr'] = reader
        print("✅ EasyOCR setup successful!")
    except Exception as e:
        print(f"❌ EasyOCR failed: {e}")
    
    # 2. Try PaddleOCR with correct parameters
    try:
        print("📥 Setting up PaddleOCR...")
        import paddleocr
        ocr = paddleocr.PaddleOCR(
            use_angle_cls=True, 
            lang='hi',
            use_gpu=False,  # Force CPU mode to avoid GPU issues
            show_log=False
        )
        working_models['paddleocr'] = ocr
        print("✅ PaddleOCR setup successful!")
    except Exception as e:
        print(f"❌ PaddleOCR failed: {e}")
    
    # 3. Try a simple transformer model
    try:
        print("📥 Setting up Vision Transformer...")
        from transformers import pipeline
        vision_pipe = pipeline(
            "image-to-text", 
            model="nlpconnect/vit-gpt2-image-captioning",
            device=-1  # CPU
        )
        working_models['vit_gpt2'] = vision_pipe
        print("✅ ViT-GPT2 setup successful!")
    except Exception as e:
        print(f"❌ ViT-GPT2 failed: {e}")
    
    return working_models

def create_test_image():
    """Create a test image with Devanagari text"""
    
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to find a Devanagari font
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 24)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
            
    except:
        font = ImageFont.load_default()
    
    # Draw sample Devanagari text
    draw.text((10, 10), "धन माया गौतम", fill='black', font=font)
    draw.text((10, 50), "राम कुमार शर्मा", fill='black', font=font)
    
    return img

def test_ocr_models(working_models):
    """Test all working OCR models"""
    
    print("\n🧪 Testing OCR models...")
    
    test_img = create_test_image()
    test_img.save("test.png")
    print("📸 Test image saved as 'test_devanagari.png'")
    
    results = {}
    
    for model_name, model in working_models.items():
        try:
            print(f"\n🔍 Testing {model_name}...")
            
            if model_name == 'easyocr':
                result = model.readtext(np.array(test_img))
                if result:
                    text = ' '.join([item[1] for item in result])
                    results[model_name] = text
                    print(f"📝 EasyOCR Result: {text}")
                else:
                    results[model_name] = "No text detected"
                    print("📝 EasyOCR Result: No text detected")
                    
            elif model_name == 'paddleocr':
                result = model.ocr(np.array(test_img), cls=True)
                if result and result[0]:
                    text = ' '.join([line[1][0] for line in result[0]])
                    results[model_name] = text
                    print(f"📝 PaddleOCR Result: {text}")
                else:
                    results[model_name] = "No text detected"
                    print("📝 PaddleOCR Result: No text detected")
                    
            elif model_name == 'vit_gpt2':
                result = model(test_img)
                if result and len(result) > 0:
                    text = result[0]['generated_text']
                    results[model_name] = text
                    print(f"📝 ViT-GPT2 Result: {text}")
                else:
                    results[model_name] = "No text detected"
                    print("📝 ViT-GPT2 Result: No text detected")
                    
        except Exception as e:
            print(f"❌ {model_name} test failed: {e}")
            results[model_name] = f"Error: {e}"
    
    return results

def create_batch_processor(working_models):
    """Create batch processing script"""
    
    print("\n📝 Creating batch processing script...")
    
    # Choose the best performing model
    best_model = None
    if 'easyocr' in working_models:
        best_model = 'easyocr'
    elif 'paddleocr' in working_models:
        best_model = 'paddleocr'
    elif 'vit_gpt2' in working_models:
        best_model = 'vit_gpt2'
    
    if not best_model:
        print("❌ No working models found!")
        return
    
    print(f"🎯 Using {best_model} for batch processing")
    
    batch_script = f'''#!/usr/bin/env python3
"""
Batch OCR Processing Script for 1M Devanagari Images
Using {best_model} model
"""

import torch
from PIL import Image
import pandas as pd
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Import the chosen OCR library
{"import easyocr" if best_model == "easyocr" else "import paddleocr" if best_model == "paddleocr" else "from transformers import pipeline"}

def setup_ocr():
    """Setup OCR model"""
    {"return easyocr.Reader(['hi', 'en'], gpu=False)" if best_model == "easyocr" else "return paddleocr.PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=False, show_log=False)" if best_model == "paddleocr" else "return pipeline('image-to-text', model='nlpconnect/vit-gpt2-image-captioning', device=-1)"}

def process_single_image(args):
    """Process a single image"""
    image_path, ocr_model = args
    
    try:
        # Open image
        img = Image.open(image_path).convert('RGB')
        
        {"# EasyOCR processing" if best_model == "easyocr" else "# PaddleOCR processing" if best_model == "paddleocr" else "# ViT-GPT2 processing"}
        {"result = ocr_model.readtext(np.array(img))" if best_model == "easyocr" else "result = ocr_model.ocr(np.array(img), cls=True)" if best_model == "paddleocr" else "result = ocr_model(img)"}
        
        {"if result:" if best_model == "easyocr" else "if result and result[0]:" if best_model == "paddleocr" else "if result and len(result) > 0:"}
            {"text = ' '.join([item[1] for item in result])" if best_model == "easyocr" else "text = ' '.join([line[1][0] for line in result[0]])" if best_model == "paddleocr" else "text = result[0]['generated_text']"}
            return os.path.basename(image_path), text.strip()
        else:
            return os.path.basename(image_path), ""
            
    except Exception as e:
        return os.path.basename(image_path), f"ERROR: {{str(e)[:50]}}"

def batch_process_images(image_folder, output_file, max_workers=8):
    """Process all images in folder"""
    
    print(f"🚀 Starting batch processing...")
    print(f"📁 Input folder: {{image_folder}}")
    print(f"📄 Output file: {{output_file}}")
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_folder) 
                          if f.lower().endswith(ext)])
    
    print(f"📊 Found {{len(image_files)}} images to process")
    
    # Setup OCR model
    print("🔧 Setting up OCR model...")
    ocr_model = setup_ocr()
    
    # Process images
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_image, (os.path.join(image_folder, img), ocr_model)) 
                  for img in image_files]
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            results.append(future.result())
    
    # Save results
    df = pd.DataFrame(results, columns=['filename', 'text'])
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    df.to_excel(output_file.replace('.csv', '.xlsx'), index=False)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✅ Processing complete!")
    print(f"⏱️  Time taken: {{processing_time:.2f}} seconds")
    print(f"📈 Speed: {{len(image_files)/processing_time:.2f}} images/second")
    print(f"📄 Results saved to {{output_file}}")
    
    # Show sample results
    print("\\n📝 Sample results:")
    for i, (filename, text) in enumerate(results[:5]):
        print(f"  {{i+1}}. {{filename}}: {{text}}")

if __name__ == "__main__":
    # Configuration
    IMAGE_FOLDER = "/path/to/your/images"  # CHANGE THIS
    OUTPUT_FILE = "ocr_results.csv"
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Image folder not found: {{IMAGE_FOLDER}}")
        print("Please update the IMAGE_FOLDER variable in this script")
    else:
        batch_process_images(IMAGE_FOLDER, OUTPUT_FILE)
'''
    
    with open("batch_ocr_processor.py", "w", encoding="utf-8") as f:
        f.write(batch_script)
    
    print("✅ Batch processing script created: 'batch_ocr_processor.py'")
    print("💡 Edit the IMAGE_FOLDER variable and run: python batch_ocr_processor.py")

if __name__ == "__main__":
    print("🚀 Working OCR Setup")
    print("=" * 50)
    
    # Setup working models
    working_models = setup_working_ocr()
    
    if working_models:
        # Test models
        results = test_ocr_models(working_models)
        
        # Create batch processor
        create_batch_processor(working_models)
        
        print("\n✅ Setup complete!")
        print("📋 Summary of working models:")
        for model_name, result in results.items():
            print(f"  - {model_name}: {result[:50]}...")
            
    else:
        print("❌ No working OCR models found. Please check your environment.")
