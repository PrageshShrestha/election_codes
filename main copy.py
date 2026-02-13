from pathlib import Path
from PIL import Image
import time
import cv2
import os
import subprocess
import csv

# Global model variables to avoid reloading
main_model = "/home/pragesh-shrestha/Desktop/nishant_sir/allenai_olmOCR-2-7B-1025-Q4_K_M.gguf"
mmproj_file = "/home/pragesh-shrestha/Desktop/nishant_sir/mmproj-allenai_olmOCR-2-7B-1025-f16.gguf"

# Check if model files exist
if not os.path.exists(main_model):
    print(f"Model file not found: {main_model}")
    exit(1)
if not os.path.exists(mmproj_file):
    print(f"MMProj file not found: {mmproj_file}")
    exit(1)

print("Model files found")

def get_text(image_path , default = 0):
    """Extract text from image using GGUF model"""
    print(f"Processing image: {os.path.basename(image_path)}")
    if default == 0:
        prompt_file = """
    You are an OCR transcription engine for Nepali text.

    Your task is strict character transcription.

    IMPORTANT LANGUAGE SPECIFICATION:
    - This text contains NEPALI language (Devanagari script)
    - Use NEPALI numbers: १, २, ३, ४, ५, ६, ७, ८, ९, ०
    - DO NOT confuse with Bangla script
    - Preserve exact Nepali characters and diacritics

    Rules you MUST follow:
    - Output only the characters visible in the image.
    - Preserve exact order.
    - Preserve spaces and line breaks.
    - Do NOT translate.
    - Do NOT explain.
    - Do NOT identify language.
    - Do NOT add quotes.
    - Do NOT add punctuation that is not visible.
    - Do NOT add headings.
    - Do NOT say "The text is".
    - Do NOT describe the image.
    - If unsure about a character, copy it as seen.

    Your response must contain ONLY the raw Nepali text.

    <END_OF_INSTRUCTIONS>
    """
    else:
        prompt_file = """
        
        
        You are an OCR transcription engine for Nepali numbers.

    Your task is strict devnagari transcription.

    IMPORTANT LANGUAGE SPECIFICATION:
    - This numbers contains NEPALI language (Devanagari script)
    - Use NEPALI numbers: १, २, ३, ४, ५, ६, ७, ८, ९, ०
    -dont confuse 1 nepali with 9 of nepali ie : १ vs ९(accuracy should be highest)
    - DO NOT confuse with bangla script.
    - Preserve exact Nepali characters and diacritics

    Rules you MUST follow:
    - Output only the characters visible in the image.
    - Preserve exact order.
    - Preserve spaces and line breaks.
    - Do NOT translate.
    - Do NOT explain.
    - Do NOT identify language.
    - Do NOT add quotes.
    - Do NOT add punctuation that is not visible.
    - Do NOT add headings.
    - Do NOT say "The text is".
    - Do NOT describe the image.
    - If unsure about a character, copy it as seen.

    Your response must contain ONLY the raw Nepali numbers.

    <END_OF_INSTRUCTIONS>"""

    # Build CLI command with maximum GPU optimization
    cmd = [
        "./llama.cpp/build/bin/llama-mtmd-cli",
        "-m", main_model,
        "--mmproj", mmproj_file,
        "--image", str(image_path),
        "-p", prompt_file,
        "--temp", "0.1",
        "-n", "512",
        "-c", "4096",
        "--gpu-layers", "999",  # Offload ALL possible layers to GPU
        "--mmproj-offload",     # Enable GPU offloading for multimodal projector
        "--ctx-size", "4096",   # Explicit context size
        "--batch-size", "512",   # Larger batch for better GPU utilization
        "--threads", "8",        # Optimal thread count for GPU processing
        "--no-mmap",            # Disable memory mapping for faster GPU access
        "--numa", "distribute"   # Optimize NUMA for GPU workloads
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # Reduced timeout to 30s
        if result.returncode == 0:
            text = result.stdout.strip()
            print(f"Extracted: {text[:50]}...")
            return text
        else:
            print(f"OCR failed: {result.stderr}")
            return ""
    except subprocess.TimeoutExpired:
        print(f"OCR timeout for {os.path.basename(image_path)}")
        return ""
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        return ""

def parse_actual(image_path):
    """Parse voter_info image and extract text from all regions (optimized)"""
    print(f"Processing voter info: {os.path.basename(image_path)}")
    
    # Load image with optimized flags
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []

    # Create output directories once
    output_folder = "temp_ocr_img"
    voter_images_folder = "voter_images"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(voter_images_folder, exist_ok=True)

    # Optimized regions dictionary
    regions = {
        "voter_id": [0, 0, 512, 107],
        "name": [8, 114, 843, 240],
        "age_gender": [0, 251, 623, 375],
        "parent_name": [0, 374, 1400, 488],
        "spouse": [710, 251, 1458, 374],
        "picture": [1466, 3, 1899, 415]
    }
    
    voter_id_text = "unknown"
    listed_output = []
    
    # Pre-allocate list for better performance
    for label, coords in regions.items():
        x1, y1, x2, y2 = coords
        crop = img[y1:y2, x1:x2]
        
        if label == "picture":
            # Optimized image saving
            save_path = os.path.join(voter_images_folder, f"{voter_id_text}.jpg")
            cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            listed_output.append(save_path)
        else:
            save_path = os.path.join(output_folder, f"{label}.jpg")
            cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Extract text
            text = get_text(save_path)
            listed_output.append(text)
            
            if label == "voter_id":
                voter_id_text = text
    
    return listed_output

def parse_extra(image_path):
    """Parse voter_extra image and extract text from all regions (optimized)"""
    print(f"Processing voter extra: {os.path.basename(image_path)}")
    
    # Load image with optimized flags
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return [""] * 3

    # Create output directory once
    output_folder = "temp_ocr_img"
    os.makedirs(output_folder, exist_ok=True)

    # Optimized regions dictionary
    regions = {
        "municipality": [3394, 11, 4800, 139],
        "ward": [5060, 17, 5280, 139],
        "booth": [5752, 11, 7354, 139]
    }
    
    listed_output = []
    for label, coords in regions.items():
        x1, y1, x2, y2 = coords
        crop = img[y1:y2, x1:x2]
        
        save_path = os.path.join(output_folder, f"extra_{label}.jpg")
        cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Extract text with optimized default
        default = 1 if label == "ward" else 0
        text = get_text(save_path, default=default)
        listed_output.append(text)
    
    return listed_output

def clear_temp_folder(folder_path):
    """Clear temporary folder"""
    path = Path(folder_path)
    if path.exists():
        for file in path.glob("*"):
            if file.is_file():
                file.unlink()
        print(f"Cleared temp folder: {folder_path}")

def main():
    """Main processing function (optimized for maximum GPU performance)"""
    print("Starting Voter OCR Processing - MAX GPU MODE")
    print("=" * 60)
    
    folder = Path("voter_info")
    if not folder.exists():
        print(f"Voter info folder not found: {folder}")
        return
    
    # Setup CSV file with optimized buffering
    with open("output.csv", "w", newline='', encoding='utf-8', buffering=8192) as csvfile:
        fieldnames = ["voter_id", "name", "age_gender", "parent_name", "spouse", 
                   "picture", "municipality", "ward", "booth"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()
        
        # Process all images (no limit for maximum throughput)
        image_files = sorted(list(folder.glob("*.jpg")))
        total_files = len(image_files)
        print(f"Found {total_files} images to process")
        
        if total_files == 0:
            print("No images found to process!")
            return
        
        temp_extra_name = ''
        processed_count = 0
        total_time = 0
        
        for i, image_path in enumerate(image_files, 1):
            # Create loading bar
            progress = i / total_files
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            percent = f"{progress*100:.1f}%"
            
            print(f"\r[{bar}] {percent} ({i}/{total_files}) Processing: {image_path.name}", end="", flush=True)
            start = time.time()
            
            try:
                # Get corresponding extra image (optimized caching)
                clear_temp_folder("temp_ocr_img")
                name_only = image_path.name
                img1 = image_path 
                img2_name = f"{name_only[:10]}.jpg"
                
                if temp_extra_name != img2_name:
                    img2 = Path(f"voter_extra/{img2_name}")
                    var2 = parse_extra(img2) 
                    temp_extra_name = img2_name
                
                
                # Process main image
                var1 = parse_actual(img1)
                
                # Ensure correct list sizes
                
                
                # Combine and map to fields
                combined_list = var1 + var2
                row_dict = dict(zip(fieldnames, combined_list))
                
                # Optimized CSV writing
                cleaned_row = {}
                for key, value in row_dict.items():
                    if isinstance(value, str):
                        cleaned_value = value.replace('\n', ' ').replace(',', '|').replace('\r', ' ')
                    else:
                        cleaned_value = str(value) if value is not None else ""
                    cleaned_row[key] = cleaned_value
                
                writer.writerow(cleaned_row)
                csvfile.flush()
                processed_count += 1
                
                # Performance metrics
                end = time.time()
                duration_sec = end - start
                total_time += duration_sec
                
                # Show progress with timing every 10 images
                if i % 10 == 0 or i == total_files:
                    avg_time = total_time / processed_count
                    eta = avg_time * (total_files - processed_count)
                    print(f"\nProgress: {i}/{total_files} | Avg time: {avg_time:.2f}s | ETA: {eta/60:.1f}min")
                
            except Exception as e:
                print(f"\nError processing voter {i}: {e}")
                # Write empty row to maintain CSV structure
                empty_row = {field: "" for field in fieldnames}
                writer.writerow(empty_row)
                csvfile.flush()
            
            # Optimized cleanup (only every 5 iterations)
            
            
    
    print(f"\n\nPROCESSING COMPLETE!")
    print(f"Processed: {processed_count}/{total_files} voters")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Results saved to output.csv")
    print(f"GPU acceleration: MAXIMUM")

if __name__ == "__main__":
    main()
