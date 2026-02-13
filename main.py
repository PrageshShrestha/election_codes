from pathlib import Path
from PIL import Image
import time
import cv2
import os
import subprocess
import csv
import threading
import json
import requests
import tempfile
import shutil
import signal
import sys

import easyocr 
reader = easyocr.Reader(["hi"],gpu = True)

import random
en_to_dev_map = str.maketrans('0123456789', '०१२३४५६७८९')

def clean_text(value):
    """Removes all spaces and converts English digits to Devanagari."""
    
    
    # Convert to string to handle any data type
    val_str = str(value)
    
    # Step 1: Remove all spaces
    val_str = val_str.replace(" ", "")
    
    # Step 2: Translate English digits to Devanagari digits
    val_str = val_str.translate(en_to_dev_map)
    
    return val_str
def get_text_splitting(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    dig = random.random()

    if dig > 0.5:

        x1,y1,x2,y2,x3,y3 = 0,0,238,103,504,103
    else:
        x1,y1,x2,y2,x3,y3 = 0,0,243,103,504,103
    crop1 = img[y1:y2, x1:x2]
    crop2 = img[y1:y2,x2:x3]
    result1 = reader.readtext(crop1)
    result2= reader.readtext(crop2)
    
    result = result1+result2
    texts = [detection[1] for detection in result]
    texts = "".join(texts)

    return texts
def get_text(image_path , default = 0):
    result = reader.readtext(image_path)

    texts = [detection[1] for detection in result]
    texts = " ".join(texts)

    return texts
def check_devnagari(text):
    # If text is empty, return False
    if not text:
        return False
    
    i = 0
    # The Unicode range for Devanagari digits is ० (\u0966) to ९ (\u096F)
    while i < len(text):
        char = text[i]
        
        # Check if character is NOT a Devanagari digit AND NOT a space
        if not ('\u0966' <= char <= '\u096F' or char.isspace()):
            return False  # Return False immediately if any other char is found
            
        i += 1
    
    # If the loop finishes, all characters were valid
    return True
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
        "sn":[870,23,1466,131],
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
            if label == "voter_id":
                not_devnagri = True
                if len(text) < 4 :
                    continue
                while not_devnagri:
                    text = clean_text(text)
                    if len(text)>4 and check_devnagari(text):

                        not_devnagri = False
                    else:
                        text = get_text_splitting(save_path)
                        print(text)
                voter_id_text = text
            listed_output.append(text)
            
            
    
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
        "ward": [4800, 17, 5280, 139],
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
    
    
    # Setup CSV file with optimized buffering
    with open("output.csv", "w", newline='', encoding='utf-8', buffering=8192) as csvfile:
        fieldnames = ["voter_id", "name", "age_gender", "parent_name", "spouse", "sn",
                   "picture", "municipality", "ward", "booth"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()
        import easyocr

        reader = easyocr.Reader(['hi'], gpu=True)  # or ['hi', 'ne'] if you want to try Nepali too
        folder = Path("voter_info")
        # Process all images
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
                img2_name = f"{name_only[:15]}.jpg"
                
                if temp_extra_name != img2_name:
                    img2 = Path(f"voter_extra/{img2_name}")
                    var2 = parse_extra(img2) 
                    temp_extra_name = img2_name
                   
                # Process main image
                var1 = parse_actual(img1)
                
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
                
            except Exception as e:
                print(f"\nError processing voter {i}: {e}")
                # Write empty row to maintain CSV structure
                empty_row = {field: "" for field in fieldnames}
                writer.writerow(empty_row)
                csvfile.flush()
        
        # Stop the persistent server
        
        
        print(f"\n\n PERSISTENT SERVER PROCESSING COMPLETE!")
        print(f"Processed: {processed_count}/{total_files} voters")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per voter: {total_time/max(processed_count, 1):.2f} seconds")
        print(f"Throughput: {processed_count/(total_time/60):.1f} voters per minute")
        print(f"Results saved to output.csv")
        
       



if __name__ == "__main__":
    main()
