from pathlib import Path
from traceback import print_exc
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
import traceback
import easyocr 
reader = easyocr.Reader(["hi"],gpu = True)

reader2 = easyocr.Reader(["hi"],gpu = True)




import pytesseract
import cv2
import numpy as np


import easyocr
import cv2
import numpy as np
import os
import shutil
from pathlib import Path




def smart_copy(source_path, dest_folder):
    src = Path(source_path)
    dest_dir = Path(dest_folder)
    
    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Start with the original filename
    new_name = src.name
    dest_path = dest_dir / new_name
    counter = 1

    # Loop until we find a filename that doesn't exist
    while dest_path.exists():
        new_name = f"{src.stem}_{counter}{src.suffix}"
        dest_path = dest_dir / new_name
        counter += 1

    shutil.copy2(src, dest_path)
    print(f"Copied to: {dest_path}")

# Example usage:

def crop_then_paste(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    regions = {
        "1": [10, 10, 65, 107],
        "2": [62, 8, 135,110],
        "3": [129, 7, 191,100],
        "4": [182,10,240,100],
        "5": [236,10,303,97],
        "6":[295,10,353,102],
        "7": [343,13,406,101],
        "8": [401,11, 502,98]
    }

    digits = []
    for label, coords in regions.items():
        x1, y1, x2, y2 = coords
        crop = img[y1:y2, x1:x2]
        
        
        
        # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 50, 150)
        # kernel = np.ones((3,3), np.uint8)
        # crop = cv2.dilate(edges, kernel, iterations=1)
        x = cv2.imwrite(f"filtered_images/{label}.png", crop)
        image_path = cv2.imread(f"filtered_images/{label}.png")
        
        result = reader2.readtext(image_path,allowlist = "०१२३४५६७८९")
        try:
            texts = [detection[1][0] for detection in result]
            texts = "".join(texts)
            
        except:
            texts = ""
        if texts in "०१२३४५६७८९":
            digits.append(texts)
    text = "".join(digits)

    return text
    






def get_numbers_only(image_path: str) -> str:
    if not os.path.exists(image_path):
        return ""
    text_threshold = random.random()
    img = cv2.imread(image_path)
    if img is None:
        return ""
    some_new = random.random()
    if 0.1<some_new<0.12:
        return crop_then_paste(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    temp_path = "filtered_images/temp_num.jpg"
    cv2.imwrite(temp_path, inverted)
    allowlist = '०१२३४५६७८९'
    result = reader.readtext(
        temp_path,
        allowlist=allowlist,
        text_threshold=text_threshold
    )
    
    if not result:
        return "01"
    
    
    numbers = "".join(d[1] for d in result)
    while len(numbers) < 7 and some_new>0.6:
        numbers = crop_then_paste(image_path)
        some_new = random.random()
        h = random.randint(1,10)
        if some_new<0.5:
            denoised = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)
            cv2.imwrite(temp_path, denoised)
            allowlist = '०१२३४५६७८९'
            result = reader.readtext(
        temp_path,
        allowlist=allowlist,
        text_threshold=text_threshold
            )
            numbers = "".join(d[1] for d in result if d[1] in allowlist)
    return numbers
    # os.makedirs("filtered_images", exist_ok=True)
    
    # best_numbers = ""
    # best_score = -1.0
    
    # allowlist = '०१२३४५६७८९'
    
    # def try_ocr_numbers(preprocessed_img):




        
    #     nonlocal best_numbers, best_score
    #     temp_path = "filtered_images/temp_num.jpg"
    #     cv2.imwrite(temp_path, preprocessed_img)
        
    #     result = reader.readtext(
    #         temp_path,
    #         allowlist=allowlist,
    #         text_threshold=text_threshold
    #     )
        
    #     if not result:
    #         return
        
    #     score = sum(d[2] for d in result)
    #     numbers = "".join(d[1] for d in result)
    #     print(numbers , score)
    #     if score > best_score:
    #         best_score = score
    #         best_numbers = numbers
    
    # try_ocr_numbers(img)
    # try_ocr_numbers(gray)
    
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # try_ocr_numbers(binary)
    
    # c = random.randint(0,5)
    # adaptive = cv2.adaptiveThreshold(
    #     gray, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY, 9, c
    # )
    # try_ocr_numbers(adaptive)
    
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # contrast = clahe.apply(gray)
    # try_ocr_numbers(contrast)
    
    # kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # sharpened = cv2.filter2D(gray, -1, kernel_sharp)
    # try_ocr_numbers(sharpened)
    
    # h, w = gray.shape
    # resized = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    # try_ocr_numbers(resized)
    # h = random.randint(1,10)
    # denoised = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)
    # try_ocr_numbers(denoised)
    # inverted = cv2.bitwise_not(gray)
    # try_ocr_numbers(inverted)
    # return best_numbers

def preprocess_for_numbers(image_path):
    img=cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    grscale = random.randint(200,255)
    gray = cv2.cvtColor(img,grscale)
    
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    inverted_path = "temp_ocr_img/inverted_num.jpg"
    cv2.imwrite(inverted_path, sharpened)
    
    return inverted_path

# def get_numbers_only(image_path):
   
#     result = reader2.readtext(image_path,allowlist = ['०','१','२''३','४','५','६','७','८','९'])
    
#     texts = [detection[1] for detection in result]
#     texts = "".join(texts)
#     return texts
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
# # def get_text_splitting(image_path):
#     img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
#     dig = random.random()

#     if dig > 0.5:

#         x1,y1,x2,y2,x3,y3 = 0,0,238,103,504,103
#     else:
#         x1,y1,x2,y2,x3,y3 = 0,0,243,103,504,103
#     crop1 = img[y1:y2, x1:x2]
#     crop2 = img[y1:y2,x2:x3]
#     # result1 = reader.readtext(crop1)
#     # result2= reader.readtext(crop2)
    
    
#     # result = result1+result2
#     # texts = [detection[1][0] for detection in result[0]]
#     result = reader2.readtext(img)
#     texts = [detection[1][0] for detection in result]
#     texts = "".join(texts)

#     return texts


def get_text_splitting(image_path):

    # image_path = preprocess_for_numbers(image_path)
    # text = get_numbers_only(image_path)
    return get_numbers_only(image_path)
    
    
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
            return True  # Return False immediately if any other char is found
            
        i += 1
    
    # If the loop finishes, all characters were valid
    return False
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
        "voter_id": [0, 20, 512, 107],
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
            cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
            listed_output.append(save_path)
        else:
            save_path = os.path.join(output_folder, f"{label}.jpg")
            cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            # Extract text
            
            text = get_text(save_path)
            if label == "voter_id":
                if len(text)>2:
                    while len(text) < 7:
                        text = get_text_splitting(save_path)
                        print(text)
                    voter_id_text = text
                else:
                    continue
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
        "ward": [5063, 17, 5280, 139],
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
        text =get_text(save_path, default=default)
        
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
        clear_temp_folder("voter_images")
        clear_temp_folder("temp_ocr_img")
        clear_temp_folder("garbage_collector")

        

# This is the equivalent of the pathlib method above
           
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
                traceback.print_exc()
                #print(f"\nError processing voter {i}: {e}")
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
