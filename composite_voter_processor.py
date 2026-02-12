#!/usr/bin/env python3
"""
Composite Voter Image Processor
Crops individual voter cards from composite images using grid coordinates
Saves individual voter files and shared extra information
"""

import cv2
import os
import numpy as np
from pathlib import Path
import time

def load_bounding_boxes():
    """Load bounding box coordinates from file"""
    bounding_boxes = {}
    
    # Load column coordinates
    bounding_boxes['columns'] = {
        1: [759, 2652],
        2: [2772, 4665], 
        3: [4773, 6678],
        4: [6789, 8691]
    }
    
    # Load row coordinates
    bounding_boxes['rows'] = {
        1: [732, 1237],
        2: [1254, 1759],
        3: [1776, 2281],
        4: [2292, 2795],
        5: [2818, 3322],
        6: [3333, 3839],
        7: [3849, 4352],
        8: [4374, 4878],
        9: [4893, 5397],
        10: [5408, 5914]
    }
    
    # Load extra bounding box
    bounding_boxes['extra'] = [624, 561, 8000, 700]
    
    return bounding_boxes

def create_output_directories():
    """Create output directories if they don't exist"""
    os.makedirs("voter_info", exist_ok=True)
    os.makedirs("voter_extra", exist_ok=True)
    print("✅ Created output directories")

def crop_voter_from_composite(image, voter_num, row, col, bounding_boxes):
    """Crop individual voter from composite image using grid coordinates"""
    
    # Get coordinates for this voter position
    x_start = bounding_boxes['columns'][col][0]
    y_start = bounding_boxes['rows'][row][0]
    x_end = bounding_boxes['columns'][col][1]
    y_end = bounding_boxes['rows'][row][1]
    
    # Crop the voter region
    voter_crop = image[y_start:y_end, x_start:x_end]
    
    return voter_crop

def crop_extra_from_composite(image, bounding_boxes):
    """Crop shared extra information from composite image"""
    
    x_start, y_start, x_end, y_end = bounding_boxes['extra']
    
    # Crop the extra region
    extra_crop = image[y_start:y_end, x_start:x_end]
    
    return extra_crop

def process_composite_images():
    """Process all composite images in output_images directory"""
    
    # Load bounding boxes
    bounding_boxes = load_bounding_boxes()
    
    # Create output directories
    create_output_directories()
    
    # Get all composite images
    composite_dir = Path("output_images")
    if not composite_dir.exists():
        print(f"❌ Composite images directory not found: {composite_dir}")
        return
    
    composite_files = sorted(composite_dir.glob("*.jpg"))
    print(f"📊 Found {len(composite_files)} composite images to process")
    
    voter_counter = 1
    
    for composite_file in composite_files:
        # Extract page number from filename (e.g., page-1056.jpg -> 1056)
        page_num = composite_file.stem.split('-')[1] if '-' in composite_file.stem else composite_file.stem
        
        print(f"\n🔄 Processing composite image: {composite_file.name} (Page {page_num})")
        start_time = time.time()
        
        # Load composite image
        composite_image = cv2.imread(str(composite_file))
        if composite_image is None:
            print(f"❌ Could not load composite image: {composite_file}")
            continue
        
        # Process all 40 voters in this composite image
        page_voter_counter = 1  # Reset counter for each page
        for row in range(1, 11):  # Rows 1-10
            for col in range(1, 5):   # Columns 1-4
                row_for = f"{row:02d}"
                col_for = f"{col:02d}"
                # Crop individual voter
                voter_crop = crop_voter_from_composite(
                    composite_image, page_num, row, col, bounding_boxes
                )
                
                # Save individual voter
                voter_filename = f"voter_{page_num}_{row_for}_{col_for}.jpg"
                voter_path = os.path.join("voter_info", voter_filename)
                cv2.imwrite(voter_path, voter_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
                
                page_voter_counter += 1
                voter_counter += 1
        
        # Crop and save shared extra information for this page
        extra_crop = crop_extra_from_composite(composite_image, bounding_boxes)
        extra_filename = f"voter_{page_num}.jpg"
        extra_path = os.path.join("voter_extra", extra_filename)
        cv2.imwrite(extra_path, extra_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        print(f"Page {page_num} done")
    
    print(f"\n🎉 Processing complete!")
    print(f"📁 Total voters processed: {voter_counter - 1}")
    print(f"📂 Voter cards saved to: voter_info/")
    print(f"📂 Extra info saved to: voter_extra/")

def main():
    """Main processing function"""
    print("🚀 Composite Voter Image Processor")
    print("=" * 50)
    
    process_composite_images()

if __name__ == "__main__":
    main()
