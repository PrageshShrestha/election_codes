#!/usr/bin/env python3
"""
Simple PDF to Images Converter
Converts PDF pages to JPG images with page_0001.jpg format in same location.
"""

import os
import sys
from pathlib import Path

try:
    from pdf2image import convert_from_path
    import PIL
    from PIL import Image
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install: pip install pdf2image pillow")
    sys.exit(1)

def pdf_to_images_same_location(pdf_path, dpi=300, width=9356, height=6612):
    """
    Convert PDF pages to JPG images in the output_images folder with fixed dimensions.
    
    Args:
        pdf_path (str): Path to the PDF file
        dpi (int): Resolution of output images (default: 300)
        width (int): Target width in pixels (default: 9356)
        height (int): Target height in pixels (default: 6612)
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output_images folder
    output_dir = pdf_path.parent / "output_images"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Converting {pdf_path.name} to images...")
    print(f"Output location: {output_dir}")
    print(f"DPI: {dpi}, Format: JPG")
    print(f"Target size: {width}x{height} pixels")
    
    try:
        # Get total page count first
        print("Counting pages...")
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path)
        total_pages = info['Pages']
        print(f"Found {total_pages} pages")
        
        # Convert and save one page at a time
        for page_num in range(1, total_pages + 1):
            print(f"Processing page {page_num}/{total_pages}...", end=" ")
            
            # Convert single page
            images = convert_from_path(
                pdf_path, 
                dpi=dpi, 
                fmt='jpeg',
                first_page=page_num,
                last_page=page_num
            )
            
            # Resize to target dimensions and save
            image = images[0]
            # Resize image to exact dimensions
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
            image_path = output_dir / f"page_{page_num:04d}.jpg"
            resized_image.save(image_path, 'JPEG', quality=95)
            
            print(f"✓ Saved: {image_path.name} ({width}x{height})")
        
        print(f"\n✅ Successfully converted {total_pages} pages to JPG images!")
        print(f"📁 Images saved in: {output_dir}")
        print(f"📏 All images resized to {width}x{height} pixels")
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 pdf_to_images_simple.py <pdf_file>")
        print("Example: python3 pdf_to_images_simple.py document.pdf")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    try:
        pdf_to_images_same_location(pdf_file, dpi=300, width=9356, height=6612)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
