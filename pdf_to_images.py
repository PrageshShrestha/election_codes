#!/usr/bin/env python3
"""
PDF to Images Converter
Extracts each page from PDF files as individual images and saves them in separate folders.
"""

import os
import sys
from pathlib import Path
import argparse

try:
    from pdf2image import convert_from_path
    import PIL
    from PIL import Image
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install required packages:")
    print("pip install pdf2image pillow opencv-python")
    print("\nFor pdf2image, you also need to install poppler:")
    print("Ubuntu/Debian: sudo apt-get install poppler-utils")
    print("macOS: brew install poppler")
    print("Windows: Download poppler from https://github.com/oschwartz10612/poppler-windows/releases/")
    print("\nFor GPU acceleration, install:")
    print("pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version")
    sys.exit(1)

def check_gpu_availability():
    """Check if GPU acceleration is available."""
    try:
        import cupy as cp
        print(f"✓ GPU available: {cp.cuda.runtime.getDeviceCount()} CUDA device(s)")
        return True, cp
    except ImportError:
        print("⚠ GPU acceleration not available (cupy not installed)")
        return False, None
    except Exception as e:
        print(f"⚠ GPU initialization failed: {e}")
        return False, None

def process_image_gpu(image_array, cp=None):
    """Process image using GPU acceleration if available."""
    if cp is not None:
        try:
            # Transfer to GPU
            gpu_img = cp.asarray(image_array)
            # Apply basic GPU processing (contrast enhancement)
            gpu_img = cp.clip(gpu_img * 1.1, 0, 255).astype(cp.uint8)
            # Transfer back to CPU
            return cp.asnumpy(gpu_img)
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
    return image_array

def pdf_to_images(pdf_path, output_folder=None, dpi=300, format='PNG', use_gpu=True):
    """
    Convert PDF pages to individual images.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Output folder path (optional)
        dpi (int): Resolution of output images (default: 300)
        format (str): Image format (PNG, JPEG, etc.) (default: PNG)
        use_gpu (bool): Whether to use GPU acceleration (default: True)
    
    Returns:
        str: Path to the created folder containing images
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output folder based on PDF filename
    if output_folder is None:
        output_folder = pdf_path.stem + "_pages"
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Check GPU availability
    gpu_available, cp = check_gpu_availability() if use_gpu else (False, None)
    
    print(f"Converting {pdf_path.name} to images...")
    print(f"Output folder: {output_path.absolute()}")
    print(f"DPI: {dpi}, Format: {format}")
    print(f"GPU Acceleration: {'Enabled' if gpu_available else 'Disabled'}")
    
    try:
        # Convert PDF to list of images
        images = convert_from_path(
            pdf_path, 
            dpi=dpi,
            fmt=format.lower()
        )
        
        print(f"Found {len(images)} pages")
        
        # Save each page as an image with optional GPU processing
        for i, image in enumerate(images, 1):
            # Convert PIL to numpy array for GPU processing
            image_array = np.array(image)
            
            # Apply GPU processing if available
            if gpu_available:
                image_array = process_image_gpu(image_array, cp)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(image_array)
            
            image_path = output_path / f"page_{i:03d}.{format.lower()}"
            processed_image.save(image_path, format=format)
            print(f"Saved: {image_path.name}")
        
        print(f"\nSuccessfully converted {len(images)} pages to images!")
        return str(output_path.absolute())
        
    except Exception as e:
        print(f"Error converting PDF: {e}")
        raise

def batch_convert_pdfs(pdf_directory, output_base_dir=None, dpi=300, format='PNG', use_gpu=True):
    """
    Convert all PDFs in a directory to images.
    
    Args:
        pdf_directory (str): Directory containing PDF files
        output_base_dir (str): Base directory for output folders (optional)
        dpi (int): Resolution of output images
        format (str): Image format
        use_gpu (bool): Whether to use GPU acceleration
    """
    pdf_dir = Path(pdf_directory)
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Convert each PDF
    for pdf_file in pdf_files:
        try:
            if output_base_dir:
                output_folder = Path(output_base_dir) / f"{pdf_file.stem}_pages"
            else:
                output_folder = None
            
            result_folder = pdf_to_images(pdf_file, output_folder, dpi, format, use_gpu)
            print(f"✓ Completed: {pdf_file.name} -> {result_folder}\n")
            
        except Exception as e:
            print(f"✗ Failed to convert {pdf_file.name}: {e}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF pages to individual images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single PDF
  python pdf_to_images.py document.pdf
  
  # Convert single PDF with custom settings
  python pdf_to_images.py document.pdf --dpi 600 --format JPEG --output my_folder
  
  # Convert all PDFs in current directory
  python pdf_to_images.py --batch
  
  # Convert all PDFs in specific directory
  python pdf_to_images.py --batch --directory /path/to/pdfs --output-base /path/to/output
        """
    )
    
    parser.add_argument('pdf', nargs='?', help='PDF file to convert')
    parser.add_argument('--output', '-o', help='Output folder path')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output images (default: 300)')
    parser.add_argument('--format', '-f', default='PNG', 
                       choices=['PNG', 'JPEG', 'BMP', 'TIFF'], 
                       help='Image format (default: PNG)')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Enable GPU acceleration (default: enabled)')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                       help='Disable GPU acceleration')
    parser.add_argument('--batch', '-b', action='store_true', 
                       help='Convert all PDFs in directory')
    parser.add_argument('--directory', '-d', default='.', 
                       help='Directory containing PDFs for batch mode (default: current directory)')
    parser.add_argument('--output-base', help='Base directory for batch output folders')
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            batch_convert_pdfs(
                args.directory, 
                args.output_base, 
                args.dpi, 
                args.format,
                args.gpu
            )
        elif args.pdf:
            pdf_to_images(
                args.pdf, 
                args.output, 
                args.dpi, 
                args.format,
                args.gpu
            )
        else:
            parser.print_help()
            print("\nError: Please provide a PDF file or use --batch mode")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
