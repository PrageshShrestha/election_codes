import easyocr 
import cv2
import numpy as np
import os

# Initialize OCR reader
reader = easyocr.Reader(["hi"], gpu=True)

def apply_filters_and_ocr(image_path):
    """Apply various filters to image and run OCR on each"""
    
    # Read original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    print("=" * 60)
    print("ORIGINAL IMAGE")
    print("=" * 60)
    result = reader.readtext(image_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # Create output directory for filtered images
    os.makedirs("filtered_images", exist_ok=True)
    
    # 1. Grayscale
    print("=" * 60)
    print("1. GRAYSCALE FILTER")
    print("=" * 60)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_path = "filtered_images/gray.jpg"
    cv2.imwrite(gray_path, gray)
    result = reader.readtext(gray_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 2. Binary Threshold
    print("=" * 60)
    print("2. BINARY THRESHOLD (OTSU)")
    print("=" * 60)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_path = "filtered_images/binary.jpg"
    cv2.imwrite(binary_path, binary)
    result = reader.readtext(binary_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 3. Adaptive Threshold
    print("=" * 60)
    print("3. ADAPTIVE THRESHOLD")
    print("=" * 60)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    adaptive_path = "filtered_images/adaptive.jpg"
    cv2.imwrite(adaptive_path, adaptive)
    result = reader.readtext(adaptive_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 4. Gaussian Blur + Threshold
    print("=" * 60)
    print("4. GAUSSIAN BLUR + THRESHOLD")
    print("=" * 60)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, blur_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur_path = "filtered_images/blurred_threshold.jpg"
    cv2.imwrite(blur_path, blur_thresh)
    result = reader.readtext(blur_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 5. Median Blur
    print("=" * 60)
    print("5. MEDIAN BLUR")
    print("=" * 60)
    median = cv2.medianBlur(gray, 3)
    median_path = "filtered_images/median_blur.jpg"
    cv2.imwrite(median_path, median)
    result = reader.readtext(median_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 6. Contrast Enhancement (CLAHE)
    print("=" * 60)
    print("6. CONTRAST ENHANCEMENT (CLAHE)")
    print("=" * 60)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    contrast_path = "filtered_images/contrast_enhanced.jpg"
    cv2.imwrite(contrast_path, contrast)
    result = reader.readtext(contrast_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 7. Morphological Operations (Opening)
    print("=" * 60)
    print("7. MORPHOLOGICAL OPENING")
    print("=" * 60)
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    opening_path = "filtered_images/morphological_opening.jpg"
    cv2.imwrite(opening_path, opening)
    result = reader.readtext(opening_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 8. Edge Detection + Threshold
    print("=" * 60)
    print("8. EDGE DETECTION + THRESHOLD")
    print("=" * 60)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_path = "filtered_images/edge_detection.jpg"
    cv2.imwrite(edges_path, edges_dilated)
    result = reader.readtext(edges_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 9. Noise Reduction (Non-local Means)
    print("=" * 60)
    print("9. NOISE REDUCTION (NON-LOCAL MEANS)")
    print("=" * 60)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    denoised_path = "filtered_images/denoised.jpg"
    cv2.imwrite(denoised_path, denoised)
    result = reader.readtext(denoised_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 10. Sharpening Filter
    print("=" * 60)
    print("10. SHARPENING FILTER")
    print("=" * 60)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    sharpened_path = "filtered_images/sharpened.jpg"
    cv2.imwrite(sharpened_path, sharpened)
    result = reader.readtext(sharpened_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 11. Inverted Image
    print("=" * 60)
    print("11. INVERTED IMAGE")
    print("=" * 60)
    inverted = cv2.bitwise_not(gray)
    inverted_path = "filtered_images/inverted.jpg"
    cv2.imwrite(inverted_path, inverted)
    result = reader.readtext(inverted_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()
    
    # 12. Resized Image (2x)
    print("=" * 60)
    print("12. RESIZED IMAGE (2X)")
    print("=" * 60)
    height, width = gray.shape
    resized = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    resized_path = "filtered_images/resized_2x.jpg"
    cv2.imwrite(resized_path, resized)
    result = reader.readtext(resized_path, text_threshold=0.65)
    texts = [detection[1] for detection in result]
    print("Detected text:", " ".join(texts))
    print("Confidence scores:", [f"{detection[2]:.2f}" for detection in result])
    print()

if __name__ == "__main__":
    image_path = "test.png"
    if os.path.exists(image_path):
        apply_filters_and_ocr(image_path)
    else:
        print(f"Error: Image file {image_path} not found!")