#!/usr/bin/env python3
"""
Batch OCR processor for handling 1000+ images
Uses the persistent OCR server for efficient processing
"""

import os
import base64
import requests
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import argparse

class BatchOCRProcessor:
    def __init__(self, server_url: str = "http://localhost:8000", max_workers: int = 5):
        self.server_url = server_url
        self.max_workers = max_workers
        self.session = requests.Session()
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def check_server_health(self) -> bool:
        """Check if server is running and model is loaded"""
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                health = response.json()
                return health.get('model_loaded', False)
            return False
        except Exception:
            return False
    
    def process_single_image(self, image_path: str, prompt: str = "Extract the text from this image exactly as it appears.") -> Tuple[str, bool, str]:
        """Process a single image and return (text, success, error)"""
        try:
            image_base64 = self.encode_image_to_base64(image_path)
            
            payload = {
                "image_base64": image_base64,
                "prompt": prompt
            }
            
            response = self.session.post(f"{self.server_url}/ocr", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return (result['text'], result['success'], result.get('error', ''))
            else:
                return ("", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            return ("", False, str(e))
    
    def process_batch_chunk(self, image_paths: List[str], prompt: str = "Extract the text from this image exactly as it appears.") -> List[Dict]:
        """Process a chunk of images using batch endpoint"""
        try:
            images_base64 = []
            valid_paths = []
            
            for image_path in image_paths:
                if Path(image_path).exists():
                    images_base64.append(self.encode_image_to_base64(image_path))
                    valid_paths.append(image_path)
            
            if not images_base64:
                return []
            
            payload = {
                "images_base64": images_base64,
                "prompt": prompt
            }
            
            response = self.session.post(f"{self.server_url}/ocr/batch", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                # Combine results with original paths
                combined_results = []
                for i, (path, res) in enumerate(zip(valid_paths, result['results'])):
                    combined_results.append({
                        'path': path,
                        'text': res['text'],
                        'success': res['success'],
                        'error': res.get('error', '')
                    })
                return combined_results
            else:
                # Fallback to individual processing
                return self._process_individually(image_paths, prompt)
                
        except Exception as e:
            print(f"Batch processing failed, falling back to individual: {str(e)}")
            return self._process_individually(image_paths, prompt)
    
    def _process_individually(self, image_paths: List[str], prompt: str) -> List[Dict]:
        """Fallback: process images individually"""
        results = []
        for image_path in image_paths:
            text, success, error = self.process_single_image(image_path, prompt)
            results.append({
                'path': image_path,
                'text': text,
                'success': success,
                'error': error
            })
        return results
    
    def process_directory(self, directory: str, output_file: str = None, 
                         file_extensions: List[str] = None, 
                         prompt: str = "Extract the text from this image exactly as it appears.") -> Dict:
        """Process all images in a directory"""
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        
        # Find all image files
        image_paths = []
        for ext in file_extensions:
            image_paths.extend(Path(directory).glob(f"*{ext}"))
            image_paths.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            return {'error': f'No images found in {directory}'}
        
        print(f"Found {len(image_paths)} images to process")
        
        return self.process_image_list(image_paths, output_file, prompt)
    
    def process_image_list(self, image_paths: List[str], output_file: str = None,
                          prompt: str = "Extract the text from this image exactly as it appears.") -> Dict:
        """Process a list of image paths"""
        if not self.check_server_health():
            return {'error': 'OCR server is not running or model not loaded'}
        
        start_time = time.time()
        results = []
        processed_count = 0
        success_count = 0
        
        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_image, path, prompt): path 
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    text, success, error = future.result()
                    results.append({
                        'path': path,
                        'text': text,
                        'success': success,
                        'error': error
                    })
                    
                    processed_count += 1
                    if success:
                        success_count += 1
                    
                    # Progress update
                    if processed_count % 10 == 0 or processed_count == len(image_paths):
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        print(f"Processed: {processed_count}/{len(image_paths)} | Success: {success_count} | Rate: {rate:.2f} img/s")
                
                except Exception as e:
                    results.append({
                        'path': path,
                        'text': '',
                        'success': False,
                        'error': str(e)
                    })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        summary = {
            'total_images': len(image_paths),
            'processed_images': processed_count,
            'successful_extractions': success_count,
            'failed_extractions': processed_count - success_count,
            'total_time_seconds': total_time,
            'average_time_per_image': total_time / processed_count if processed_count > 0 else 0,
            'success_rate': success_count / processed_count if processed_count > 0 else 0,
            'results': results
        }
        
        # Save results if output file specified
        if output_file:
            self.save_results(summary, output_file)
        
        return summary
    
    def save_results(self, results: Dict, output_file: str):
        """Save results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save results: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Batch OCR Processor')
    parser.add_argument('--input', '-i', required=True, help='Input directory or text file with image paths')
    parser.add_argument('--output', '-o', default='ocr_results.json', help='Output JSON file')
    parser.add_argument('--server', '-s', default='http://localhost:8000', help='OCR server URL')
    parser.add_argument('--workers', '-w', type=int, default=5, help='Number of parallel workers')
    parser.add_argument('--prompt', '-p', default='Extract the text from this image exactly as it appears.', 
                       help='OCR prompt')
    parser.add_argument('--extensions', '-e', nargs='+', 
                       default=['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'],
                       help='Image file extensions to process')
    
    args = parser.parse_args()
    
    processor = BatchOCRProcessor(args.server, args.workers)
    
    print(f"🚀 Batch OCR Processor")
    print(f"Server: {args.server}")
    print(f"Workers: {args.workers}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("=" * 50)
    
    # Check if input is directory or file
    if os.path.isdir(args.input):
        results = processor.process_directory(
            args.input, args.output, args.extensions, args.prompt
        )
    elif os.path.isfile(args.input):
        # Read image paths from file
        with open(args.input, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        results = processor.process_image_list(
            image_paths, args.output, args.prompt
        )
    else:
        print(f"❌ Input not found: {args.input}")
        return
    
    # Print summary
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print("\n" + "=" * 50)
        print("📊 PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total Images: {results['total_images']}")
        print(f"Processed: {results['processed_images']}")
        print(f"Successful: {results['successful_extractions']}")
        print(f"Failed: {results['failed_extractions']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Total Time: {results['total_time_seconds']:.2f}s")
        print(f"Avg Time/Image: {results['average_time_per_image']:.2f}s")
        print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
