#!/usr/bin/env python3
"""
Persistent OCR Server using GGUF model
Handles multiple image requests without reloading model
"""

import os
import base64
import json
import asyncio
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
from llama_cpp import Llama
import uvicorn

app = FastAPI(title="OCR Server", description="Persistent OCR using GGUF model")

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
model_lock = asyncio.Lock()

class OCRRequest(BaseModel):
    image_base64: str
    prompt: str = "Extract the text from this image exactly as it appears."

class OCRResponse(BaseModel):
    text: str
    success: bool
    error: str = None

class BatchOCRRequest(BaseModel):
    images_base64: list[str]
    prompt: str = "Extract the text from this image exactly as it appears."

class BatchOCRResponse(BaseModel):
    results: list[OCRResponse]
    success_count: int
    total_count: int

async def load_model():
    """Load the GGUF model once at startup"""
    global model
    async with model_lock:
        if model is None:
            print("Loading GGUF model...")
            model_path = "./allenai_olmOCR-2-7B-1025-Q4_K_M.gguf"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            model = Llama(
                model_path=model_path,
                n_ctx=2048,  # Reduced context for memory efficiency
                n_gpu_layers=-1,  # Use GPU for all layers
                verbose=False,
                n_threads=4,
                chat_format="llama-3"  # Use appropriate chat format
            )
            print("Model loaded successfully!")

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup"""
    await load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/ocr", response_model=OCRResponse)
async def extract_text(request: OCRRequest):
    """Extract text from a single image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Save temporarily for llama.cpp (or use direct processing if supported)
        temp_path = "/tmp/temp_ocr_image.png"
        image.save(temp_path)
        
        # Process with model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_path},
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]
        
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.0,
            stop=["</s>", "<|im_end|>"]
        )
        
        # Extract text
        extracted_text = response["choices"][0]["message"]["content"].strip()
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return OCRResponse(
            text=extracted_text,
            success=True
        )
        
    except Exception as e:
        # Clean up temp file on error
        temp_path = "/tmp/temp_ocr_image.png"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return OCRResponse(
            text="",
            success=False,
            error=str(e)
        )

@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def extract_text_batch(request: BatchOCRRequest):
    """Extract text from multiple images"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    success_count = 0
    
    for i, image_base64 in enumerate(request.images_base64):
        try:
            # Process each image
            ocr_request = OCRRequest(
                image_base64=image_base64,
                prompt=request.prompt
            )
            result = await extract_text(ocr_request)
            results.append(result)
            
            if result.success:
                success_count += 1
                
        except Exception as e:
            results.append(OCRResponse(
                text="",
                success=False,
                error=f"Image {i+1}: {str(e)}"
            ))
    
    return BatchOCRResponse(
        results=results,
        success_count=success_count,
        total_count=len(request.images_base64)
    )

@app.post("/ocr/file")
async def extract_text_from_file(image_path: str, prompt: str = "Extract the text from this image exactly as it appears."):
    """Extract text from image file path (for local testing)"""
    try:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        
        # Convert to base64
        image_base64 = encode_image_to_base64(image_path)
        
        # Process
        request = OCRRequest(image_base64=image_base64, prompt=prompt)
        return await extract_text(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting OCR Server...")
    uvicorn.run(
        "ocr_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
