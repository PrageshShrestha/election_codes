from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch
import gc

# Clear memory
gc.collect()
torch.cuda.empty_cache()

# Load model and processor
model_path = "./olmOCR-2-7B-1025-FP8"

print("Loading model...")
# Load with minimal memory usage
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="cpu",  # Start with CPU
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path)

# Load image
print("Loading image...")
image = Image.open("test.png").convert("RGB")

# Prepare prompt for OCR
prompt = "Extract the text from this image exactly as it appears."

# Create message format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

# Create text prompt
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Process inputs
print("Processing inputs...")
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
)

# Try to move to GPU if possible
try:
    print("Attempting to move to GPU...")
    model = model.to("cuda")
    inputs = inputs.to("cuda")
    device = "cuda"
    print("Successfully moved to GPU")
except Exception as e:
    print(f"GPU failed, using CPU: {e}")
    device = "cpu"

# Generate
print("Generating text...")
try:
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,  # Very conservative
            do_sample=False,
            use_cache=False,  # Disable cache to save memory
            pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else 0
        )
except Exception as e:
    print(f"Generation failed: {e}")
    print("Trying with even smaller context...")
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=64,  # Even smaller
                do_sample=False,
                use_cache=False,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else 0
            )
    except Exception as e2:
        print(f"Even small generation failed: {e2}")
        exit(1)

# Decode the generated text
generated_text = processor.batch_decode(
    generated_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print("\n" + "="*50)
print("OCR Result:")
print("="*50)
print(generated_text)
print("="*50)

# Clear memory
del model, inputs, generated_ids
gc.collect()
torch.cuda.empty_cache()
