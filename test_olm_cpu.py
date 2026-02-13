from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch
import gc

# Clear memory
gc.collect()

# Load model and processor
model_path = "./olmOCR-2-7B-1025-FP8"

print("Loading model on CPU...")
# Load on CPU only
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.float32  # Use float32 on CPU
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

# Keep on CPU
print("Running on CPU...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else 0
    )

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
