from vllm import LLM, SamplingParams
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
llm = LLM(
    model="./olmOCR-2-7B-1025-FP8",       # or "allenai/olmOCR-2-7B-1025-FP8"
    dtype="fp8",                          # or auto — vLLM detects FP8
    gpu_memory_utilization=0.88,          # tune down to 0.80–0.85 if needed
    max_model_len=200,                   # or lower if you don't need long output
    enforce_eager=True,                   # sometimes helps memory on consumer GPUs
)

# For single image → text
prompt = "<image>\nExtract the text from this image exactly as it appears."
# Note: vLLM uses slightly different chat template / image token handling
# Check allenai/olmocr repo or HF model card for exact prompt format

outputs = llm.generate(
    prompt,
    SamplingParams(temperature=0.0, max_tokens=512),
    image="test.png"   # vLLM >= certain version supports PIL / path directly
)
print(outputs[0].outputs[0].text)