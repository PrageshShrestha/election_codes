from huggingface_hub import snapshot_download, hf_hub_download
import os
from pathlib import Path

model_id = "allenai/olmOCR-2-7B-1025-FP8"
save_dir = "./olmOCR-2-7B-1025-FP8"          # change this path if you want

print(f"Downloading {model_id} → {save_dir}\n")

# Option 1: Download everything (recommended for first time)
# This is the most convenient way — downloads model weights + config + tokenizer + processor
snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False,          # set to False = real files (safer for moving later)
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md"],  # optional
)

print("\nDownload finished!")

# Optional: show disk usage
size_bytes = sum(f.stat().st_size for f in Path(save_dir).rglob("*") if f.is_file())
print(f"Total size on disk: {size_bytes / 1024**3:.2f} GB")