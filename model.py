# download_llama.py
from huggingface_hub import snapshot_download
import os

# Model configuration
model_name = "unsloth/DeepSeek-OCR"
local_dir = "./deepseek-ocr"
HF_TOKEN = "token -hf"

print(f" Downloading {model_name}...")
print(f" Saving to: {local_dir}\n")

try:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        token=HF_TOKEN,
        resume_download=True,
        ignore_patterns=["*.md", "*.txt"]
    )

    print("\n" + "=" * 60)
    print(" DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f" Location: {os.path.abspath(local_dir)}")

    # Verify files
    print("\n Files downloaded:")
    for file in os.listdir(local_dir):
        size_mb = os.path.getsize(os.path.join(local_dir, file)) / (1024 * 1024)
        print(f"   {file} ({size_mb:.1f} MB)")

    print("\n Ready for training!")

except Exception as e:
    print(f" Error: {e}")
    print("Check your internet connection and try again.")
