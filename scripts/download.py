"""Download Qwen3-0.6B weights, config and tokenizer into ./weights/qwen3-0.6b."""

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-0.6B"
OUT_DIR = Path(__file__).resolve().parent.parent / "weights" / "qwen3-0.6b"

if __name__ == "__main__":
    path = snapshot_download(
        MODEL_ID,
        local_dir=OUT_DIR,
        # Only what we need: weights, config, tokenizer.
        allow_patterns=["*.safetensors", "config.json", "tokenizer*", "generation_config.json"],
    )
    print(f"Downloaded {MODEL_ID} to {path}")
