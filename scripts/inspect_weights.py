"""Print what's inside the model: config, tensor names and shapes, and parameter count.

Run: uv run python scripts/inspect_weights.py
"""

import json
from pathlib import Path

from safetensors import safe_open

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "qwen3-0.6b"

config = json.loads((WEIGHTS / "config.json").read_text())
print("=== config.json (the architecture's hyperparameters) ===")
for key in [
    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "intermediate_size", "vocab_size", "rope_theta", "rms_norm_eps",
    "tie_word_embeddings", "max_position_embeddings", "torch_dtype",
]:
    print(f"{key:>26}: {config.get(key)}")

total = 0
print("\n=== tensors (layer 0 shown; layers 1..N are identical) ===")
for file in sorted(WEIGHTS.glob("*.safetensors")):
    with safe_open(file, framework="pt") as f:
        for name in f.keys():
            t = f.get_slice(name)
            shape = t.get_shape()
            n = 1
            for d in shape:
                n *= d
            total += n
            if ".layers." not in name or ".layers.0." in name:
                print(f"{name:<50} {str(shape):<16} {t.get_dtype()}")

print(f"\nTotal parameters: {total / 1e6:.1f}M  ({total * 2 / 1e9:.2f} GB in bf16)")
