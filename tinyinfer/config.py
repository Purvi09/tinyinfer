"""Model configuration: the hyperparameters from config.json, as a typed object."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Qwen3Config:
    hidden_size: int          # numbers per token vector (1024)
    num_hidden_layers: int    # number of stacked transformer layers (28)
    num_attention_heads: int  # query heads (16)
    num_key_value_heads: int  # key/value heads (8), shared by query heads (GQA)
    head_dim: int             # size of each head (128)
    intermediate_size: int    # MLP's expanded width (3072)
    vocab_size: int           # number of possible tokens (151936)
    rope_theta: float         # base for rotary position encoding (1,000,000)
    rms_norm_eps: float       # tiny number that avoids division by zero (1e-6)
    tie_word_embeddings: bool # lm_head reuses the embedding table

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Qwen3Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        # Keep only the fields we defined above; ignore the rest of the JSON.
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})
