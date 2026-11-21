"""Shared test helpers: model paths and loading individual weight tensors."""

from pathlib import Path

import pytest
from safetensors import safe_open

from tinyinfer.config import Qwen3Config

MODEL_DIR = Path(__file__).resolve().parent.parent / "weights" / "qwen3-0.6b"


@pytest.fixture(scope="session")
def config() -> Qwen3Config:
    return Qwen3Config.from_dir(MODEL_DIR)


@pytest.fixture(scope="session")
def load_tensor():
    """Return a function that loads one named tensor from the checkpoint."""
    files = sorted(MODEL_DIR.glob("*.safetensors"))

    def _load(name: str):
        for file in files:
            with safe_open(file, framework="pt") as f:
                if name in f.keys():
                    return f.get_tensor(name)
        raise KeyError(name)

    return _load
