"""Our RMSNorm must produce the same output as Hugging Face's, using the real weights."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from tinyinfer.layers import RMSNorm


def test_rmsnorm_matches_hf(config, load_tensor):
    weight = load_tensor("model.layers.0.input_layernorm.weight").float()

    ours = RMSNorm(config.hidden_size, config.rms_norm_eps)
    ref = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    with torch.no_grad():
        ours.weight.copy_(weight)
        ref.weight.copy_(weight)

    torch.manual_seed(0)
    x = torch.randn(2, 5, config.hidden_size)  # (batch=2, tokens=5, hidden=1024)

    torch.testing.assert_close(ours(x), ref(x))


def test_rmsnorm_output_has_unit_rms(config):
    """Without the weight, every output vector should have RMS ~= 1."""
    norm = RMSNorm(config.hidden_size, config.rms_norm_eps)  # weight = all ones
    x = torch.randn(3, config.hidden_size) * 50  # deliberately huge values
    rms = norm(x).pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(3), atol=1e-4, rtol=0)
