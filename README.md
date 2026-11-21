# tinyinfer

An LLM inference engine built from scratch in plain PyTorch, to understand how engines like vLLM actually work.

No `transformers`, no vLLM. Every component is written by hand, tested against the official implementation, and explained in [beginner-friendly notes](notes/README.md).

- **Model:** [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Hardware:** an 8 GB M1 MacBook

## Progress

**The model, from scratch**
- [x] RMSNorm
- [ ] Rotary position embeddings (RoPE)
- [ ] Attention (grouped-query)
- [ ] MLP (SwiGLU)
- [ ] Full forward pass, matching Hugging Face's output

**The engine**
- [ ] Generation loop and sampling
- [ ] KV cache
- [ ] Benchmarks: time to first token, time per token, throughput
- [ ] Batching: static and continuous
- [ ] OpenAI-compatible streaming server
- [ ] Paged KV cache and prefix caching
- [ ] Quantization
- [ ] Speculative decoding

## Learn along

Start with [01 · How an LLM works](notes/01-how-an-llm-works.md). It covers what's inside a model file, what the 1,024 numbers per token mean, and why the model scores all 151,936 tokens every time it writes one.

## Run it

```bash
uv sync                                    # install dependencies
uv run python scripts/download.py          # download Qwen3-0.6B (1.4 GB)
uv run pytest -v                           # check every component against Hugging Face
uv run python scripts/peek.py "The capital of India is"   # watch one sentence go through the model
```

## Layout

| Folder | What's in it |
|---|---|
| `tinyinfer/` | The engine, plain PyTorch |
| `tests/` | Checks each component against Hugging Face's implementation (used only as a reference) |
| `scripts/` | Tools: download weights, inspect the model file, peek at a forward pass |
| `notes/` | Learning notes, one chapter per topic |
