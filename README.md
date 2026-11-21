# tinyinfer

An LLM inference engine built from scratch, one commit at a time, to learn how inference engines like vLLM work.

Built in public over 14 days. Model: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). Hardware: an 8 GB M1 MacBook.

## Roadmap
- [ ] Day 1: Qwen3 forward pass from scratch, matching Hugging Face's output
- [ ] Day 2: generation loop, sampling, KV cache
- [ ] Day 3: benchmarks (TTFT, TPOT, tokens/sec), prefill vs decode
- [ ] Day 4: static batching
- [ ] Day 5: continuous batching
- [ ] Day 6: OpenAI-compatible streaming server
- [ ] Day 7–8: paged KV cache and paged attention
- [ ] Day 9: prefix caching
- [ ] Day 10: quantization (English vs Hindi quality)
- [ ] Day 11: speculative decoding
- [ ] Day 12: torch.compile / CUDA graphs, comparison with vLLM
- [ ] Day 13–14: profiling, write-up, launch

## Layout
- `tinyinfer/`: the engine. Uses only plain PyTorch, no `transformers`.
- `tests/`: checks our output against Hugging Face's implementation, which is used only as a reference.
- `scripts/`: one-off tools such as weight download and inspection.

## Setup
```bash
uv sync
uv run python scripts/download.py
uv run pytest
```
