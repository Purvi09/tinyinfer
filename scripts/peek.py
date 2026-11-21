"""Peek inside the model: watch one sentence travel through Qwen3-0.6B, step by step.

Uses Hugging Face's implementation purely as a demo. Our engine will reproduce this from scratch.
Run: uv run python scripts/peek.py "The capital of India is"
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "weights" / "qwen3-0.6b"
text = sys.argv[1] if len(sys.argv) > 1 else "The capital of India is"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

print(f'STEP 1: text\n  "{text}"\n')

ids = tok.encode(text)
print("STEP 2: tokenizer → token IDs")
for i in ids:
    print(f"  {i:>6}  ←  {tok.decode([i])!r}")
x = torch.tensor([ids])
print(f"  shape: {list(x.shape)}   (1 sentence, {len(ids)} tokens)\n")

with torch.no_grad():
    emb = model.model.embed_tokens(x)
    print("STEP 3: embedding lookup → one vector of 1,024 numbers per token")
    print(f"  shape: {list(emb.shape)}   (1 sentence, {len(ids)} tokens, 1024 numbers each)")
    print(f"  first 6 numbers of ' India': {[round(v, 4) for v in emb[0, 3, :6].tolist()]} …\n")

    out = model(x, output_hidden_states=True)
    print("STEP 4: through 28 layers → still one vector per token, same shape")
    print(f"  shape after layer 28: {list(out.hidden_states[-1].shape)}\n")

    logits = out.logits
    print("STEP 5: LM head → a score for EVERY token in the vocabulary")
    print(f"  shape: {list(logits.shape)}   (1 sentence, {len(ids)} positions, 151,936 scores each)")
    print("  we only need the LAST position's scores to predict what comes next\n")

    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    top = torch.topk(probs, 10)
    print("STEP 6: turn scores into probabilities (softmax) and look at the top 10")
    print(f"  {'token':<14}{'score (logit)':>14}{'probability':>14}")
    for p, i in zip(top.values, top.indices):
        print(f"  {tok.decode([i])!r:<14}{last[i].item():>14.2f}{p.item():>13.1%}")
    worst = torch.argmin(last)
    print(f"  … and the LEAST likely of all 151,936: {tok.decode([worst])!r} ({probs[worst].item():.2e})\n")

    print("STEP 7: pick the top token, append it, repeat (greedy generation)")
    seq = x
    for _ in range(8):
        nxt = model(seq).logits[0, -1].argmax()
        seq = torch.cat([seq, nxt.view(1, 1)], dim=1)
        print(f"  + {tok.decode([nxt])!r:<12} → {tok.decode(seq[0])!r}")
