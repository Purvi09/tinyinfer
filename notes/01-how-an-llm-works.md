# 01 · How an LLM works, and what we are building

> Written for a complete beginner. No ML background needed.
> Read it in order: each part builds on the one before.
> Budget 2–3 hours. Try running the commands as you read.

---

## Part 0: The whole project in one minute

You've used ChatGPT. You type something, and text comes back. Behind that, there are **two separate things**:

| Thing | What it is | Everyday analogy |
|---|---|---|
| **The model** | A big file full of numbers (1.4 GB for ours) that someone else trained | A **song file** (.mp3) |
| **The inference engine** | The program that *runs* the model: reads the numbers, does the math, produces text | The **music player** (Spotify, VLC) |

The same .mp3 can be played by different players. Some start faster, use less battery, or play to many speakers at once. In the same way, the same model can be run by different engines. **vLLM** and **SGLang** are famous engines used by companies to serve millions of users.

**"Inference infra" / "inference engineering"** = building and running those engines so they're **fast** and **cheap**, and can serve **many users at once**.

👉 **The project:** build our own small engine, called **tinyinfer**, from scratch.
👉 **This chapter's job:** make our engine produce **exactly the same numbers** as the official one, so we know our math is correct before we try to make it fast.

---

## Part 1: The words you need

### 1.1 What is "AI model", "LM", "LLM"?

- **Model:** a mathematical function with millions of adjustable numbers inside. You give it an input, it does a lot of math, and gives an output.
- **LM = Language Model:** a model whose job is: **given some text, guess what comes next.**
- **LLM = Large Language Model:** the same thing, just big (hundreds of millions to trillions of numbers). ChatGPT, Claude, Gemini, Llama and Qwen are all LLMs.

### 1.2 Your phone keyboard is a tiny language model

When you type on your phone:

```
"I'll call you ___"      phone suggests:  [ later ]  [ tomorrow ]  [ back ]
```

That's a language model. It looked at your words and guessed the most likely next word.

**An LLM does exactly the same thing**, just much better:
- your phone looks at the last few words → an LLM looks at **thousands** of words
- your phone shows 3 suggestions → an LLM gives a score to **every one of the 151,936 tokens it knows**
- your phone stops after one word → an LLM **picks one, adds it, and repeats**, writing whole paragraphs

> 💡 **Key insight:** an LLM never writes a paragraph at once. It only ever predicts **one next token**. A long answer is that single step repeated hundreds of times.

### 1.3 What is Qwen?

**Qwen** (pronounced "chwen") is a family of open LLMs made by **Alibaba**. "Open" means they published the model file, so anyone can download it and run it. (You can't download ChatGPT's model.)

We use **Qwen3-0.6B**:
- **Qwen3** = version 3 of the Qwen family
- **0.6B** = 0.6 **b**illion (600 million) numbers inside

Why this one? It's small enough to run on your 8 GB MacBook, but it's built exactly like the big models (same parts, just fewer of them). Learn it here, and the same knowledge applies to models 100× bigger.

### 1.4 Training vs inference

| | Training | Inference |
|---|---|---|
| What | **Teaching** the model by adjusting its numbers on huge amounts of text | **Using** the finished model to get predictions |
| Who | Alibaba did this, using thousands of GPUs for weeks | **Us**: we just download the result |
| Analogy | A student studying for years | The student answering exam questions |

**We never train anything.** The numbers are fixed. We only do inference, and our job is to do it **correctly** first, then **fast**.

---

## Part 2: What actually happens, step by step

This is the most important part. Here is the **real output** of running our model on one sentence. You can run it yourself:

```bash
uv run python scripts/peek.py "The capital of India is"
```

(That script uses Hugging Face's official code to show the steps. Our own engine will reproduce every step from scratch.)

### Step 1: You give it text

```
"The capital of India is"
```

### Step 2: The tokenizer chops text into tokens and gives each one an ID

Computers can't do math on letters, so text becomes numbers first.

```
   785  ←  'The'
  6722  ←  ' capital'
   315  ←  ' of'
  6747  ←  ' India'
   374  ←  ' is'
```

- A **token** is a chunk of text: often a whole word, sometimes part of a word (e.g. "unbelievable" might become "un" + "believ" + "able").
- Notice the **spaces**: `' capital'` (with a leading space) is a different token from `'capital'`.
- Qwen knows **151,936** different tokens. This list is called the **vocabulary**. Each token has a fixed ID number. " India" is always 6747.

### Step 3: The embedding looks up a vector for each token

Each ID is now swapped for a list of **1,024 numbers**, called a **vector**. For example, " India" becomes:

```
[0.004, -0.003, -0.0164, 0.0557, -0.0251, -0.0503, … 1,018 more numbers]
```

**Where do these numbers come from?** From a giant table in the model file called the **embedding table**. It has 151,936 rows (one per token), and each row has 1,024 numbers. Token 6747 → take row 6747. That's it. It's a lookup, not a calculation.

**What do the numbers mean?** Training arranged them so similar words get similar vectors. "India" and "Pakistan" end up with vectors closer together than "India" and "banana". You can't read meaning from any single number. The meaning is in the pattern of all 1,024 together.

### Step 4: The vectors pass through 28 layers

Each **layer** reads all the vectors and refines them. It's like a document passing through 28 editors, where each editor improves it a bit:
- early layers handle things like grammar ("is" needs a noun after it)
- later layers handle meaning and facts ("capital" + "India" → New Delhi)

Inside each layer are two main parts:
1. **Attention:** each token **looks at the earlier tokens** to gather context. " is" looks back at " capital" and " India" to understand what's being asked.
2. **MLP** (a "feed-forward network"): processes each token's vector on its own, and is where much of the model's stored knowledge lives.

Plus **RMSNorm** before each part, to keep the numbers stable (Part 6).

After 28 layers, each token still has a vector of 1,024 numbers, but those numbers now carry **context**: the vector for " is" now "knows" the sentence is about India's capital.

### Step 5: The LM head gives a score for every possible next token

This is the step you asked about, so here it is slowly.

**The question the model must answer:** "Which token comes next?"
**The possible answers:** any of the 151,936 tokens in the vocabulary.

So the model does the most straightforward thing: it **gives every single token a score**. High score = "likely next", low score = "unlikely next". That's why there are **151,936 scores**: one for each token it could possibly say. It can't skip any, because the answer could be any of them.

> 🎓 **Analogy: a multiple-choice exam with 151,936 options.** The model doesn't write an answer. It rates every option, and we pick the highest-rated one.

**LM head** = "language model head". It's the final part ("head") of the model that turns the last vector into those scores. These raw scores are called **logits**.

**How does it compute a score?** The LM head has one row of 1,024 numbers for each token (a table shaped like the embedding table). For each token, it multiplies the final vector with that token's row, number by number, and adds it all up. This is called a **dot product**:

```
final vector:     [ 2,  1, -1]
" New"'s row:     [ 3,  1,  0]    →  2×3 + 1×1 + (-1)×0 = 7    ← high score: vectors point the same way
" banana"'s row:  [-1,  0,  2]    →  2×(-1) + 1×0 + (-1)×2 = -4  ← low score: vectors point in different directions
```

A dot product measures **how much two vectors agree**. So the score answers: "How well does this token match what the model thinks should come next?" Done 151,936 times (with 1,024 numbers each instead of 3), that's the LM head.

> 💡 For Qwen3-0.6B, the LM head table is **literally the same table** as the embedding table from Step 3 (this is called **tied embeddings**). The same table is used for "token → vector" at the start and "vector → score for each token" at the end.

### Step 6: Turn scores into probabilities with softmax

Scores like 15.48 aren't easy to read. **Softmax** converts them into percentages that add up to 100%. Higher score → bigger share. Here are the real results:

```
token          score    probability
' located'     15.48      22.9%
' __'          14.30       7.0%
' ______'      14.29       7.0%
' the'         13.49       3.1%
...
least likely of all 151,936: 'körper'   0.0000000000007%
```

**Wait, it didn't say "New Delhi"!** Its top guess is " located", followed by blanks like "____". Why? The model learned from internet text, and sentences like "The capital of India is ____" mostly appear in **quiz worksheets**, while "The capital of India is located in…" is common in articles. **The model doesn't "know" answers. It predicts what text usually comes next.**

Now watch what happens when we add **one example sentence** first:

```bash
uv run python scripts/peek.py "The capital of France is Paris. The capital of India is"
```
```
' New'      65.3%   ← now it's #1 by far
' Delhi'     6.2%
' Ind'       2.7%
' London'    2.6%
```

The example sets up a **pattern** (question → answer), so the most likely continuation becomes the answer. **That's why prompting works.** You're not "asking" the model anything. You're writing text whose most likely continuation is what you want. (Chat apps also wrap your message in a special chat format behind the scenes, for the same reason.)

### Step 7: Pick one, add it, repeat. That's generation.

```
+ ' New'      → "The capital of France is Paris. The capital of India is New"
+ ' Delhi'    → "... India is New Delhi"
+ '.'         → "... India is New Delhi."
+ ' The'      → "... New Delhi. The"
+ ' capital'  → "... New Delhi. The capital"
+ ' of'       → "... The capital of"
+ ' Japan'    → "... The capital of Japan"
```

Picking the highest-scoring token every time is called **greedy decoding**. (In a later chapter we'll add *sampling*: sometimes picking a lower-ranked token on purpose, which makes text more creative.)

**Notice the problem:** to generate 7 tokens, we ran the whole model 7 times, and each time it reprocessed **the entire sentence from the start**. That's very wasteful. Fixing it is the **KV cache**, the first big inference trick, coming in a later chapter.

### The whole picture

```
"The capital of India is"
        │  Step 2: tokenizer
        ▼
[785, 6722, 315, 6747, 374]                         5 token IDs
        │  Step 3: embedding (lookup)
        ▼
5 vectors × 1,024 numbers
        │  Step 4: 28 layers (attention + MLP + RMSNorm)
        ▼
5 vectors × 1,024 numbers  (now full of context)
        │  Step 5: LM head (dot product with every token's row)
        ▼
5 positions × 151,936 scores
        │  take the LAST position's scores → Step 6: softmax
        ▼
" located" 22.9%,  " __" 7.0%, ...
        │  Step 7: pick one, append, repeat
        ▼
"The capital of India is located in the state of..."
```

---

## Part 2B: What ARE the 1,024 numbers?

### Two different numbers, two different jobs

Think of the embedding table as a **dictionary**:

| Number | What it means | In a dictionary |
|---|---|---|
| **151,936** | How many tokens exist (the **rows**) | Number of **entries** |
| **1,024** | How many numbers describe each token (the **columns**) | Length of each entry's **description** |

The table is **151,936 rows × 1,024 columns**. Each row is one token's description, written in numbers instead of words.

### Toy example: describing words with 4 numbers

| Word | is a place? | is food? | is alive? | royalty? |
|---|---|---|---|---|
| India | **0.9** | 0.0 | 0.0 | 0.0 |
| Pakistan | **0.9** | 0.0 | 0.0 | 0.0 |
| banana | 0.0 | **0.9** | 0.1 | 0.0 |
| king | 0.1 | 0.0 | **0.8** | **0.9** |
| queen | 0.1 | 0.0 | **0.8** | **0.9** |

Each word becomes a list of 4 numbers (India = `[0.9, 0, 0, 0]`). **Similar meaning means similar lists.**

The real model does the same thing with **1,024** numbers, with two differences:
1. **Nobody chose the traits.** Training adjusted the numbers automatically. No single column means "is a country". The meaning is spread across all 1,024 together.
2. 1,024 traits capture much more: meaning, grammar, language, tone, and so on.

### Proof: real similarity scores from our model

Comparing " India"'s 1,024 numbers with other tokens (1 = very similar, 0 = unrelated, negative = opposite):

```
' India' vs ' Pakistan'  0.515      ' king' vs ' queen'   0.562
' India' vs ' China'     0.492      ' king' vs ' banana'  0.068
' India' vs ' Delhi'     0.451      ' cat'  vs ' dog'     0.398
' India' vs ' banana'    0.122      ' cat'  vs ' car'     0.188
' India' vs ' the'      -0.077
```

The 12 closest tokens to " India" out of all 151,936:
`' India', 'India', ' india', '印度', ' Indian', 'Indian', ' Indians', ' Pakistan', ' China', …`

`印度` is **Chinese for "India"**. Nobody told the model that. It learned it just from reading text.

### Why 1,024?

It's a **design choice** (`hidden_size`). More numbers mean a richer description, but more memory and slower math. Small models use 1,024; bigger ones use 4,096 or more.

### What happens to our 5 tokens

1. **Tokenizer:** `[785, 6722, 315, 6747, 374]`. Each ID is a **row number**.
2. **Embedding:** copy out those 5 rows to get a small table of **5 × 1,024** (shape `[5, 1024]`). Just a lookup, no math.
3. **28 layers:** attention lets each token look at the others and **mix in context**. At the start, " is" has the same numbers in every sentence. By layer 28, its 1,024 numbers mean something like *"end of a sentence about India's capital; a place name or 'located' comes next."*
4. **LM head:** compares the **last** token's final 1,024 numbers with all 151,936 rows (dot product). The best matches get the highest scores.

> 🎓 **Analogy:** each token starts with a plain **name tag** ("I am 'is'"). After 28 layers, it holds a **detailed note** about its role in *this* sentence. The LM head reads the last note and scores every possible next word.

```
5 IDs → look up 5 rows (5 × 1,024) → 28 layers mix in context (still 5 × 1,024)
      → compare the last row with all 151,936 tokens → 151,936 scores
```

<details><summary><b>Quick check:</b> In the shape <code>[151936, 1024]</code>, which number is "how many words" and which is "how much description per word"?</summary>

151,936 = how many tokens (rows). 1,024 = numbers describing each token (columns).
</details>

---

## Part 3: Tensors and shapes, explained properly

### 3.1 What is a tensor?

A **tensor** is just a box of numbers arranged in a grid. The only question is: **how many directions does the grid go in?**

| Dimensions | Common name | Python-ish example | Shape |
|---|---|---|---|
| 0 | a single number | `7` | `[]` |
| 1 | a list (vector) | `[7, 2, 9]` | `[3]` |
| 2 | a table (matrix) | `[[7, 2, 9], [1, 0, 4]]` | `[2, 3]` |
| 3 | a stack of tables | 2 tables, each 5 rows × 1024 columns | `[2, 5, 1024]` |

**Shape** = the size in each direction, listed from outside to inside.
- `[2, 3]` = 2 rows, each with 3 numbers.
- `[2, 5, 1024]` = 2 things, each with 5 things, each with 1,024 numbers.

**Read a shape like a sentence.** For example, `[1, 5, 1024]` reads as: "**1** sentence, with **5** tokens, each having **1,024** numbers."

### 3.2 Following the shapes through our model

These are the **real shapes** printed by `scripts/peek.py`:

| Step | Shape | Read it as |
|---|---|---|
| Token IDs | `[1, 5]` | 1 sentence, 5 token IDs |
| After embedding | `[1, 5, 1024]` | 1 sentence, 5 tokens, 1,024 numbers each |
| After 28 layers | `[1, 5, 1024]` | same shape, but the numbers now carry context |
| After LM head | `[1, 5, 151936]` | 1 sentence, 5 positions, 151,936 scores each |
| What we use | `[151936]` | just the last position's scores |

**Why does the first dimension say `1`?** That's the **batch**: how many sentences we process at once. Today it's 1. Later we'll process many users' sentences together in one go (*batching*), and that number grows. Running many at once is the main way inference engines save money.

**Why do we get scores for all 5 positions if we only need the last?** The model predicts "what comes next" after **every** position at once: after "The", after "The capital", and so on. That's useful in training, but for generating we only care about the **last** one. In code that's `logits[0, -1]`, which means "sentence 0, last position". (`-1` in Python means "the last one".)

### 3.3 How shapes change: matrix multiplication

Most of what a model does is **matrix multiplication**: a table of numbers times another table. There's one rule to remember:

```
[A, B]  ×  [B, C]  =  [A, C]
     ↑      ↑
   these must match, and they "disappear"
```

The LM head, for example:
```
final vectors      LM head table          scores
[5, 1024]    ×    [1024, 151936]    =    [5, 151936]
```
Each of the 5 vectors is dot-producted with each of the 151,936 token rows. That gives 5 × 151,936 scores.

> 💡 **Weight files store layers as `[out, in]`.** The LM head is stored as `[151936, 1024]`, so the code flips it (*transposes* it) to `[1024, 151936]` before multiplying. You'll see `[out, in]` everywhere in PyTorch.

### 3.4 Shape bugs are the most common bug

If shapes don't line up (e.g. `[5, 1024] × [512, 3072]`), PyTorch crashes with an error. That's actually helpful, since it tells you exactly where the problem is. **When stuck, print the shapes.** That's the number 1 debugging habit in this project.

---

## Part 4: Setting up the project (commit 1)

| Tool or file | What it is | Why we need it |
|---|---|---|
| **git** | Saves every version of your code | Each step is a commit, so you can always go back. Your history shows real, steady work |
| **GitHub** | Stores your git history online | Your public portfolio: [github.com/Purvi09/tinyinfer](https://github.com/Purvi09/tinyinfer) |
| **uv** | Installs Python packages into a private `.venv/` folder for this project only | Keeps this project's packages separate from everything else on your Mac |
| `pyproject.toml` | The list of packages this project needs | Anyone can run `uv sync` and get the same setup |
| `uv.lock` | The *exact* versions installed | Guarantees it works the same on every machine |
| `.gitignore` | Files git must never upload | The 1.4 GB model (anyone can download it), `.venv/`, and your private `posts/` |

**Packages we installed:**
| Package | Job |
|---|---|
| `torch` (PyTorch) | Does all the tensor math, on the CPU or your Mac's GPU (Apple calls its GPU backend **MPS**) |
| `safetensors` | Reads the model file |
| `huggingface_hub` | Downloads models from huggingface.co (the "GitHub of AI models") |
| `transformers` | Hugging Face's official model code. **Only used in tests and demos as the answer key.** Our engine never uses it |
| `pytest` | Runs our tests |

**Folders:**
```
tinyinfer/   ← our engine: the real code we write
tests/       ← checks our engine against the answer key
scripts/     ← helper tools: download, inspect, peek
notes/       ← these notes
weights/     ← the model file (not uploaded to GitHub)
posts/       ← Twitter drafts (not uploaded to GitHub)
```

---

## Part 5: Looking inside the model file (commit 2)

### 5.0 In one sentence

**The model file is 751,632,384 decimal numbers, plus a small label list saying which numbers belong to which part.** No words, no facts written down, no code.

> 🎓 **The knob analogy:** imagine a machine with 751 million knobs. Text goes in, and the knob positions decide what comes out.
> - **Training** = Alibaba fed it trillions of words and kept nudging every knob so its next-token guesses got better (weeks, thousands of GPUs).
> - **The model file** = a **photo of where every knob ended up**.
>
> "The capital of India is New Delhi" isn't written anywhere in the file. It's spread across millions of knob positions, the way your ability to ride a bike isn't stored in your brain as written instructions.

**Numbers vs. recipe:**

| | What it is | Where |
|---|---|---|
| **The numbers** (knob positions) | What training produced | `model.safetensors`, downloaded |
| **The recipe** (what math, in what order) | embedding → (RMSNorm → attention → RMSNorm → MLP) × 28 → LM head | **Our code in `tinyinfer/`** |

> 🧮 The model file is like an **Excel sheet full of numbers with no formulas**. Our engine is the **formulas**. Neither does anything without the other.

**What the raw file looks like, from byte 1:**
```
e0 8a 00 00 00 00 00 00                    ← 8 bytes: "the label list is 35,552 bytes long"
{"lm_head.weight":{"dtype":"BF16",         ← 35,552 bytes: the label list (readable text)
  "shape":[151936,1024],"data_offsets":[0,311164928]}, ...}
83 3b 47 bb 86 bc 64 3d ce bc 4e bd ...    ← 1.4 GB: the numbers, 2 bytes each
```
For example, those bytes `83 3b · 47 bb · 86 bc · 64 3d · …` are the start of " India"'s row, and they decode to `0.004, -0.003, -0.0164, 0.0557, …`, the same numbers we saw in Part 2B.

### 5.1 The files we downloaded

`weights/qwen3-0.6b/` contains:

```
model.safetensors        1.4 GB   ← THE MODEL: all 751 million learned numbers
tokenizer.json           11 MB    ← the vocabulary: 151,936 tokens and their IDs
config.json              726 B    ← settings: 28 layers, 1,024 numbers per token, …
tokenizer_config.json    9.5 KB   ← tokenizer settings
generation_config.json   239 B    ← default generation settings
```

### 5.2 What's inside `model.safetensors`

It's a **binary** file (raw numbers), so you can't open it in an editor. It has two parts:

**A. A small header** (35 KB), like a book's table of contents. For each of the 311 tensors, it lists the name, number format, shape and **where its numbers sit in the file**:
```
model.embed_tokens.weight               BF16  [151936, 1024]  bytes 311,164,928 → 622,329,856
model.layers.0.self_attn.q_proj.weight  BF16  [2048, 1024]    bytes 647,500,288 → 651,694,592
model.norm.weight                       BF16  [1024]          bytes 1,503,262,720 → 1,503,264,768
… 308 more
```

**B. The numbers themselves** (1.4 GB), packed one after another.

To load `model.embed_tokens.weight`, the library looks up its position in the header, jumps to those bytes and arranges them into a `[151936, 1024]` table. (Check: 151,936 × 1,024 × 2 bytes = 311,164,928 bytes, exactly the size of that range.)

> 🎓 **Analogy:** the file is a **warehouse**. The header is the **inventory list** ("embedding table: shelves 311M–622M"). The rest is shelves full of numbers.

So conceptually, the model file is just a **dictionary of named tensors**:
```python
{
  "model.embed_tokens.weight":              tensor of shape [151936, 1024],
  "model.layers.0.self_attn.q_proj.weight": tensor of shape [2048, 1024],
  ... 309 more ...
}
```
(This isn't code from any file. It's a picture of what's inside `model.safetensors`.)

### 5.3 The script that reads it: [scripts/inspect_weights.py](../scripts/inspect_weights.py)

This script **only reads** the model file and prints its inventory. Run:
```bash
uv run python scripts/inspect_weights.py
```

| Code | What it does |
|---|---|
| `json.loads(... "config.json" ...)` | Reads the settings and prints the main ones |
| `safe_open(file, framework="pt")` | Opens `model.safetensors` and reads its **header** |
| `for name in f.keys()` | Goes through all 311 tensor names |
| `f.get_slice(name).get_shape()` | Gets each shape from the header **without loading the numbers**, which is why it's instant |
| `n *= d` | Multiplies the shape's sizes to count the numbers (151,936 × 1,024 = 155.6M) |
| `if ".layers.0." in name …` | Prints only layer 0, since layers 1–27 are identical |
| last line | Prints the total: **751.6M** numbers |

### 5.4 The settings: `config.json`

| Setting | Value | Plain English |
|---|---|---|
| `vocab_size` | 151,936 | Number of different tokens the model knows |
| `hidden_size` | 1,024 | Numbers per token vector |
| `num_hidden_layers` | 28 | How many layers the vectors pass through |
| `num_attention_heads` | 16 | Attention runs as 16 parallel "lookers" (explained below) |
| `num_key_value_heads` | 8 | …but they share 8 sets of keys and values (GQA, explained below) |
| `head_dim` | 128 | Numbers each "looker" works with |
| `intermediate_size` | 3,072 | The MLP temporarily expands 1,024 → 3,072 numbers |
| `torch_dtype` | bfloat16 | Number format the weights are stored in (Part 6.4) |

### 5.5 Every tensor, explained

Layer 0 is shown. Layers 1–27 have exactly the same tensors, just different numbers.

| Tensor | Shape `[out, in]` | What it does |
|---|---|---|
| `embed_tokens` | `[151936, 1024]` | **Step 3's lookup table.** Row = token ID, 1,024 numbers per row |
| `input_layernorm` | `[1024]` | RMSNorm before attention |
| `q_proj` | `[2048, 1024]` | Makes **queries** ("what am I looking for?"): 16 heads × 128 = 2,048 |
| `k_proj` | `[1024, 1024]` | Makes **keys** ("what do I contain?"): **8** heads × 128 = 1,024 |
| `v_proj` | `[1024, 1024]` | Makes **values** ("what info do I pass on?"): 8 heads × 128 |
| `q_norm`, `k_norm` | `[128]` | A small RMSNorm on each head's queries and keys (a Qwen3-specific detail) |
| `o_proj` | `[1024, 2048]` | Combines the 16 heads' outputs back into 1,024 numbers |
| `post_attention_layernorm` | `[1024]` | RMSNorm before the MLP |
| `gate_proj`, `up_proj` | `[3072, 1024]` | MLP expands 1,024 → 3,072 |
| `down_proj` | `[1024, 3072]` | MLP shrinks back to 1,024 |
| `model.norm` | `[1024]` | Final RMSNorm after layer 28 |
| `lm_head` | `[151936, 1024]` | **Step 5: a score for every token** |

### 5.6 Attention heads and GQA (just the idea)

Think of attention as each token **asking a question** and the earlier tokens **answering**:
- **Query (Q):** what the current token is looking for. (" is" asks: "what's the subject here?")
- **Key (K):** what each earlier token offers. (" India" says: "I'm a country name.")
- **Value (V):** the actual information passed along if there's a match.

There are **16 heads**: 16 separate question-askers working in parallel, each looking for something different (grammar, topic, what "it" refers to, …).

**GQA (grouped-query attention):** 16 query heads, but only **8** key/value heads, so every 2 query heads share one K/V head. Why should *you* care? Because during generation, K and V get **saved in memory** for every token (the **KV cache**, coming soon). Fewer K/V heads means **half the memory**, which means more users per GPU.

### 5.7 Puzzle: why 752M numbers when the model is "0.6B"?

`inspect_weights.py` counted **751.6 million** numbers. But the name says 0.6 billion.

- `lm_head` (Step 5) and `embed_tokens` (Step 3) are **the same table** (tied embeddings).
- The file stores it twice anyway.
- 751.6M − 155.6M = **596M ≈ 0.6B** ✅

Where the 596M real numbers live: **MLP 264M** · **Attention 176M** · **Embedding 156M** · Norms 0.07M.

### 5.8 Your first inference calculation: KV cache memory

For every token in a conversation, every layer saves a K and a V vector:

```
28 layers × 2 (K and V) × 8 heads × 128 numbers × 2 bytes each
= 114,688 bytes ≈ 112 KB per token
```

- A 4,000-token chat → 4,000 × 112 KB ≈ **450 MB**, for **one** user, on the **smallest** Qwen3 model.
- 100 users at once → 45 GB.

**This is why inference engineering is mostly about managing memory.** Remember this number. It comes back when we build the KV cache, paged attention and prefix caching.

---

## Part 6: RMSNorm, the first thing we built ourselves (commit 3)

### 6.1 The problem

A token's vector goes through 28 layers, and each layer **adds** its output to the vector. Without control, the numbers can grow huge (1,000,000) or shrink to almost nothing (0.000001), like a microphone getting louder and louder until it's pure noise. Computers handle very big or very small numbers badly, so the math breaks.

### 6.2 The fix: automatic volume control

RMSNorm rescales each vector to a standard "loudness" before each attention and MLP part.

**Recipe for one vector:**
```
1. square every number                    x²
2. take the average                       mean(x²)
3. take the square root  → this is RMS    √mean(x²)        "how loud is this vector?"
4. divide every number by RMS             x / RMS           now loudness = 1
5. multiply by a learned weight           × weight          some positions louder, some quieter
```

**Worked example** with 4 numbers instead of 1,024:
```
x       = [ 2, -2,  2, -2 ]
x²      = [ 4,  4,  4,  4 ]   → average = 4  → RMS = √4 = 2
x / RMS = [ 1, -1,  1, -1 ]   ← loudness is now 1
```
And a loud vector: `[200, -200, 200, -200]` → RMS = 200 → also becomes `[1, -1, 1, -1]`. **Big or small, every vector comes out the same size.**

**`eps`** (0.000001) is added inside the square root, so if a vector is all zeros we don't divide by zero.

**The learned weight** (`input_layernorm.weight`, 1,024 numbers from the file): training discovered that some of the 1,024 positions should be amplified and others muted. We just load those numbers.

### 6.3 The code, line by line: [tinyinfer/layers.py](../tinyinfer/layers.py)

```python
class RMSNorm(nn.Module):
```
`nn.Module` is PyTorch's base class for "a piece of a model". Every component we build will inherit from it.

```python
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
```
Runs once, when the component is created. It makes a slot for the 1,024 learned weights (`nn.Parameter` = "numbers that come from the model file"). They start as all 1s and get overwritten with the real values when we load the file.

```python
    def forward(self, x):
```
`forward` = "what this component computes". When you call `norm(x)`, PyTorch runs `forward(x)`.

```python
        input_dtype = x.dtype
        x = x.float()
```
Remember the original number format (e.g. bfloat16), then switch to float32 for precise math (Part 6.4).

```python
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
```
Recipe steps 1–3 in one line, read from the inside out:
- `x.pow(2)`: square every number
- `.mean(dim=-1, ...)`: average across the **last** dimension, i.e. the 1,024 numbers of each token. Each token gets its own average, so tokens are normalized **separately**.
- `+ self.eps`: add the tiny safety number
- `torch.rsqrt(...)`: **r**eciprocal **sq**uare **r**oo**t** = 1 / √(…). Multiplying by 1/RMS is the same as dividing by RMS, but faster.
- `keepdim=True`: keeps the result shaped `[..., 1]` (one number per token) instead of dropping that dimension, so the next line can line it up with the vectors.

```python
        x = x * rms_inv
```
Step 4: divide by RMS. `x` is `[2, 5, 1024]` and `rms_inv` is `[2, 5, 1]`. PyTorch automatically stretches the `1` to match `1024` (this is called **broadcasting**), so every one of a token's 1,024 numbers is divided by that token's RMS.

```python
        return self.weight * x.to(input_dtype)
```
Switch back to the original format, then step 5: multiply by the learned weights.

### 6.3b The same code, traced with real numbers

Our real `RMSNorm`, run on a 4-number vector (instead of 1,024) with weights `[1, 2, 0.5, 1]`:

```
input x                    [2, -4, 6, -8]              (bfloat16)
1. x.float()               [2, -4, 6, -8]              (float32: same values, more precise)
2. .pow(2)                 [4, 16, 36, 64]             square each
3. .mean(dim=-1)           30                          average → shape [1, 1] thanks to keepdim
4. rsqrt(30 + eps)         0.18257                     = 1/√30 = 1/5.477   (5.477 is the RMS)
5. x * rms_inv             [0.365, -0.730, 1.095, -1.461]   loudness is now exactly 1
6. weight * x              [0.365, -1.461, 0.547, -1.461]   ×1, ×2 (louder), ×0.5 (quieter), ×1
```

**Two moments of a class:**

| Moment | Method | Runs | What happens |
|---|---|---|---|
| Building | `__init__` | once | creates the `weight` slot (all 1s), which is then **filled with numbers from the model file** |
| Using | `forward` | every call to `norm(x)` | steps 1–6 above |

> 💡 **This is how file + code connect:** the file provides `input_layernorm.weight` (the numbers). Our class provides steps 1–6 (the recipe).

**Try it yourself:**
```bash
uv run python
```
```python
import torch
from tinyinfer.layers import RMSNorm
norm = RMSNorm(4)
norm(torch.tensor([2.0, -4.0, 6.0, -8.0]))
norm(torch.tensor([200.0, -400.0, 600.0, -800.0]))   # 100× louder → same output!
```

### 6.4 float32 vs bfloat16: why we switch formats

Computers store decimal numbers in formats with limited precision:

| Format | Memory per number | Precision | Used for |
|---|---|---|---|
| **float32** | 4 bytes | ~7 digits (3.141593) | Sensitive math: norms, softmax |
| **bfloat16** (bf16) | 2 bytes | ~3 digits (3.14) | Storing weights: **half the memory**, faster |

The model is stored in bf16 (1.2 GB instead of 2.4 GB). Memory is the scarcest resource in inference, so this matters a lot. But squaring and averaging 1,024 numbers with only 3 digits of precision introduces errors, and 28 layers later those errors add up. So RMSNorm briefly switches to float32 for its math. The official code does this too, and if we didn't, our output wouldn't match theirs.

### 6.5 The config: [tinyinfer/config.py](../tinyinfer/config.py)

```python
@dataclass
class Qwen3Config:
    hidden_size: int     # 1024
    num_hidden_layers: int   # 28
    ...
```
A **dataclass** is a Python class that just holds named values. Writing `@dataclass` auto-generates the boring setup code for you.

```python
    @classmethod
    def from_dir(cls, model_dir):
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})
```
Reads `config.json`, picks **only the fields we listed above**, and builds the object. Now code can write `config.hidden_size` (with autocomplete, and typos get caught) instead of `raw["hidden_size"]`.

---

## Part 7: Tests, or how we know we're right

**Rule: never trust our own code.** A slightly wrong model still outputs believable-looking numbers, and the bug would only show up days later as weird text, with no idea where it came from.

So for every component we:
1. load the **same real weights** into our version and the official Hugging Face version
2. give both the **same input**
3. check the outputs **match**

### [tests/conftest.py](../tests/conftest.py): shared helpers
- A **fixture** (`@pytest.fixture`) is a helper that pytest hands to tests automatically. If a test function has a parameter called `config`, pytest runs the `config()` fixture and passes its result in.
- `scope="session"` = create it **once** and share it with all tests.
- `load_tensor("model.layers.0.input_layernorm.weight")` reads **one** tensor without loading the whole 1.4 GB file.

### [tests/test_rmsnorm.py](../tests/test_rmsnorm.py)
```python
weight = load_tensor("model.layers.0.input_layernorm.weight").float()   # real weights
ours = RMSNorm(...)        ;  ref = Qwen3RMSNorm(...)                    # ours vs official
ours.weight.copy_(weight)  ;  ref.weight.copy_(weight)                   # same weights in both
torch.manual_seed(0)
x = torch.randn(2, 5, 1024)                                              # same random input
torch.testing.assert_close(ours(x), ref(x))                              # must match
```
- `torch.randn(2, 5, 1024)` = random numbers shaped like 2 sentences × 5 tokens × 1,024.
- `torch.manual_seed(0)` = "random", but the same random numbers every run, so the test is repeatable.
- `torch.no_grad()` = "we're not training", so PyTorch doesn't track extra info. It's faster and uses less memory.
- `assert_close` = the numbers must match within a tiny tolerance. Computer decimals are never perfectly exact, so "close enough" is the correct check.

The second test checks the behavior: vectors 50× too loud still come out with loudness exactly 1.

Run them yourself:
```bash
uv run pytest -v
```

---

## Part 8: Commits so far

| # | Commit message | What it added |
|---|---|---|
| 1 | `Add weight download and inspection scripts for Qwen3-0.6B` | Project setup, download script, inspect script |
| 2 | `Add model config and RMSNorm, tested against Hugging Face` | `config.py`, `layers.py` (RMSNorm), tests |
| 3 | `Add learning notes: how an LLM works` | These notes, plus `scripts/peek.py` |

---

## Part 9: Check yourself

Answer out loud **before** opening each answer. If you can explain it in 2 sentences, you've got it.

<details><summary><b>Q1.</b> What's the difference between a model and an inference engine?</summary>

The model is a file of learned numbers (like an .mp3). The inference engine is the program that runs it (like a music player). We're building an engine.
</details>

<details><summary><b>Q2.</b> What is the ONE thing an LLM does?</summary>

Predicts the next token. A full answer is that step repeated: predict, append, predict, append…
</details>

<details><summary><b>Q3.</b> Why does the LM head output 151,936 scores?</summary>

The next token could be any token in the vocabulary, and there are 151,936 of them. So the model scores every option, like a multiple-choice exam with 151,936 choices, and we pick from the highest-scoring ones.
</details>

<details><summary><b>Q4.</b> How does the LM head compute one token's score?</summary>

A dot product between the final vector and that token's row in the LM head table: multiply number by number and add it up. A higher result means the vectors "agree", so that token is more likely next.
</details>

<details><summary><b>Q5.</b> Why did the model say " located" instead of " New" for "The capital of India is"?</summary>

It predicts what text usually comes next, not "the answer". In its training text, "...is located in" and quiz blanks "____" were common continuations. Adding an example ("The capital of France is Paris.") changed the pattern, and " New" jumped to 65%.
</details>

<details><summary><b>Q6.</b> Read this shape out loud: <code>[1, 5, 151936]</code></summary>

1 sentence, 5 positions, 151,936 scores at each position. For generation we only use the last position's scores.
</details>

<details><summary><b>Q7.</b> What shape do you get from <code>[5, 1024] × [1024, 151936]</code>?</summary>

`[5, 151936]`. The inner sizes (1024) must match and "disappear"; the outer sizes remain.
</details>

<details><summary><b>Q8.</b> What does RMSNorm do, and why is it needed?</summary>

It rescales each token's vector to a standard size (divide by its root-mean-square), then multiplies by learned weights. Without it, numbers would explode or vanish over 28 layers.
</details>

<details><summary><b>Q9.</b> Why switch to float32 inside RMSNorm?</summary>

bf16 only keeps about 3 digits. Squaring and averaging loses precision, and errors pile up over 28 layers. float32 keeps it accurate, and it matches the official code.
</details>

<details><summary><b>Q10.</b> Why is generation (Step 7) wasteful right now, and what fixes it?</summary>

Each new token re-runs the model on the whole sentence from the start, redoing work it already did. The KV cache saves past results so each step only processes the one new token.
</details>

<details><summary><b>Q11.</b> How much KV cache memory does one token need in Qwen3-0.6B?</summary>

28 layers × 2 (K and V) × 8 heads × 128 × 2 bytes = 114,688 bytes ≈ 112 KB.
</details>

---

## Part 10: Glossary

| Term | Plain-English meaning |
|---|---|
| **LLM** | Large language model: predicts the next token, over and over |
| **Qwen3-0.6B** | Alibaba's open LLM, version 3, 0.6 billion numbers. The model we use |
| **Inference** | Using a trained model to get predictions (vs training = teaching it) |
| **Inference engine** | The program that runs a model efficiently (vLLM, SGLang, and our tinyinfer) |
| **Token** | A chunk of text (word or part of a word) |
| **Vocabulary** | All tokens the model knows: 151,936 |
| **Tokenizer** | Converts text ↔ token IDs |
| **Vector** | A list of numbers. Each token becomes a vector of 1,024 numbers |
| **Tensor** | A grid of numbers with any number of dimensions |
| **Shape** | Size of each dimension, e.g. `[1, 5, 1024]` |
| **Batch** | How many sentences are processed at once (first number in the shape) |
| **Embedding** | Lookup table: token ID → vector |
| **Layer** | One repeated block: RMSNorm → attention → RMSNorm → MLP (×28) |
| **Attention** | Lets each token look at earlier tokens for context |
| **Query / Key / Value** | What I'm looking for / what I contain / what I pass on |
| **Head** | One of several parallel attention "lookers" |
| **GQA** | Fewer K/V heads than query heads, giving a smaller KV cache |
| **MLP** | Per-token processing block, where much of the knowledge lives |
| **LM head** | Final part: vector → a score for every token |
| **Logits** | Those raw scores |
| **Softmax** | Converts scores into probabilities that add up to 100% |
| **Dot product** | Multiply two lists number by number and add up: "how much do they agree" |
| **Greedy decoding** | Always pick the highest-probability token |
| **KV cache** | Saved keys and values from past tokens, so we don't recompute them |
| **Tied embeddings** | Embedding table and LM head are the same matrix |
| **bf16 / float32** | Number formats: 2 bytes and rough vs 4 bytes and precise |
| **Broadcasting** | PyTorch automatically stretching a size-1 dimension to match |
| **MPS** | Apple's GPU backend for PyTorch on Mac |

---

## ⏭️ Next: RoPE, or how the model knows word order

Surprising fact: attention on its own **can't tell word order**. "Dog bites man" and "man bites dog" would look identical to it. **RoPE** (rotary position embedding) fixes this by rotating each query and key vector by an angle based on its position. That's the next thing we build.
