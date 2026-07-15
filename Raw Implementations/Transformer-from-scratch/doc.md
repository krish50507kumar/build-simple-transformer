# Transformer From Scratch — OOP Implementation Notes

Summary of all classes built in this session, wrapping earlier functional BPE/attention code into a composable, class-based transformer encoder block.

---

## 1. `BPETokenizer`

**Owns (`__init__`):** `token_to_id`, `id_to_token`, `all_merges` — all empty until `train()` is called.

**Methods:**
- `train(corpus, num_merges)` — learns byte-level BPE merges from a list of sentences. Builds vocab + merge rules.
- `encode(text)` — splits text into words, converts each to byte-symbols (with `</w>` marker), applies learned merges *per word* (keyed by index to avoid collisions on repeated words), returns a flat list of token IDs.
- `decode(ids)` — joins byte tokens, replaces `</w>` with a space, decodes UTF-8 back to text.

**Key bugs fixed this session:**
- `encode` originally treated the whole input string as one "word" — fixed to split into words first, matching how `train` builds per-word symbol sequences.
- Used a `dict` keyed by *word string* to store per-word splits during encode — silently deduped repeated words (e.g. two occurrences of "the" collapsed into one). Fixed by keying on **word index** instead of the word string, preserving both uniqueness and original order (Python dicts preserve insertion order).

---

## 2. `TokenEmbedding`

**Owns:** `embed_table`, shape `(vocab_size, embed_dim)`, random-initialized (`* 0.01`).

**`forward(x)`:** fancy-indexes the table with token IDs → `(seq_len, embed_dim)`.

---

## 3. `PositionalEncoding`

**Owns:** precomputed sinusoidal table, shape `(max_seq_len, embed_dim)`, built once in `__init__` since it depends only on position/dimension, not on input data.

**`forward(x)`:** slices the table to `x`'s actual length and adds it elementwise: `x + pe_table[:len(x)]`.

---

## 4. `SelfAttention` (single-head)

**Owns:** `W_q`, `W_k`, `W_v`, each `(embed_dim, embed_dim)`, random-initialized (`* 0.01`).

**`forward(x)`:**
```
Q, K, V = x @ W_q, x @ W_k, x @ W_v
scores = (Q @ K.T) / sqrt(embed_dim)
scores_stable = scores - max(scores, axis=1, keepdims=True)   # numerical stability
attention_probs = softmax(scores_stable, axis=1)
return attention_probs @ V
```
**Key fix:** original softmax lacked the max-subtraction stability trick — without it, large scores can overflow `exp()` into `inf`/`NaN`.

---

## 5. `MultiHeadAttention`

**Owns:** `embed_dim`, `num_heads`, `d_k = embed_dim // num_heads` (asserted to divide evenly), `W_q`/`W_k`/`W_v`/`W_o`, all `(embed_dim, embed_dim)`.

**`forward(x)`** — vectorized across heads, no Python loop:
1. Project: `Q = x @ W_q` → `(seq_len, embed_dim)` (same for K, V)
2. Reshape + transpose into per-head form: `.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)` → `(num_heads, seq_len, d_k)`
3. Scores: `Q @ K.transpose(0, 2, 1) / sqrt(d_k)` → `(num_heads, seq_len, seq_len)`
4. Softmax over **axis=2** (the key axis — shifted by one vs. single-head case due to the leading heads axis)
5. `context = attention_probs @ V` → `(num_heads, seq_len, d_k)`
6. Transpose back + flatten: `.transpose(1, 0, 2).reshape(seq_len, embed_dim)`
7. Output projection: `concatenated @ W_o`

**Design insight:** one big projection + reshape (approach used by PyTorch/HuggingFace) is preferred over separate per-head weight matrices, since large batched matmuls are faster than many small ones.

---

## 6. `LayerNorm`

**Owns:** `gamma` (ones, shape `(embed_dim,)`), `beta` (zeros, shape `(embed_dim,)`), `epsilon` (`1e-5`).

**`forward(x)`:** normalizes **per token** (axis=1, across `embed_dim`):
```
mean, var = mean(x, axis=1), var(x, axis=1)
normalized = (x - mean) / sqrt(var + epsilon)
return gamma * normalized + beta
```
**Init logic:** `gamma=1, beta=0` → at initialization, LayerNorm behaves as pure normalization (identity scale/shift), letting training adjust from there. `epsilon` guards against division by ~zero variance.

---

## 7. `AddNorm` (residual + norm wrapper)

**Owns:** its own internal `LayerNorm(embed_dim)` instance (each `AddNorm` needs independent `gamma`/`beta`).

**`forward(x, sublayer_output)`:**
```
return self.layer_norm.forward(x + sublayer_output)
```
Implements `LayerNorm(x + Sublayer(x))` from the paper. A `TransformerBlock` needs **two** separate `AddNorm` instances — one for attention, one for feed-forward — since their normalization parameters aren't shared.

---

## 8. `FeedForward`

**Owns:** `W1 (embed_dim, d_ff)`, `b1 (d_ff,)`, `W2 (d_ff, embed_dim)`, `b2 (embed_dim,)`. Weights random-scaled (`* 0.01`), biases zero-initialized (biases are additive, so 0 is the "do nothing yet" value — unlike `gamma`, which is multiplicative and starts at 1).

**`forward(x)`:**
```
return relu(x @ W1 + b1) @ W2 + b2
```

---

## 9. `TransformerBlock` (full encoder block)

**Owns:** one `MultiHeadAttention`, one `FeedForward`, two `AddNorm` instances.

**`forward(x)`:**
```
x1 = add_norm1.forward(x, multi_head_attention.forward(x))
output = add_norm2.forward(x1, feed_forward.forward(x1))
return output
```

Matches the paper's block exactly:
```
x1     = LayerNorm(x  + MultiHeadAttention(x))
output = LayerNorm(x1 + FeedForward(x1))
```

---

## Verified end-to-end pipeline

```
raw text
  → BPETokenizer.encode()        → token IDs
  → TokenEmbedding.forward()     → (seq_len, embed_dim)
  → PositionalEncoding.forward() → (seq_len, embed_dim)
  → TransformerBlock.forward()   → (seq_len, embed_dim)
```

Tested on `"the cat and the dog"` (BPE-split into 6 tokens) → confirmed correct output shape `(6, 50)`.

**Important caveat:** every weight is currently randomly initialized and untrained — no loss function or backprop exists yet. This is why attention output rows look nearly identical (uniform softmax over untrained, near-zero scores). This is expected, not a bug.

---

## Naming cross-reference (your code vs. real libraries)

| Concept | Your code | PyTorch (`nn.MultiheadAttention`) | HuggingFace (BERT/GPT-style) |
|---|---|---|---|
| Model dimension | `embed_dim` | `embed_dim` | `hidden_size` |
| Per-head dimension | `d_k` | `head_dim` | `head_dim` |
| Query/Key/Value weights | `W_q`, `W_k`, `W_v` | fused `in_proj_weight` | `query`, `key`, `value` (Linear layers) |
| Output projection | `W_o` | `out_proj` | `output.dense` |
| Post-softmax weights | `attention_probs` | internal | `attention_probs` |
| Final attention output | returned value | `attn_output` | `context` / `context_layer` |

---

## What's next (not yet built)

- Stack multiple `TransformerBlock`s (real transformers use several, e.g. 6–12)
- Output head: project final hidden states back to vocab-size logits
- Loss function (e.g. cross-entropy for next-token prediction)
- Backpropagation / autodiff for every operation (matmuls, softmax, LayerNorm, ReLU) — needed for any actual "learning" to happen
- Optimizer / weight update step (e.g. gradient descent)