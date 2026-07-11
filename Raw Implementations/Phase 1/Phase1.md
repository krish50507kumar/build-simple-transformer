# BPE Tokenizer — Function Reference

Quick-revision notes: each function's purpose, plus a concrete input → output example.

---

### `get_word_freqs(corpus: list[str]) -> dict[str, int]`
Splits a corpus of sentences into words and counts how often each word occurs.

**Input:**
```python
["low lower lowest", "newer newest", "wide widest"]
```
**Output:**
```python
{'low': 1, 'lower': 1, 'lowest': 1, 'newer': 1, 'newest': 1, 'wide': 1, 'widest': 1}
```

---

### `word_to_symbols(word: str) -> tuple`
Breaks a single word into a tuple of individual characters, appending an end-of-word marker `</w>`.

**Input:**
```python
"lower"
```
**Output:**
```python
('l', 'o', 'w', 'e', 'r', '</w>')
```

---

### `spliter(freqs) -> dict` *(lambda)*
Applies `word_to_symbols` to every word in `freqs`, building the `word → symbol tuple` mapping used throughout training.

**Input:**
```python
{'low': 1, 'lower': 1}
```
**Output:**
```python
{'low': ('l', 'o', 'w', '</w>'), 'lower': ('l', 'o', 'w', 'e', 'r', '</w>')}
```

---

### `get_pair_counts(splits: dict, freqs: dict) -> dict`
Counts every adjacent symbol pair across all words, weighted by each word's frequency. This is what decides which pair gets merged next.

**Input:**
```python
splits = {'low': ('l', 'o', 'w', '</w>'), 'lower': ('l', 'o', 'w', 'e', 'r', '</w>')}
freqs = {'low': 3, 'lower': 2}
```
**Output:**
```python
{('l','o'): 5, ('o','w'): 5, ('w','</w>'): 3, ('w','e'): 2, ('e','r'): 2, ('r','</w>'): 2}
```

---

### `merge_pair(pair: tuple, splits: dict) -> dict`
Fuses one specific symbol pair together wherever it appears (adjacently) across all words in `splits`.

**Input:**
```python
pair = ('l', 'o')
splits = {'low': ('l', 'o', 'w', '</w>'), 'lower': ('l', 'o', 'w', 'e', 'r', '</w>')}
```
**Output:**
```python
{'low': ('lo', 'w', '</w>'), 'lower': ('lo', 'w', 'e', 'r', '</w>')}
```

---

### `train_bpe(freqs: dict, num_merges: int) -> (list, dict)`
Runs the full training loop: repeatedly finds the most frequent pair and merges it, `num_merges` times (or until no pairs remain). Returns the ordered list of learned merges plus the final splits.

**Input:**
```python
freqs = {'low': 1, 'lower': 1, 'lowest': 1, 'newer': 1, 'newest': 1, 'wide': 1, 'widest': 1}
num_merges = 10
```
**Output:**
```python
merges = [('w','e'), ('l','o'), ('s','t'), ('st','</w>'), ('lo','we'), ('r','</w>'), ('n','e'), ('ne','we'), ('w','i'), ('wi','d')]

final_splits = {
    'low': ('lo', 'w', '</w>'),
    'lower': ('lowe', 'r</w>'),
    'lowest': ('lowe', 'st</w>'),
    'newer': ('newe', 'r</w>'),
    'newest': ('newe', 'st</w>'),
    'wide': ('wid', 'e', '</w>'),
    'widest': ('wid', 'e', 'st</w>')
}
```

---

### `encode_word(word: str, merges: list) -> tuple`
Tokenizes a brand-new word by replaying the learned merges **in order**. Works on both seen and unseen words — unseen words just get partially merged.

**Input:**
```python
encode_word("loudest", merges)
```
**Output:**
```python
('lo', 'u', 'd', 'e', 'st</w>')
```

---

### `build_vocab(splits: dict) -> (dict, dict)`
Collects every unique token across `splits` and assigns each a unique integer ID. Returns `token_to_id` and its reverse, `id_to_token`.

**Input:**
```python
final_splits  # from train_bpe
```
**Output:**
```python
token_to_id = {'</w>': 0, 'e': 1, 'lo': 2, 'lowe': 3, ..., 'st</w>': 6, ...}
id_to_token = {0: '</w>', 1: 'e', 2: 'lo', 3: 'lowe', ..., 6: 'st</w>', ...}
```

---

### `encode(word: str, merges: list, token_to_id: dict) -> list[int]`
Full encoding pipeline: tokenize the word (`encode_word`), then map each resulting token to its integer ID.

**Input:**
```python
encode("lowest", merges, token_to_id)
```
**Output:**
```python
[3, 6]   # ids for 'lowe' and 'st</w>'
```

---

### `decode(ids: list[int], id_to_token: dict) -> str`
Reverses `encode`: maps ids back to tokens, joins them into one string, and turns `</w>` markers into spaces.

**Input:**
```python
decode([3, 6], id_to_token)
```
**Output:**
```python
"lowest"
```

---

## Pipeline at a glance

```
corpus
  → get_word_freqs        → word frequencies
  → spliter/word_to_symbols → symbol tuples per word
  → train_bpe (loops get_pair_counts + merge_pair)
      → merges (ordered list)  +  final_splits
  → build_vocab            → token_to_id, id_to_token
  → encode_word / encode   → tokenize + map new words to ids
  → decode                 → ids back to text
```

## Known gaps (extensions to explore)
- **Efficiency**: pair counts are recomputed from scratch every merge — should update incrementally instead.
- **Byte-level fallback**: unseen characters currently map to `-1` (unknown) — starting from raw bytes avoids this entirely.
- **Pre-tokenization regex**: no handling yet for punctuation/contraction boundaries before BPE runs.