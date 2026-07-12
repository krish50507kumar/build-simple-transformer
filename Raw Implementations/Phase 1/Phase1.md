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

## Byte-level fallback (upgrade from character-level)

**The problem it solves:** character-level BPE can only tokenize characters it saw during training. Feed it an unseen script (emoji, Japanese, accented letters not in the corpus) and those characters have no symbol at all — they fall back to an `<unk>`/`-1` placeholder, and the original information is lost.

**The fix:** use raw UTF-8 **bytes** as the base vocabulary instead of characters. Every character in every language decomposes into 1–4 bytes from a fixed set of 256 possible values, so nothing is ever truly unrepresentable.

### `word_to_byte_symbols(word: str) -> tuple`
Same job as `word_to_symbols`, but breaks a word into its raw UTF-8 **bytes** (each a single-byte `bytes` object) instead of characters. The end-of-word marker is also stored as `bytes` (`b'</w>'`), not a string, so it can be concatenated with the other symbols during merging.

**Input:**
```python
"café"
```
**Output:**
```python
(b'c', b'a', b'f', b'\xc3', b'\xa9', b'</w>')
```
Note `'é'` takes 2 bytes (`\xc3\xa9`) — byte length isn't always the same as character length.

**Why not a digit-string like `'99'` for each byte?** Tried that first — it breaks because merging two digit-strings (e.g. `'108'` + `'111'` → `'108111'`) is ambiguous to reverse: there's no way to know if that was originally `108, 111` or `1, 0, 8, 111`, etc. Raw `bytes` objects avoid this entirely — concatenation is always losslessly reversible because Python tracks exact byte boundaries, no parsing needed.

**Everything downstream is unchanged** — `get_pair_counts`, `merge_pair`, `train_bpe`, `encode_word`, `build_vocab`, `encode` all operate generically on "symbols" and don't care whether a symbol is a `str` or a `bytes` object.

**Input:**
```python
encode_word("日本語", merges_bytes)   # never seen in training
```
**Output:**
```python
(b'\xe6', b'\x97', b'\xa5', b'\xe6', b'\x9c', b'\xac', b'\xe8', b'\xaa', b'\x9e', b'</w>')
```
9 bytes (3 per CJK character) — nothing dropped, nothing unknown, just falls back to individual bytes when no merge rule applies.

---

### `decode_bytes(ids: list[int], id_to_token: dict) -> str`
Reverses byte-level `encode`: looks up each id's `bytes` object, joins them all into **one** combined `bytes` object first, replaces the `b'</w>'` marker with a space, then calls `.decode('utf-8')` **once** at the very end.

**Input:**
```python
ids = encode("héllo", merges_bytes, token_to_id_b)
decode_bytes(ids, id_to_token_b)
```
**Output:**
```python
"héllo "
```

**Why decode once, at the end, instead of per-token?** A multi-byte character's bytes (like `é` → `\xc3\xa9`) can end up split across two separate tokens. Decoding a lone `\xc3` by itself would crash (`UnicodeDecodeError`) since it's only half a character. Joining *all* bytes first guarantees every multi-byte character's pieces are back together, in order, before decoding is attempted.

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
- ~~**Byte-level fallback**~~ — done. See "Byte-level fallback" section above (`word_to_byte_symbols`, `decode_bytes`).
- **Pre-tokenization regex**: no handling yet for punctuation/contraction boundaries before BPE runs.