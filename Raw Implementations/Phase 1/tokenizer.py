from collections import Counter,defaultdict

def get_word_freqs(corpus: list[str]) -> dict[str, int]:
    freqs = Counter()
    for sentence in corpus:
        words = sentence.strip().split()
        for word in words:
            freqs[word] += 1
    return freqs

corpus = ["low lower lowest", "newer newest", "wide widest"]
freqs = get_word_freqs(corpus)
# print(freqs)

def word_to_byte_symbols(word: str) -> tuple:
    word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
    marker = (b"</w>",)
    return word_bytes + marker

def word_to_symbols(word: str) -> tuple:
    return tuple(list(word.strip())+['</w>'])
# print(word_to_symbols("lower"))
spliter = lambda freqs:{word : word_to_byte_symbols(word) for word in freqs.keys()}
splits = spliter(freqs)
# print(splits)

def get_pair_counts(splits, word_freqs):
    pair_counts = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = splits[word]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] += freq
    return pair_counts

def merge_pair(pair: tuple, splits: dict) -> dict:
    a, b = pair
    merged = a + b
    new_splits = {}
    for word, symbols in splits.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_splits[word] = tuple(new_symbols)
    return new_splits

def train_bpe(freqs: dict, num_merges: int):
    all_merges = []
    splits = spliter(freqs)
    for _ in range(num_merges):
        pair_counts = get_pair_counts(splits, freqs)
        if not pair_counts: break
        pair = max(pair_counts, key=pair_counts.get)
        merged = merge_pair(pair, splits)
        all_merges.append(pair)
        splits = merged
    return all_merges, splits

merges, final_splits = train_bpe(freqs, num_merges=10)
# print(merges)
# print(final_splits)
def encode_word(word: str, merges: list) -> list:
    symbols = list(word_to_byte_symbols(word))
    splits = {word:symbols}
    for pair in merges:
        a, b = pair
        splits = merge_pair(pair, splits)
    return splits[word]
# print(encode_word("lowest", merges))
# print(encode_word("loudest", merges))
def decode_bytes(words: list) -> str:
    return  bytes(list(map(int,words[:-1]))).decode('utf-8',)

def build_vocab(splits: dict) -> tuple[dict, dict]:
    unique_symbols = set()
    unique_symbols.update([bytes([i]) for i in range(256)])
    for split in splits.values():
        unique_symbols.update(split)
    token_to_id = dict(zip(sorted(unique_symbols), range(len(unique_symbols))))
    id_to_token = {v: k for k, v in token_to_id.items()}
    return token_to_id, id_to_token
token_to_id, id_to_token = build_vocab(final_splits)
# print(token_to_id)
# print(id_to_token)

def encode(word: str, merges: list, token_to_id: dict) -> list[int]:
    tokens = encode_word(word, merges)
    ids = []
    for token in tokens:
        ids.append(token_to_id.get(token,-1))
    return ids
# ids = encode("lowest", merges, token_to_id)
# print(ids)

def decode_bytes(ids: list[int], id_to_token: dict) -> str:
    list_of_tokens = [id_to_token[i] for i in ids]
    joined = b''.join(list_of_tokens)
    result = joined.replace(b'</w>', b" ")
    return result.decode('utf-8')

def decode(ids: list[int], id_to_token: dict) -> str:
    tokens = []
    for i in ids:
        tokens.append(id_to_token[i])
    joined = "".join(tokens)
    result = joined.replace("</w>", " ")
    return result

print(decode_bytes(encode("loudest",merges,token_to_id),id_to_token))


