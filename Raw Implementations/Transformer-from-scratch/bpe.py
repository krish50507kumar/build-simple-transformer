from collections import Counter,defaultdict

class BPETokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.all_merges = []

    def __get_word_freqs(self,corpus: list[str]) -> dict[str, int]:
        freqs = Counter()
        for sentence in corpus:
            words = sentence.strip().split()
            for word in words:
                freqs[word] += 1
        return freqs

    def __word_to_byte_symbols(self,word: str) -> tuple:
        word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
        marker = (b"</w>",)
        return word_bytes + marker

    def __get_pair_counts(self,splits, word_freqs):
        pair_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq
        return pair_counts

    def __spliter(self,freqs):
        return {word : self.__word_to_byte_symbols(word) for word in freqs.keys()}

    def __merge_pair(self,pair: tuple, splits: dict) -> dict:
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

    def __train_bpe(self,freqs: dict, num_merges: int):
        all_merges = []
        splits = self.__spliter(freqs)
        for _ in range(num_merges):
            pair_counts = self.__get_pair_counts(splits, freqs)
            if not pair_counts: break
            pair = max(pair_counts, key=pair_counts.get)
            merged = self.__merge_pair(pair, splits)
            all_merges.append(pair)
            splits = merged
        return all_merges, splits

    def __build_vocab(self,splits: dict) -> tuple[dict, dict]:
        unique_symbols = set()
        unique_symbols.update([bytes([i]) for i in range(256)])
        for split in splits.values():
            unique_symbols.update(split)
        token_to_id = dict(zip(sorted(unique_symbols), range(len(unique_symbols))))
        id_to_token = {v: k for k, v in token_to_id.items()}
        return token_to_id, id_to_token

    def __encode_word(self,word: str, merges: list) -> list:
        list_of_words = word.split()
        splits = {}
        for idx,word in enumerate(list_of_words):
            symbols = list(self.__word_to_byte_symbols(word))
            splits[idx] = symbols
        for pair in merges:
            a, b = pair
            splits = self.__merge_pair(pair, splits)
        token = []
        for idx,symbols in splits.items():
            token.extend(symbols)
        return token

    def train(self, corpus, num_merges):
        freqs = self.__get_word_freqs(corpus)
        self.all_merges,final_splits = self.__train_bpe(freqs, num_merges)
        self.token_to_id , self.id_to_token = self.__build_vocab(final_splits)

    def encode(self, text):
        text = text.strip()
        tokens = self.__encode_word(text, self.all_merges)
        ids = []
        for token in tokens:
            ids.append(self.token_to_id.get(token, -1))
        return ids

    def decode(self, ids):
        list_of_tokens = [self.id_to_token[i] for i in ids]
        joined = b''.join(list_of_tokens)
        result = joined.replace(b'</w>', b" ")
        return result.decode('utf-8')