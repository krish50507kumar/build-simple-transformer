import numpy as np
class TokenEmbedding :
    def __init__(self, vocab_size,embed_dim) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_table = np.random.randn(self.vocab_size, self.embed_dim)*0.01
    def forward(self,x):
        return self.embed_table[x]