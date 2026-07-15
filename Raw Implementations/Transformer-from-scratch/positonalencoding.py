import numpy as np
class PositionalEncoding:
    def __init__(self, seq_len,embed_dim) -> None:
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pe_table = self.positional_encoding(seq_len,embed_dim)

    def positional_encoding(self,seq_len: int, embedding_dim: int) -> np.ndarray:
        positon_matrix = np.arange(seq_len)[:, None]
        i = np.arange(embedding_dim // 2)[None, :]
        freq_exponent = 2 * i / embedding_dim
        freq_denom = 10000 ** freq_exponent
        angles = positon_matrix / freq_denom
        sin_part = np.sin(angles)
        cos_part = np.cos(angles)
        output = np.zeros((seq_len, embedding_dim))
        output[:, 0::2] = sin_part
        output[:, 1::2] = cos_part
        return output

    def forward(self,x):
        return x+self.pe_table[:len(x)]