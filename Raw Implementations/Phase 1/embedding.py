import numpy as np
from debugpy.launcher import output

np.random.seed(42)

def create_embedding_table(vocab_size: int, embedding_dim: int) -> np.ndarray:
    return np.random.randn(vocab_size, embedding_dim)

table = create_embedding_table(vocab_size=10, embedding_dim=4)
# print(table.shape)   # should print (10, 4)
# print(table)

def embed_tokens(token_ids: list[int], table: np.ndarray) -> np.ndarray:
    return table[token_ids]

token_ids = [3, 6]
embeddings = embed_tokens(token_ids, table)
# print(embeddings.shape)   # (2, 4)
# print(embeddings)

def positional_encoding(seq_len: int, embedding_dim: int) -> np.ndarray:
    positon_matrix = np.arange(seq_len)[:, None]
    i = np.arange(embedding_dim // 2)[None, :]
    freq_exponent = 2 * i / embedding_dim
    freq_denom = 10000 ** freq_exponent
    angles = positon_matrix / freq_denom
    sin_part = np.sin(angles)
    cos_part = np.cos(angles)
    output = np.zeros((seq_len,embedding_dim))
    output[:, 0::2] = sin_part
    output[:, 1::2] = cos_part
    return output

# final_input = embed_tokens(token_ids, table) + positional_encoding(seq_len=table.shape[0], embedding_dim=table.shape[1])

def create_qkv_weights(embedding_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W_q = np.random.randn(embedding_dim, embedding_dim)
    W_k = np.random.randn(embedding_dim, embedding_dim)
    W_v = np.random.randn(embedding_dim, embedding_dim)
    return W_q, W_k, W_v

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V
seq_len, embedding_dim = 3, 4
X = np.random.randn(seq_len, embedding_dim)   # pretend this is your embeddings+positions
W_q, W_k, W_v = create_qkv_weights(embedding_dim)
Q, K, V = compute_qkv(X, W_q, W_k, W_v)
# print(Q.shape, K.shape, V.shape)
def compute_attention_scores(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.matmul(Q, K.T)/np.sqrt(Q.shape[1])

def softmax(scores: np.ndarray) -> np.ndarray:
    exp_scores = np.exp(scores)
    row_sums = np.sum(exp_scores, axis=1, keepdims=True)
    return exp_scores / row_sums
scaled_scores = compute_attention_scores(Q, K)
weights = softmax(scaled_scores)
print(weights.shape)
print(weights.sum(axis=1))
def attention_output(weights: np.ndarray, V: np.ndarray) -> np.ndarray:
    return np.matmul(weights, V)