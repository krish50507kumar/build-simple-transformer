from stringprep import b1_set

import numpy as np
from debugpy.launcher import output
from numpy.ma.extras import hstack

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
# print(weights.shape)
# print(weights.sum(axis=1))
def attention_output(weights: np.ndarray, V: np.ndarray) -> np.ndarray:
    return np.matmul(weights, V)

def create_multihead_qkv_weights(embedding_dim: int, num_heads: int) -> list:
    head_dim = embedding_dim // num_heads
    list_of_qkv_weights = []
    for _ in range(num_heads):
        list_of_qkv_weights.append((np.random.randn(embedding_dim, head_dim), np.random.randn(embedding_dim, head_dim), np.random.randn(embedding_dim, head_dim)))
    return list_of_qkv_weights
# weights_per_head = create_multihead_qkv_weights(embedding_dim=8, num_heads=2)
# for W_q, W_k, W_v in weights_per_head:
#     print(W_q.shape, W_k.shape, W_v.shape)
def multihead_attention(X: np.ndarray, weights_per_head: list) -> np.ndarray:
    head_outputs = []
    for W_q, W_k, W_v in weights_per_head:
        Q, K, V = compute_qkv(X, W_q, W_k, W_v)
        scores = compute_attention_scores(Q, K)
        weights = softmax(scores)
        output = attention_output(weights, V)
        head_outputs.append(output)
    return np.concatenate(head_outputs, axis=1)
# X = np.random.randn(3, 8)
# weights_per_head = create_multihead_qkv_weights(embedding_dim=8, num_heads=2)
# result = multihead_attention(X, weights_per_head)
# # print(result.shape)
def create_output_projection(embedding_dim: int) -> np.ndarray:
    return np.random.randn(embedding_dim, embedding_dim)

def apply_output_projection(concatenated: np.ndarray, W_o: np.ndarray) -> np.ndarray:
    return concatenated @ W_o

def add_residual(X: np.ndarray, sublayer_output: np.ndarray) -> np.ndarray:
    return X + sublayer_output

X = np.random.randn(3, 8)
weights_per_head = create_multihead_qkv_weights(embedding_dim=8, num_heads=2)
concatenated = multihead_attention(X, weights_per_head)
W_o = create_output_projection(embedding_dim=8)
attention_result = apply_output_projection(concatenated, W_o)

residual_output = add_residual(X, attention_result)
# print(residual_output.shape)

def compute_mean_variance(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=1, keepdims=True)
    variance = np.var(X, axis=1, keepdims=True)
    return mean, variance
def normalize(X: np.ndarray, mean: np.ndarray, variance: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    return (X - mean) / np.sqrt(variance + epsilon)

def create_layernorm_params(embedding_dim: int) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.ones(embedding_dim)
    beta = np.zeros(embedding_dim)
    return gamma, beta

def apply_scale_shift(normalized: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return normalized * gamma + beta
mean, var = compute_mean_variance(residual_output)
normalized = normalize(residual_output, mean, var)
gamma, beta = create_layernorm_params(embedding_dim=8)
final = apply_scale_shift(normalized, gamma, beta)
# print(final.shape)
# print(final.mean(axis=1))
# print(final.var(axis=1))

def create_ffn_weights(embedding_dim: int, hidden_dim: int) -> tuple:
    W1 = np.random.randn(embedding_dim, hidden_dim)
    W2 = np.random.randn(hidden_dim, embedding_dim)
    b1= np.zeros(hidden_dim,)
    b2= np.zeros(embedding_dim,)
    return W1,b1,W2,b2
def feed_forward(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return np.maximum(X @ W1 + b1,0)@ W2 + b2

W1, b1, W2, b2 = create_ffn_weights(embedding_dim=8, hidden_dim=32)
ffn_output = feed_forward(final, W1, b1, W2, b2)
print(ffn_output.shape)  