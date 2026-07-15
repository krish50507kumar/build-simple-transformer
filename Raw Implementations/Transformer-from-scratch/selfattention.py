from idlelib.query import Query
from lib2to3.fixes.fix_input import context

import numpy as np
class SelfAttention:
    def __init__(self,embed_dim):
        self.embed_dim = embed_dim
        self.w_q = np.random.randn(embed_dim, embed_dim)*0.01
        self.w_k = np.random.randn(embed_dim, embed_dim)*0.01
        self.w_v = np.random.randn(embed_dim, embed_dim)*0.01
    def forward(self, x):
        Q = x @ self.w_q
        K = x @ self.w_k
        V = x @ self.w_v
        attention_scores = (Q @ K.T)/np.sqrt(self.embed_dim)
        scores_stable = attention_scores - np.max(attention_scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_stable)
        row_sums = np.sum(exp_scores, axis=1, keepdims=True)
        attention_probs = exp_scores / row_sums
        return attention_probs @ V

