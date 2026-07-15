from idlelib.query import Query
from lib2to3.fixes.fix_input import context

import numpy as np
from debugpy.launcher import output


class MultiHeadAttention:
    def __init__(self,embed_dim:int,num_heads:int):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.w_q = np.random.randn(embed_dim,embed_dim)*0.01
        self.w_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.w_v = np.random.randn(embed_dim, embed_dim) * 0.01
        self.w_o = np.random.randn(embed_dim, embed_dim) * 0.01
    def forward(self,x):
        Q = (x @ self.w_q).reshape(len(x), self.num_heads, self.d_k).transpose(1, 0, 2)
        K = (x @ self.w_k).reshape(len(x), self.num_heads, self.d_k).transpose(1, 0, 2)
        V = (x @ self.w_v).reshape(len(x), self.num_heads, self.d_k).transpose(1, 0, 2)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)
        scores_stable = scores - np.max(scores, axis=2, keepdims=True)
        exp_scores = np.exp(scores_stable)
        attention_probs = exp_scores / np.sum(exp_scores, axis=2, keepdims=True)
        context = (attention_probs @ V).transpose(1,0,2)
        concatenated = context.reshape(len(x),self.num_heads*self.d_k)
        output = concatenated @ self.w_o
        return output
