import numpy as np

from multiheadattention import MultiHeadAttention
from feed_forward import FeedForward
from residual  import AddNorm
class TransformerBlock:
    def __init__(self,embed_dim,num_heads,d_ff):
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, d_ff)
        self.add_norm1 = AddNorm(embed_dim)
        self.add_norm2 = AddNorm(embed_dim)
    def forward(self,x):
        x1 = self.add_norm1.forward(x,self.multi_head_attention.forward(x))
        output = self.add_norm2.forward(x1,self.feed_forward.forward(x1))
        return output