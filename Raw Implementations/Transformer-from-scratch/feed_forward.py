import numpy as np
class FeedForward:
    def __init__(self, embed_dim,d_ff) -> None:
        self.embed_dim = embed_dim
        self.d_ff = d_ff
        self.W1 = np.random.randn(embed_dim, d_ff)* 0.01
        self.W2 = np.random.randn(d_ff, embed_dim)* 0.01
        self.b1 = np.zeros(d_ff, )
        self.b2 = np.zeros(embed_dim, )
    def forward(self,X: np.ndarray) -> np.ndarray:
        return np.maximum(X @ self.W1 + self.b1, 0) @ self.W2 + self.b2