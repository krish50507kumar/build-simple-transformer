import numpy as np
class LayerNorm:
    def __init__(self,embed_dim):
        self.gamma = np.ones((embed_dim,))
        self.beta = np.zeros((embed_dim,))
        self.epsilon = 1e-5
    def forward(self,x):
        mean = np.mean(x, axis=1, keepdims=True)
        variance = np.var(x, axis=1, keepdims=True)
        normalized_x = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * normalized_x + self.beta