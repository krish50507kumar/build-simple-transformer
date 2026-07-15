import numpy as np
class OutputHead:
    def __init__(self, token_embedding):
        self.token_embedding = token_embedding
        self.bias = np.zeros((self.token_embedding.embed_table.shape[0],))
    def forward(self, x):
        return x @ self.token_embedding.embed_table.T  + self.bias