from transformer import TransformerBlock
class Transformer:
    def __init__(self,num_layers,embed_dim,num_heads,d_kk):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_kk = d_kk
        self.layers = [TransformerBlock(embed_dim,num_heads,d_kk) for _ in range(num_layers)]
    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x