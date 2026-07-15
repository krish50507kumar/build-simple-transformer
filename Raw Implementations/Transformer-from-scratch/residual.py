from layernorm import LayerNorm
class AddNorm:
    def __init__(self, embed_dim):
        self.layer_norm = LayerNorm(embed_dim)
    def forward(self, x, sublayer_output):
        return self.layer_norm.forward(x + sublayer_output)