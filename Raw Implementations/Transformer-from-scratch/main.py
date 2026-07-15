import numpy as np
from debugpy.launcher import output

from bpe import BPETokenizer
from tokenembedding import TokenEmbedding
from positonalencoding import PositionalEncoding
from transformer import TransformerBlock
from transformerencoder import Transformer
from outputhead import OutputHead
tokenizer = BPETokenizer()

corpus = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the cat and the dog sat in the park",
    "a dog and a cat ran on the mat",
    "the bird sat on the tree",
    "a mouse ran across the floor",
    "the cat chased the mouse in the park",
    "the dog chased the cat",
    "a bird and a cat sat on the mat",
    "the dog slept on the floor",
    "the cat slept in the car",
    "a dog played in the park",
    "the bird flew over the tree and the park",
    "the mouse slept on the mat",
    "the cat sat and slept on the mat",
    "the dog and the bird played in the park",
    "a cat chased a bird in the tree",
    "the dog ran across the park and the floor",
    "the mouse and the cat played on the floor",
    "the cat walked to the park",
    "the dog walked to the mat",
    "the bird sat on the dog",
    "the cat and the mouse ran to the car",
    "a dog slept in the car and a cat slept on the mat",
    "the tree in the park had a bird",
    "the mouse ran away from the cat",
    "the dog barked at the bird in the tree",
    "a cat and a dog slept on the floor",
    "the bird sang in the park",
    "the mat on the floor had a sleeping cat"
]

tokenizer.train(corpus, num_merges=15)

test_sentence = "the cat and the dog"
ids = np.array(tokenizer.encode(test_sentence))
embed_dim = 50
embed_dimmbeding_obj = TokenEmbedding(len(tokenizer.token_to_id),embed_dim)
embeded_token =embed_dimmbeding_obj.forward(ids)
position_embedding = PositionalEncoding(len(ids),embed_dim).forward(embeded_token)
num_heads = 5
d_kk = 100
# transformer_block = TransformerBlock(embed_dim,num_heads,d_kk)
# output = transformer_block.forward(position_embedding)
# print(output.shape)
num_layers = 6
details = f"ids:{ids}\n embed_dim:{embed_dim} \n num_heads:{num_heads} \n d_kk:{d_kk} \n num_layers:{num_layers} \n vocab_size:{len(tokenizer.token_to_id)}"
print(details)
tranformer_obj = Transformer(num_layers, embed_dim, num_heads, d_kk)
output = OutputHead(embed_dimmbeding_obj)
final_output = output.forward(tranformer_obj.forward(position_embedding))
print(final_output.shape)





