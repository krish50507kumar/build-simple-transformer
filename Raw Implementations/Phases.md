**Phase 1: Build the pieces from scratch (NumPy)**

* Implement token embeddings
* Implement positional encoding
* Implement scaled dot-product attention
* Implement multi-head attention
* Implement LayerNorm
* Implement the Feed-Forward Network
* Combine them into a Transformer block



This builds intuition because I'll see every matrix multiplication myself.



**Phase 2: PyTorch Implementation**



Reimplement everything using PyTorch:



* nn.Embedding
* nn.Linear
* nn.LayerNorm
* Multi-head attention
* Transformer block
* Tiny GPT model

**Phase 3: Train a Tiny GPT**



Train on a very small dataset like:



* Shakespeare
* TinyStories
* Simple text files



The goal isn't to build ChatGPT. The goal is to watch the entire training pipeline work.



**Phase 4: Use Hugging Face**



Once I understand the internals:

* Load pretrained models
* Fine-tune them
* Generate text
* Experiment with temperature, top-k, and top-p
* Try LoRA/QLoRA fine-tuning

