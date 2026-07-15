import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits, targets) -> float:
        pass
    def backward(self) -> np.ndarray:
        pass