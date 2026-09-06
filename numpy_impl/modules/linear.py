import numpy as np
from initializer import he

class Linear:
    def __init__(self, in_dim, out_dim, rng, init=he):
        self.W = init(rng=rng, in_dim=in_dim, out_dim=out_dim)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T