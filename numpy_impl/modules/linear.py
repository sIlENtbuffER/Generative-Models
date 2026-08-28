import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim, rng):
        # I use He initialization here :)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(loc=0.0, scale=scale, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T