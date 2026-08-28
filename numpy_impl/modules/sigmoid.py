import numpy as np

class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -500, 500) # Avoid overflow
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)
