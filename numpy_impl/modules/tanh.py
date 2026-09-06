import numpy as np

class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad):
        return grad * (1 - self.output**2)
