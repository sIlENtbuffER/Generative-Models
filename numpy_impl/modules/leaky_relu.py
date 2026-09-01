import numpy as np

class LeakyReLU:
    def __init__(self, slope=0.2):
        # Set to 0.2 as DCGAN
        self.slope = slope

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.slope * x)

    def backward(self, grad):
        return np.where(self.x > 0, grad, self.slope * grad)
