import numpy as np

class SGD:
    def __init__(self, layers, lr=1e-3):
        self.layers = list(layers.values())
        self.lr = lr

    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass