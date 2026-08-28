import numpy as np

class Adam:
    def __init__(self, layers, lr=1e-3, beta1 = 0.9, beta2 = 0.999, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0 # Update time

        self.m_W = {}
        self.m_b = {}
        self.v_W = {}
        self.v_b = {}

        for name, layer in self.layers.items():
            self.m_W[name] = np.zeros_like(layer.W)
            self.m_b[name] = np.zeros_like(layer.b)
            self.v_W[name] = np.zeros_like(layer.W)
            self.v_b[name] = np.zeros_like(layer.b)

    def step(self):
        self.t += 1

        for name, layer in self.layers.items():
            self.m_W[name] = self.beta1 * self.m_W[name] + (1.0 - self.beta1) * layer.dW
            self.m_b[name] = self.beta1 * self.m_b[name] + (1.0 - self.beta1) * layer.db
            self.v_W[name] = self.beta2 * self.v_W[name] + (1.0 - self.beta2) * layer.dW**2
            self.v_b[name] = self.beta2 * self.v_b[name] + (1.0 - self.beta2) * layer.db**2

            # Bias correction
            m_W_hat = self.m_W[name] / (1 - self.beta1**self.t)
            m_b_hat = self.m_b[name] / (1 - self.beta1**self.t)
            v_W_hat = self.v_W[name] / (1 - self.beta2**self.t)
            v_b_hat = self.v_b[name] / (1 - self.beta2**self.t)

            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

    def state_dict(self):
        state = {}
        for name in self.layers:
            state[f"m.{name}_W"] = self.m_W[name]
            state[f"m.{name}_b"] = self.m_b[name]
            state[f"v.{name}_W"] = self.v_W[name]
            state[f"v.{name}_b"] = self.v_b[name]
        return state

    def load_state_dict(self, state):
        for name in self.layers:
            self.m_W[name] = state[f"m.{name}_W"]
            self.m_b[name] = state[f"m.{name}_b"]
            self.v_W[name] = state[f"v.{name}_W"]
            self.v_b[name] = state[f"v.{name}_b"]
