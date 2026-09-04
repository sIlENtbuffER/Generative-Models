import numpy as np
from modules import Linear, LeakyReLU, Tanh
from initializer import normal

class Generator:
    def __init__(self, rng, input_dim, latent_dim, hidden_dims):
        self.rng = rng
        self.latent_dim = latent_dim
        self.layers = []

        dims = [latent_dim] + hidden_dims
        for in_dim, out_dim in zip(dims, dims[1:]):
            self.layers.append(Linear(in_dim=in_dim, out_dim=out_dim, rng=rng))
            self.layers.append(LeakyReLU())
        self.layers.append(Linear(in_dim=hidden_dims[-1], out_dim=input_dim, rng=rng))
        self.layers.append(Tanh())

    def forward(self, z):
        for layer in self.layers:
            z = layer.forward(z)
        return z

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def sample(self, num_samples):
        z = self.rng.normal(loc=0.0, scale=1.0, size=(num_samples, self.latent_dim)).astype(np.float32)
        return self.forward(z)

    def trainable_layers(self):
        trainable_layers = {}
        cnt = 1
        for layer in self.layers:
            if isinstance(layer, Linear):
                trainable_layers[f"fc{cnt}"] = layer
                cnt += 1
        return trainable_layers

    def state_dict(self):
        state = {}
        for name, layer in self.trainable_layers().items():
            state[f"{name}_W"] = layer.W
            state[f"{name}_b"] = layer.b
        return state

    def load_state_dict(self, state):
        for name, layer in self.trainable_layers().items():
            layer.W = state[f"{name}_W"]
            layer.b = state[f"{name}_b"]

class Discriminator:
    def __init__(self, rng, input_dim, hidden_dims):
        self.rng = rng
        self.layers = []

        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims, dims[1:]):
            self.layers.append(Linear(in_dim=in_dim, out_dim=out_dim, rng=rng))
            self.layers.append(LeakyReLU())
        self.layers.append(Linear(in_dim=hidden_dims[-1], out_dim=1, rng=rng))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def trainable_layers(self):
        trainable_layers = {}
        cnt = 1
        for layer in self.layers:
            if isinstance(layer, Linear):
                trainable_layers[f"fc{cnt}"] = layer
                cnt += 1
        return trainable_layers

    def state_dict(self):
        state = {}
        for name, layer in self.trainable_layers().items():
            state[f"{name}_W"] = layer.W
            state[f"{name}_b"] = layer.b
        return state

    def load_state_dict(self, state):
        for name, layer in self.trainable_layers().items():
            layer.W = state[f"{name}_W"]
            layer.b = state[f"{name}_b"]

class GAN:
    def __init__(self, rng, input_dim, latent_dim, g_hidden_dims, d_hidden_dims):
        self.latent_dim = latent_dim
        self.generator = Generator(rng=rng, input_dim=input_dim, latent_dim=latent_dim, hidden_dims=g_hidden_dims)
        self.discriminator = Discriminator(rng=rng, input_dim=input_dim, hidden_dims=d_hidden_dims)

