import numpy as np
from modules import Linear, LeakyReLU, Sigmoid, Tanh

class Generator:
    def __init__(self, rng, input_dim, latent_dim, hidden_dims):
        ...

    def forward(self, z):
        ...

    def backward(self, grad):
        ...

    def sample(self, num_samples):
        ...

    def trainable_layers(self):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state):
        ...

class Discriminator:
    def __init__(self, rng, input_dim, hidden_dims):
        ...

    def forward(self, x):
        ...

    def backward(self, grad):
        ...

    def trainable_layers(self):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state):
        ...

class GAN:
    def __init__(self, rng, input_dim, latent_dim, g_hidden_dims, d_hidden_dims):
        self.latent_dim = latent_dim
        self.generator = Generator(rng=rng, input_dim=input_dim, latent_dim=latent_dim, hidden_dims=g_hidden_dims)
        self.discriminator = Discriminator(rng=rng, input_dim=input_dim, hidden_dims=d_hidden_dims)
