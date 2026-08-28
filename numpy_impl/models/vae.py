import numpy as np
from modules import Linear, ReLU, Sigmoid

class VAE:
    def __init__(self, rng, input_dim=784, hidden_dim=400, latent_dim=20):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rng = rng

        # Encoder
        self.fc1 = Linear(in_dim=input_dim, out_dim=hidden_dim, rng=rng)
        self.relu1 = ReLU()
        self.fc_mu = Linear(in_dim=hidden_dim, out_dim=latent_dim, rng=rng)
        self.fc_logvar = Linear(in_dim=hidden_dim, out_dim=latent_dim, rng=rng)

        # Decoder
        self.fc2 = Linear(in_dim=latent_dim, out_dim=hidden_dim, rng=rng)
        self.relu2 = ReLU()
        self.fc3 = Linear(in_dim=hidden_dim, out_dim=input_dim, rng=rng)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Encode
        h = self.relu1.forward(self.fc1.forward(x=x))
        self.mu = self.fc_mu.forward(h)
        self.logvar = self.fc_logvar.forward(h)
        self.eps = self.rng.normal(loc=0.0, scale=1.0, size=self.mu.shape)
        z = self.mu + np.exp(0.5 * self.logvar) * self.eps

        # Decode
        x_hat = self.sigmoid.forward(self.fc3.forward(self.relu2.forward(self.fc2.forward(z))))
        return x_hat

    def backward(self, x, x_hat):
        batch_size = x.shape[0]
        x_hat = np.clip(x_hat, 1e-7, 1.0 - 1e-7)

        recon_loss = -np.sum(x * np.log(x_hat) + (1.0 - x) * np.log(1.0 - x_hat)) / batch_size
        kl_loss = 0.5 * np.sum(self.mu**2 + np.exp(self.logvar) - 1.0 - self.logvar) / batch_size

        # Decoder
        dx_hat = ((1.0 - x) / (1.0 - x_hat) - x / x_hat) / batch_size
        grad = self.sigmoid.backward(dx_hat)
        grad = self.fc3.backward(grad)
        grad = self.relu2.backward(grad)
        dz = self.fc2.backward(grad)

        # Reparameterization
        dmu = dz + self.mu / batch_size
        dlogvar = (dz * self.eps * 0.5 * np.exp(0.5 * self.logvar) + 0.5 * (np.exp(self.logvar) - 1.0) / batch_size)

        # Encoder
        dh = self.fc_mu.backward(dmu) + self.fc_logvar.backward(dlogvar)
        dh = self.relu1.backward(dh)
        self.fc1.backward(dh)

        return {
            "loss": recon_loss + kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def sample(self, num_samples):
        z = self.rng.normal(loc=0.0, scale=1.0, size=(num_samples, self.latent_dim))
        return self.sigmoid.forward(self.fc3.forward(self.relu2.forward(self.fc2.forward(z))))

    def trainable_layers(self):
        return {
            "fc1": self.fc1,
            "fc_mu": self.fc_mu,
            "fc_logvar": self.fc_logvar,
            "fc2": self.fc2,
            "fc3": self.fc3,
        }

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
