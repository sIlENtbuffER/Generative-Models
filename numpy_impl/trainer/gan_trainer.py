from data import batches
from optimizer import build_optimizer
from .trainer import Trainer
from modules import Sigmoid
import numpy as np

class GANTrainer(Trainer):
    def __init__(self, model, cfg):
        self.generator = model.generator
        self.discriminator = model.discriminator
        self.g_optimizer = build_optimizer(cfg=cfg["optimizer"], layers=self.generator.trainable_layers())
        self.d_optimizer = build_optimizer(cfg=cfg["optimizer"], layers=self.discriminator.trainable_layers())
        super().__init__(
            models={"generator": self.generator, "discriminator": self.discriminator},
            optimizers={"generator": self.g_optimizer, "discriminator": self.d_optimizer},
            cfg=cfg,
        )

    def sample(self, num_samples):
        return self.generator.sample(num_samples)

    def train_epoch(self, data, rng):
        loss = {"d_loss": 0.0, "g_loss": 0.0} 
        seen = 0
        sigmoid = Sigmoid()

        for x_real in batches(x=data, batch_size=self.batch_size, rng=rng, low=self.low, high=self.high):
            batch_size = len(x_real)
            seen += batch_size

            z = rng.normal(loc=0.0, scale=1.0, size=(batch_size, self.generator.latent_dim)).astype(np.float32)
            x_fake = self.generator.forward(z)
            x_all = np.concatenate([x_fake, x_real], axis=0)
            y = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))], axis=0)
            logits = self.discriminator.forward(x_all)
            p = sigmoid.forward(logits)

            # D step
            logits = self.discriminator.forward(x_all)
            loss["d_loss"] += np.mean(np.logaddexp(0, logits) - logits*y) * batch_size
            self.discriminator.backward((p - y) / (2 * batch_size))
            self.d_optimizer.step()

            # G step
            logits = self.discriminator.forward(x_fake)
            p = sigmoid.forward(logits)
            loss["g_loss"] += np.mean(np.logaddexp(0, logits) - logits) * batch_size
            dx = self.discriminator.backward((p - 1) / batch_size)
            self.generator.backward(dx)
            self.g_optimizer.step()

        return {key: value / seen for key, value in loss.items()}
