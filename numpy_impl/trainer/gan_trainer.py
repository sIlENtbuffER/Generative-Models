from data import batches
from optimizer import build_optimizer
from .trainer import Trainer

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
        ...
