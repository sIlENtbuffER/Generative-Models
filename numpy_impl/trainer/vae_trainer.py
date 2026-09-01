from data import batches
from optimizer import build_optimizer
from .trainer import Trainer

class VAETrainer(Trainer):
    def __init__(self, model, cfg):
        self.model = model
        self.optimizer = build_optimizer(cfg=cfg["optimizer"], layers=model.trainable_layers())
        super().__init__(models={"model": model}, optimizers={"optimizer": self.optimizer}, cfg=cfg)

    def sample(self, num_samples):
        return self.model.sample(num_samples)

    def train_epoch(self, data, rng):
        loss = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
        seen = 0

        for x in batches(x=data, batch_size=self.batch_size, rng=rng, low=self.low, high=self.high):
            x_hat = self.model.forward(x)
            metrics = self.model.backward(x=x, x_hat=x_hat)
            self.optimizer.step()

            batch_size = len(x)
            seen += batch_size
            for name, value in metrics.items():
                loss[name] += value * batch_size

        return {key: value / seen for key, value in loss.items()}
