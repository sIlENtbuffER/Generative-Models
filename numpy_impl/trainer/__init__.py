from .trainer import Trainer
from .vae_trainer import VAETrainer
from .gan_trainer import GANTrainer

__all__ = [
    "Trainer",
    "VAETrainer",
    "GANTrainer",
]

TRAINERS = {
    "vae": VAETrainer,
    "gan": GANTrainer,
}

def build_trainer(cfg, model):
    name = cfg["model"]["name"]
    if name not in TRAINERS:
        raise ValueError(f"Unknown trainer: {name}")
    return TRAINERS[name](model=model, cfg=cfg)
