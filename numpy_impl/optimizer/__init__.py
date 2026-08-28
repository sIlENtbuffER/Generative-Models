from .sgd import SGD
from .adam import Adam

__all__ = [
    "SGD",
    "Adam",
]

OPTIMIZERS = {
    "sgd": SGD,
    "adam": Adam,
}


def build_optimizer(cfg, layers):
    name = cfg["name"]
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZERS[name](layers=layers, **{k:v for k, v in cfg.items() if k != "name"})
