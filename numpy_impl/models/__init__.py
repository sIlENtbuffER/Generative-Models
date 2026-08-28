from .vae import VAE

__all__ = [
    "VAE",
]

MODELS = {
    "vae": VAE,
}

def build_model(cfg, rng, input_dim):
    name = cfg["name"]
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    return MODELS[name](rng=rng, **{k:v for k, v in cfg.items() if k != "name"}, input_dim=input_dim)
