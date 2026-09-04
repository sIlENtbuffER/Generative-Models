import numpy as np

def he(rng, in_dim, out_dim):
    scale = np.sqrt(2.0 / in_dim)
    return rng.normal(loc=0.0, scale=scale, size=(in_dim, out_dim)).astype(np.float32)

def normal(rng, in_dim, out_dim, std=0.02):
    return rng.normal(loc=0.0, scale=std, size=(in_dim, out_dim)).astype(np.float32)
