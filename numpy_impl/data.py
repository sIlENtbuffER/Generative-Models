import os
import struct
import numpy as np



def load_mnist_images(data_dir, low, high):
    path = os.path.join(data_dir, "train-images-idx3-ubyte")
    with open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
    images = images.reshape(num_images, rows*cols).astype(np.float32) / 255.0
    images = images * (high - low) + low
    image_shape = (rows, cols)
    return images, image_shape

DATASET_LOADERS = {
    "mnist" : load_mnist_images,
}

def load_dataset(cfg):
    name = cfg["name"]
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}")
    low, high = cfg["range"]
    return DATASET_LOADERS[name](data_dir=cfg["data_dir"], low=low, high=high)

def batches(x, batch_size, rng, shuffle=True):
    indices = np.arange(len(x))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(x), batch_size):
        batch_indices = indices[start:start+batch_size]
        yield x[batch_indices]
