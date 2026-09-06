import os
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

class Trainer:
    def __init__(self, models, optimizers, cfg):
        self.models = models
        self.optimizers = optimizers
        self.batch_size = cfg["training"]["batch_size"]
        self.num_samples = cfg["sampling"]["num_samples"]
        self.sample_dir = cfg["sampling"]["sample_dir"]
        self.cpt_dir = cfg["checkpoint"]["checkpoint_dir"]
        self.cpt_path = cfg["checkpoint"]["load_checkpoint_path"] if cfg["checkpoint"]["load_checkpoint"] else None
        self.epochs = cfg["training"]["epochs"]
        self.low, self.high = cfg["dataset"]["range"]

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.cpt_dir, exist_ok=True)

    def train(self, data, shape, rng, start_epoch=1):
        self.shape = shape

        if self.cpt_path:
            start_epoch = self.load_checkpoints() + 1

        for epoch in range(start_epoch, self.epochs + 1):
            metrics = self.train_epoch(data=data, rng=rng)
            metrics_str = " | ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
            print(f"Epoch {epoch:2d} | {metrics_str}")

            self.save_samples(epoch)
            self.save_checkpoints(epoch)

    def save_samples(self, epoch):
        samples = (self.sample(self.num_samples) - self.low) / (self.high - self.low)
        cols = int(np.ceil(np.sqrt(self.num_samples)))
        rows = int(np.ceil(self.num_samples / cols))
        images = samples.reshape(self.num_samples, *self.shape)
        gray = self.shape[0] == 1
        images = images[:, 0] if gray else images.transpose(0, 2, 3, 1) # N,H,W,C
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(rows, cols))
        for image, ax in zip(images, axes.flat):
            ax.imshow(image, cmap="gray" if gray else None, vmin=0.0, vmax=1.0)
            ax.axis("off")

        path = os.path.join(self.sample_dir, f"epoch_{epoch}.png")
        fig.tight_layout(pad=0.1)
        fig.savefig(path)
        plt.close(fig)

    def save_checkpoints(self, epoch):
        path = os.path.join(self.cpt_dir, f"epoch_{epoch}.safetensors")
        tensors = {}
        for model_name, model in self.models.items():
            for name, value in model.state_dict().items():
                tensors[f"{model_name}.{name}"] = np.ascontiguousarray(value, dtype=np.float32)

        metadata = {"epoch": str(epoch)}
        for opt_name, optimizer in self.optimizers.items():
            for name, value in optimizer.state_dict().items():
                tensors[f"{opt_name}.{name}"] = np.ascontiguousarray(value, dtype=np.float32)
            if hasattr(optimizer, "t"):
                metadata[f"{opt_name}_step"] = str(optimizer.t)

        save_file(tensors, path, metadata=metadata)

    def load_checkpoints(self):
        with safe_open(self.cpt_path, framework="np") as cpt:
            tensors = {name: cpt.get_tensor(name) for name in cpt.keys()}
            metadata = cpt.metadata()

        for model_name, model in self.models.items():
            model.load_state_dict(select(tensors=tensors, prefix=model_name))
        for opt_name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(select(tensors=tensors, prefix=opt_name))
            if hasattr(optimizer, "t") and f"{opt_name}_step" in metadata:
                optimizer.t = int(metadata[f"{opt_name}_step"])

        epoch = int(metadata["epoch"])
        print(f"Loaded model weights from {self.cpt_path} at epoch {epoch}")
        return epoch

def select(tensors, prefix):
    return {name[len(prefix) + 1:]: value for name, value in tensors.items() if name.startswith(f"{prefix}.")}
