import os
import matplotlib.pyplot as plt
import numpy as np
from data import batches
from safetensors import safe_open
from safetensors.numpy import save_file

class Trainer:
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = cfg["training"]["batch_size"]
        self.num_samples = cfg["sampling"]["num_samples"]
        self.sample_dir = cfg["sampling"]["sample_dir"]
        self.cpt_dir = cfg["checkpoint"]["checkpoint_dir"]
        self.cpt_path = cfg["checkpoint"]["load_checkpoint_path"] if cfg["checkpoint"]["load_checkpoint"] else None
        self.epochs=cfg["training"]["epochs"]

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.cpt_dir, exist_ok=True)

    def train(self, data, shape, rng, start_epoch=1):
        self.shape = shape

        if self.cpt_path:
            start_epoch = self.load_checkpoints() + 1
        
        for epoch in range(start_epoch, self.epochs + 1):
            metrics = self.train_epoch(data=data, rng=rng)
            print(f"Epoch {epoch:2d} | "
                f"loss: {metrics['loss']:.2f} | "
                f"recon: {metrics['recon_loss']:.2f} | "
                f"kl: {metrics['kl_loss']:.2f}"
            )

            self.save_samples(epoch)
            self.save_checkpoints(epoch)

    def train_epoch(self, data, rng):
        loss = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
        seen = 0

        for x in batches(x=data, batch_size=self.batch_size, rng=rng):
            x_hat = self.model.forward(x)
            metrics = self.model.backward(x=x, x_hat=x_hat)
            self.optimizer.step()

            batch_size = len(x)
            seen += batch_size 
            for name, value in metrics.items():
                loss[name] += value * batch_size

        return {key: value / seen for key, value in loss.items()}

    def save_samples(self, epoch):
        samples = self.model.sample(self.num_samples)
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
        path = os.path.join(self.cpt_dir, f"vae_epoch_{epoch}.safetensors")
        tensors = {}
        for name, value in self.model.state_dict().items():
            tensors[f"model.{name}"] = np.ascontiguousarray(value, dtype=np.float32)
        for name, value in self.optimizer.state_dict().items():
            tensors[f"optimizer.{name}"] = np.ascontiguousarray(value, dtype=np.float32)

        metadata = {"epoch": str(epoch)}
        if hasattr(self.optimizer, "t"):
            metadata["optimizer_step"] = str(self.optimizer.t)

        save_file(tensors, path, metadata=metadata)

    def load_checkpoints(self):
        with safe_open(self.cpt_path, framework="np") as cpt:
            tensors = {name: cpt.get_tensor(name) for name in cpt.keys()}
            metadata = cpt.metadata()

        model_state = {name[len("model."):]: value for name, value in tensors.items() if name.startswith("model.")}
        optimizer_state = {name[len("optimizer."):]: value for name, value in tensors.items() if name.startswith("optimizer.")}

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        if hasattr(self.optimizer, "t") and "optimizer_step" in metadata:
            self.optimizer.t = int(metadata["optimizer_step"])

        epoch = int(metadata["epoch"])
        print(f"Loaded model weights from {self.cpt_path} at epoch {epoch}")
        return epoch
