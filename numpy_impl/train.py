import argparse
import numpy as np
import json
from pathlib import Path

from models import build_model
from trainer import build_trainer
from data import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"

def main():
    args = parse_args()
    cfg = load_config(cfg_path=args.config)

    rng = np.random.default_rng(seed=cfg["seed"])
    data, shape = load_dataset(cfg=cfg["dataset"])
    model = build_model(cfg=cfg["model"], rng=rng, input_dim=int(np.prod(shape)))
    trainer = build_trainer(cfg=cfg, model=model)
    trainer.train(data=data, shape=shape, rng=rng)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=CONFIG_PATH)
    return parser.parse_args()

def load_config(cfg_path=CONFIG_PATH):
    with open(cfg_path) as f:
        cfg = json.load(f)
        cfg["dataset"]["data_dir"] = resolve_path(cfg["dataset"]["data_dir"])
        cfg["sampling"]["sample_dir"] = resolve_path(cfg["sampling"]["sample_dir"])
        cfg["checkpoint"]["checkpoint_dir"] = resolve_path(cfg["checkpoint"]["checkpoint_dir"])
        cfg["checkpoint"]["load_checkpoint_path"] = resolve_path(cfg["checkpoint"]["load_checkpoint_path"])

    return cfg

def resolve_path(path):
    return str(PROJECT_ROOT / Path(path))

if __name__ == "__main__":
    main()