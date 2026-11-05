"""Fashion-MNIST loading. Uses torchvision (auto-download); if that host is blocked
(some campus networks), falls back to the official GitHub copy of the same files."""
import gzip
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config

GITHUB_RAW = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {"train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
         "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")}


def _github_fallback(split):
    folder = config.DATA_DIR / "github_fashion"
    folder.mkdir(parents=True, exist_ok=True)
    arrays = []
    for i, name in enumerate(FILES[split]):
        path = folder / name
        if not path.exists():
            urllib.request.urlretrieve(GITHUB_RAW + name, path)
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if i == 0 else 8)
        arrays.append(data)
    images = torch.from_numpy(arrays[0].copy()).view(-1, 1, 28, 28)
    labels = torch.from_numpy(arrays[1].copy()).long()
    return images, labels


def load_split(split="train"):
    """Returns (images uint8 [N,1,28,28], labels int64 [N])."""
    try:
        from torchvision import datasets
        ds = datasets.FashionMNIST(str(config.DATA_DIR), train=(split == "train"), download=True)
        return ds.data.unsqueeze(1).clone(), ds.targets.clone()
    except Exception as exc:  # network / mirror problems
        print(f"[dataset] torchvision download failed ({exc}); using GitHub copy")
        return _github_fallback(split)


def get_loader(split="train", batch_size=128, scale="tanh", shuffle=True):
    """scale='tanh' -> [-1,1] (GAN); scale='unit' -> [0,1] (VAE)."""
    images, labels = load_split(split)
    x = images.float() / 255.0
    if scale == "tanh":
        x = x * 2 - 1
    return DataLoader(TensorDataset(x, labels), batch_size=batch_size,
                      shuffle=shuffle, drop_last=shuffle)
