"""Loads trained weights once and caches them (important on small free servers)."""
import threading
import torch

import config
from models.generator import Generator
from models.discriminator import Discriminator
from models.vae import ConditionalVAE


class ModelNotTrainedError(RuntimeError):
    """Raised when a weights file is missing so the API can answer 503 with a hint."""


class ModelRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}
        self.device = torch.device("cpu")   # inference on tiny models: CPU is plenty
        torch.set_num_threads(config.TORCH_THREADS)

    def _load(self, key, path, builder, hint):
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if not path.exists():
                raise ModelNotTrainedError(
                    f"Weights not found: {path.name}. Train first: {hint}")
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            model = builder(ckpt["config"])
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device).eval()
            self._cache[key] = model
            return model

    def generator(self):
        return self._load("gan", config.GAN_WEIGHTS, lambda c: Generator(**c),
                          "python -m training.train_gan")

    def discriminator(self):
        return self._load("disc", config.DISC_WEIGHTS, lambda c: Discriminator(**c),
                          "python -m training.train_gan")

    def vae(self):
        return self._load("vae", config.VAE_WEIGHTS, lambda c: ConditionalVAE(**c),
                          "python -m training.train_vae")

    def is_loaded(self, key):
        return key in self._cache

    def reload(self):
        with self._lock:
            self._cache.clear()


registry = ModelRegistry()
