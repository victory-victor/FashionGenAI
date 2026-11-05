"""Conditional GAN generator: noise z + class label -> 28x28 grayscale image."""
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, embed_dim=50):
        super().__init__()
        self.latent_dim, self.num_classes, self.embed_dim = latent_dim, num_classes, embed_dim
        # Conditioning: each class id becomes a learned vector that is joined to the noise
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),            # 7 -> 14
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),              # 14 -> 28
            nn.Tanh())                                       # output in [-1, 1]

    def forward(self, z, labels):
        x = torch.cat([z, self.label_embedding(labels)], dim=1)
        x = self.project(x).view(-1, 128, 7, 7)
        return self.net(x)

    def config(self):
        return {"latent_dim": self.latent_dim, "num_classes": self.num_classes,
                "embed_dim": self.embed_dim}
