"""Conditional Variational Autoencoder (CVAE).
Conditioning on the class label lets us ask the VAE for 'ankle boot' exactly
like the cGAN, so the two models can be compared fairly."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10, hidden=512):
        super().__init__()
        self.latent_dim, self.num_classes, self.hidden = latent_dim, num_classes, hidden
        self.enc = nn.Sequential(nn.Linear(784 + num_classes, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim + num_classes, 256), nn.ReLU(),
                                 nn.Linear(256, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 784), nn.Sigmoid())   # pixels in [0, 1]

    def _onehot(self, labels):
        return F.one_hot(labels, self.num_classes).float()

    def encode(self, x, labels):
        h = self.enc(torch.cat([x.view(-1, 784), self._onehot(labels)], dim=1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps  -> keeps sampling differentiable
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z, labels):
        return self.dec(torch.cat([z, self._onehot(labels)], dim=1)).view(-1, 1, 28, 28)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        return self.decode(self.reparameterize(mu, logvar), labels), mu, logvar

    @torch.no_grad()
    def sample(self, labels):
        z = torch.randn(len(labels), self.latent_dim, device=labels.device)
        return self.decode(z, labels)

    def config(self):
        return {"latent_dim": self.latent_dim, "num_classes": self.num_classes,
                "hidden": self.hidden}


def vae_loss(recon, x, mu, logvar):
    """ELBO = reconstruction (BCE) + KL divergence to N(0, I). Returns per-sample means."""
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    n = x.size(0)
    return (bce + kl) / n, bce / n, kl / n
