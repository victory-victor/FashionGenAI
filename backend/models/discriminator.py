"""Conditional GAN discriminator: (image, label) -> real/fake logit."""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_size=28):
        super().__init__()
        self.num_classes, self.img_size = num_classes, img_size
        # The label is embedded to a 28x28 map and stacked as a 2nd image channel
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),       # 28 -> 14
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),                                  # 14 -> 7
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128 * 7 * 7, 1))                                # raw logit

    def forward(self, img, labels):
        lab = self.label_embedding(labels).view(-1, 1, self.img_size, self.img_size)
        return self.net(torch.cat([img, lab], dim=1)).squeeze(1)

    def config(self):
        return {"num_classes": self.num_classes, "img_size": self.img_size}
