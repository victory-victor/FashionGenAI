"""Train the Conditional GAN on Fashion-MNIST.
    python -m training.train_gan --epochs 30
Writes: weights (outputs/saved_models), per-epoch losses (outputs/metrics/gan_metrics.json)
and sample grids (outputs/samples). These files are what the dashboard shows."""
import argparse
import json
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torchvision.utils import save_image

import config
from models.generator import Generator
from models.discriminator import Discriminator
from training.dataset import get_loader


def class_grid(gen, device, path, per_class=10):
    """One row per class, so you can see conditional generation at a glance."""
    gen.eval()
    labels = torch.arange(10, device=device).repeat_interleave(per_class)
    with torch.no_grad():
        imgs = gen(torch.randn(len(labels), config.LATENT_DIM, device=device), labels)
    save_image((imgs + 1) / 2, path, nrow=per_class, padding=2, pad_value=1)
    gen.train()


def main(epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(config.TORCH_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_loader("train", batch_size, scale="tanh")
    test_size = len(get_loader("test", 1000, scale="tanh", shuffle=False).dataset)

    G = Generator(config.LATENT_DIM, config.NUM_CLASSES).to(device)
    D = Discriminator(config.NUM_CLASSES).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    metrics = {"model": "Conditional GAN", "device": str(device), "total_epochs": epochs,
               "completed_epochs": 0, "batch_size": batch_size, "learning_rate": lr, "seed": seed,
               "train_images": len(loader.dataset), "test_images": test_size,
               "started_at": datetime.now(timezone.utc).isoformat(), "epochs": [], "samples": []}

    for epoch in range(1, epochs + 1):
        t0, g_sum, d_sum, d_real_sum, d_fake_sum, n = time.time(), 0.0, 0.0, 0.0, 0.0, 0
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            b = real.size(0)
            ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)
            fake_labels = torch.randint(0, 10, (b,), device=device)
            fake = G(torch.randn(b, config.LATENT_DIM, device=device), fake_labels)

            # 1) Discriminator: real pairs -> 1, fake pairs -> 0
            opt_d.zero_grad()
            out_real, out_fake = D(real, labels), D(fake.detach(), fake_labels)
            d_loss = bce(out_real, ones * 0.9) + bce(out_fake, zeros)   # 0.9 = label smoothing
            d_loss.backward()
            opt_d.step()

            # 2) Generator: make D say "real" for its fakes
            opt_g.zero_grad()
            g_loss = bce(D(fake, fake_labels), ones)
            g_loss.backward()
            opt_g.step()

            g_sum += g_loss.item() * b; d_sum += d_loss.item() * b; n += b
            d_real_sum += torch.sigmoid(out_real).mean().item() * b
            d_fake_sum += torch.sigmoid(out_fake).mean().item() * b

        row = {"epoch": epoch, "g_loss": g_sum / n, "d_loss": d_sum / n,
               "d_real_score": d_real_sum / n, "d_fake_score": d_fake_sum / n,
               "seconds": time.time() - t0}
        metrics["epochs"].append(row)
        metrics["completed_epochs"] = epoch
        if epoch in (1, epochs) or epoch % 5 == 0:
            name = f"gan_epoch_{epoch:03d}.png"
            class_grid(G, device, config.SAMPLES_DIR / name)
            metrics["samples"].append({"epoch": epoch, "file": name})
        print(f"epoch {epoch}/{epochs}  G {row['g_loss']:.3f}  D {row['d_loss']:.3f}  "
              f"{row['seconds']:.0f}s")

        # Save every epoch so a stopped run still leaves usable weights
        torch.save({"state_dict": G.state_dict(), "config": G.config()}, config.GAN_WEIGHTS)
        torch.save({"state_dict": D.state_dict(), "config": D.config()}, config.DISC_WEIGHTS)
        metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
        config.GAN_METRICS.write_text(json.dumps(metrics, indent=2))
    print("Saved:", config.GAN_WEIGHTS)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.epochs, a.batch_size, a.lr, a.seed)
