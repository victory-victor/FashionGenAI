"""Train the Conditional VAE on Fashion-MNIST.
    python -m training.train_vae --epochs 30"""
import argparse
import json
import time
from datetime import datetime, timezone

import torch
from torchvision.utils import save_image

import config
from models.vae import ConditionalVAE, vae_loss
from training.dataset import get_loader


def main(epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(config.TORCH_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = get_loader("train", batch_size, "unit"), get_loader("test", 1000, "unit", False)
    model = ConditionalVAE(config.VAE_LATENT_DIM, config.NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = {"model": "Conditional VAE", "device": str(device), "total_epochs": epochs,
               "completed_epochs": 0, "batch_size": batch_size, "learning_rate": lr, "seed": seed,
               "train_images": len(train.dataset), "test_images": len(test.dataset),
               "started_at": datetime.now(timezone.utc).isoformat(), "epochs": [], "samples": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot = rec = kl_sum = n = 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x, y)
            loss, bce, kl = vae_loss(recon, x, mu, logvar)
            loss.backward(); opt.step()
            b = x.size(0); tot += loss.item() * b; rec += bce.item() * b; kl_sum += kl.item() * b; n += b
        model.eval()
        with torch.no_grad():   # held-out test loss
            tl = tn = 0
            for x, y in test:
                x, y = x.to(device), y.to(device)
                r, mu, lv = model(x, y)
                tl += vae_loss(r, x, mu, lv)[0].item() * x.size(0); tn += x.size(0)
        row = {"epoch": epoch, "loss": tot / n, "recon_loss": rec / n, "kl_loss": kl_sum / n,
               "test_loss": tl / tn, "seconds": time.time() - t0}
        metrics["epochs"].append(row); metrics["completed_epochs"] = epoch
        if epoch in (1, epochs) or epoch % 5 == 0:
            name = f"vae_epoch_{epoch:03d}.png"
            labels = torch.arange(10, device=device).repeat_interleave(10)
            save_image(model.sample(labels), config.SAMPLES_DIR / name, nrow=10, padding=2, pad_value=1)
            metrics["samples"].append({"epoch": epoch, "file": name})
        print(f"epoch {epoch}/{epochs}  loss {row['loss']:.2f}  test {row['test_loss']:.2f}")
        torch.save({"state_dict": model.state_dict(), "config": model.config()}, config.VAE_WEIGHTS)
        metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
        config.VAE_METRICS.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.epochs, a.batch_size, a.lr, a.seed)
