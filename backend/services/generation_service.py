"""UNIT 4: image generation service (cGAN + VAE inference, image utilities, file handling)."""
import base64
import io
import re
import time
import uuid
import zipfile
from typing import List, Optional

import torch
from PIL import Image

import config
from models.model_loader import registry
from services import clip_evaluator

BATCH_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_STORED_BATCHES = 50


# ---- image utilities ----------------------------------------------------------
def tensor_to_pil(t: torch.Tensor, size: int = config.DISPLAY_SIZE) -> Image.Image:
    """t: [1,28,28] in [0,1] -> RGB PIL image upscaled for display / CLIP."""
    arr = (t.clamp(0, 1).squeeze(0) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode="L").resize((size, size), Image.BICUBIC).convert("RGB")


def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- raw inference ------------------------------------------------------------
@torch.no_grad()
def sample_gan(class_id: int, count: int, seed: Optional[int] = None) -> torch.Tensor:
    gen = registry.generator()
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    z = torch.randn(count, gen.latent_dim, generator=g)
    labels = torch.full((count,), class_id, dtype=torch.long)
    return (gen(z, labels) + 1) / 2          # [-1,1] -> [0,1]


@torch.no_grad()
def sample_vae(class_id: int, count: int, seed: Optional[int] = None) -> torch.Tensor:
    vae = registry.vae()
    if seed is not None:
        torch.manual_seed(seed)
    return vae.sample(torch.full((count,), class_id, dtype=torch.long))


# ---- storage ------------------------------------------------------------------
def _prune():
    batches = sorted([p for p in config.GENERATED_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    for old in batches[:-MAX_STORED_BATCHES]:
        for f in old.iterdir():
            f.unlink()
        old.rmdir()


def _store(batch_id: str, images: List[Image.Image], slug: str):
    folder = config.GENERATED_DIR / batch_id
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for i, img in enumerate(images, 1):
        name = f"{slug}_{i}.png"
        img.save(folder / name)
        names.append(name)
    _prune()
    return names


def batch_zip(batch_id: str) -> Optional[bytes]:
    if not BATCH_ID_RE.match(batch_id):
        return None
    folder = config.GENERATED_DIR / batch_id
    if not folder.is_dir():
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.glob("*.png")):
            z.write(f, f.name)
    return buf.getvalue()


# ---- public API used by routes ------------------------------------------------
def generate(model: str, class_id: int, count: int, evaluate: bool = True,
             seed: Optional[int] = None) -> dict:
    t0 = time.perf_counter()
    tensors = sample_gan(class_id, count, seed) if model == "cgan" else sample_vae(class_id, count, seed)
    pil = [tensor_to_pil(t) for t in tensors]
    gen_ms = (time.perf_counter() - t0) * 1000

    scores, clip_ms = None, None
    if evaluate:
        t1 = time.perf_counter()
        scores = clip_evaluator.evaluate(pil, class_id)
        clip_ms = (time.perf_counter() - t1) * 1000

    batch_id = uuid.uuid4().hex
    slug = config.CLASS_NAMES[class_id].lower().replace("/", "-").replace(" ", "_")
    names = _store(batch_id, pil, f"{model}_{slug}")
    images = [{"index": i + 1, "filename": names[i], "data_url": pil_to_data_url(pil[i]),
               "clip": scores[i] if scores else None} for i in range(len(pil))]
    return {"model": model, "batch_id": batch_id, "class_id": class_id,
            "category": config.CLASS_NAMES[class_id], "count": count, "images": images,
            "clip_summary": clip_evaluator.summarize(scores), "clip_status": clip_evaluator.status(),
            "timing_ms": {"generation": round(gen_ms, 1),
                          "clip": round(clip_ms, 1) if clip_ms is not None else None}}
