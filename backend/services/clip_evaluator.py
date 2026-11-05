"""UNIT 3 (CLIP): scores how well generated images match a text description.

Two numbers per image:
  similarity      cosine similarity between image and text embedding (typically 0.15-0.35)
  class_match_prob softmax over the 10 class prompts -> probability mass on the requested class
Also reports CLIP's top-1 class. IMPORTANT: this is a *similarity* signal from a model that
was not trained on 28x28 grayscale fashion pictures. It is NOT verified classification accuracy."""
import threading
from typing import List, Optional

import config

_lock = threading.Lock()
_state = {"model": None, "proc": None, "class_text": None, "error": None}


def _prompts():
    return [f"a photo of a {name}" for name in config.CLIP_LABELS]


def _ensure_loaded() -> bool:
    if _state["model"] is not None:
        return True
    if not config.ENABLE_CLIP or _state["error"]:
        return False
    with _lock:
        if _state["model"] is not None:
            return True
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            try:
                _state["model"] = CLIPModel.from_pretrained(config.CLIP_MODEL, local_files_only=True).eval()
                _state["proc"] = CLIPProcessor.from_pretrained(config.CLIP_MODEL, local_files_only=True)
            except Exception:
                _state["model"] = CLIPModel.from_pretrained(config.CLIP_MODEL).eval()
                _state["proc"] = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
            tok = _state["proc"](text=_prompts(), return_tensors="pt", padding=True)
            with torch.no_grad():
                t = _state["model"].get_text_features(**tok)
            _state["class_text"] = torch.nn.functional.normalize(t, dim=-1)
            return True
        except Exception as exc:
            _state["error"] = f"{type(exc).__name__}: {exc}"[:200]
            return False


def status() -> dict:
    return {"enabled": config.ENABLE_CLIP, "model": config.CLIP_MODEL,
            "loaded": _state["model"] is not None, "error": _state["error"]}


def evaluate(pil_images: list, class_id: int) -> Optional[List[dict]]:
    """pil_images: list of RGB PIL images. Returns one dict per image, or None if CLIP unavailable."""
    if not _ensure_loaded():
        return None
    import torch
    model, proc, class_text = _state["model"], _state["proc"], _state["class_text"]
    inputs = proc(images=pil_images, return_tensors="pt")
    with torch.no_grad():
        feats = torch.nn.functional.normalize(model.get_image_features(**inputs), dim=-1)
        sims = feats @ class_text.T                              # [N, 10] cosine similarities
        probs = torch.softmax(sims * model.logit_scale.exp(), dim=-1)
    out = []
    for i in range(len(pil_images)):
        top = int(probs[i].argmax())
        out.append({"similarity": round(float(sims[i, class_id]), 4),
                    "class_match_prob": round(float(probs[i, class_id]), 4),
                    "top1_class_id": top, "top1_category": config.CLASS_NAMES[top],
                    "top1_matches_request": top == class_id})
    return out


def summarize(scores: Optional[List[dict]]) -> Optional[dict]:
    if not scores:
        return None
    n = len(scores)
    return {"mean_similarity": round(sum(s["similarity"] for s in scores) / n, 4),
            "mean_class_match_prob": round(sum(s["class_match_prob"] for s in scores) / n, 4),
            "top1_agreement": round(sum(s["top1_matches_request"] for s in scores) / n, 4),
            "n": n}
