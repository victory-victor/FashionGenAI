"""UNIT 3 (BERT): zero-shot understanding with a BERT-family encoder (DistilBERT).

We embed the user's request and a short description of each Fashion-MNIST class,
then rank classes by cosine similarity. No fine-tuning is done: this shows how a
pretrained encoder maps text to a shared embedding space. It is used as an
independent cross-check of the LLM / rule-based category, not as the main parser."""
import threading
from typing import Optional

import config

_lock = threading.Lock()
_state = {"model": None, "tok": None, "class_emb": None, "error": None}

DESCRIPTIONS = ["t-shirt or short sleeve top", "trousers pants jeans", "pullover sweater hoodie",
                "dress gown", "coat jacket outerwear", "open sandal footwear",
                "button shirt blouse", "sneaker running shoe trainer", "handbag purse tote bag",
                "ankle boot boots footwear"]


def _embed(texts):
    import torch
    tok, model = _state["tok"], _state["model"]
    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    pooled = (out * mask).sum(1) / mask.sum(1)            # mean pooling over tokens
    return torch.nn.functional.normalize(pooled, dim=1)


def _ensure_loaded() -> bool:
    if _state["model"] is not None:
        return True
    if not config.ENABLE_BERT or _state["error"]:
        return False
    with _lock:
        if _state["model"] is not None:
            return True
        try:
            from transformers import AutoModel, AutoTokenizer
            try:
                _state["tok"] = AutoTokenizer.from_pretrained(config.BERT_MODEL, local_files_only=True)
                _state["model"] = AutoModel.from_pretrained(config.BERT_MODEL, local_files_only=True).eval()
            except Exception:
                _state["tok"] = AutoTokenizer.from_pretrained(config.BERT_MODEL)
                _state["model"] = AutoModel.from_pretrained(config.BERT_MODEL).eval()
            _state["class_emb"] = _embed(DESCRIPTIONS)
            return True
        except Exception as exc:
            _state["error"] = f"{type(exc).__name__}: {exc}"[:200]
            return False


def status() -> dict:
    return {"enabled": config.ENABLE_BERT, "model": config.BERT_MODEL,
            "loaded": _state["model"] is not None, "error": _state["error"]}


def classify(text: str) -> Optional[dict]:
    """Returns top-3 classes by embedding similarity, or None if BERT is unavailable."""
    if not _ensure_loaded():
        return None
    import torch
    sims = (_embed([text]) @ _state["class_emb"].T).squeeze(0)
    probs = torch.softmax(sims * 20, dim=0)               # temperature-scaled for readability
    top = torch.topk(probs, 3)
    return {"model": config.BERT_MODEL, "method": "zero-shot embedding similarity",
            "top": [{"class_id": int(i), "category": config.CLASS_NAMES[int(i)], "score": round(float(p), 4)}
                    for p, i in zip(top.values, top.indices)]}
