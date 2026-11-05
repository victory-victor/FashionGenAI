"""REST endpoints. Each handler is thin: validate -> call a service -> shape the response."""
import json
import time
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

import config
from models.model_loader import registry, ModelNotTrainedError
from services import bert_service, clip_evaluator, generation_service as gen
from services.llm_parser import parse_prompt, ParseError

router = APIRouter(prefix="/api")
START_TIME = time.time()


class ParseRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=300)
    strategy: str = Field("few_shot", pattern="^(zero_shot|few_shot)$")
    compare: bool = False


class GenerateRequest(BaseModel):
    prompt: Optional[str] = Field(None, max_length=300)
    class_id: Optional[int] = Field(None, ge=0, le=9)      # from the category selector
    count: Optional[int] = Field(None, ge=1, le=config.MAX_IMAGES)
    evaluate: bool = True
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)


def _resolve(req: GenerateRequest) -> dict:
    """Selector values win; otherwise parse the text prompt."""
    if req.prompt and req.class_id is None:
        try:
            parsed = parse_prompt(req.prompt)
        except ParseError as exc:
            raise HTTPException(422, str(exc))
        if req.count:                       # explicit count from the UI overrides text
            parsed["count"] = req.count
        return parsed
    if req.class_id is None:
        raise HTTPException(422, "Provide a prompt or a class_id.")
    return {"category": config.CLASS_NAMES[req.class_id].lower(), "class_id": req.class_id,
            "count": req.count or 1, "source": "category selector", "prompt": req.prompt}


def _run(model: str, req: GenerateRequest) -> dict:
    parsed = _resolve(req)
    try:
        result = gen.generate(model, parsed["class_id"], parsed["count"], req.evaluate, req.seed)
    except ModelNotTrainedError as exc:
        raise HTTPException(503, str(exc))
    result["parsed"] = parsed
    if parsed.get("prompt") and config.ENABLE_BERT:
        result["bert_check"] = bert_service.classify(parsed["prompt"])
    return result


@router.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - START_TIME, 1),
            "cgan_ready": config.GAN_WEIGHTS.exists(), "vae_ready": config.VAE_WEIGHTS.exists(),
            "llm_configured": bool(config.GEMINI_API_KEY)}


@router.get("/categories")
def categories():
    return {"categories": [{"class_id": i, "name": n, "clip_label": config.CLIP_LABELS[i]}
                           for i, n in enumerate(config.CLASS_NAMES)], "max_images": config.MAX_IMAGES}


@router.post("/parse-prompt")
def parse(req: ParseRequest):
    try:
        result = parse_prompt(req.prompt, req.strategy, req.compare)
    except ParseError as exc:
        raise HTTPException(422, str(exc))
    result["bert_check"] = bert_service.classify(req.prompt) if config.ENABLE_BERT else None
    return result


@router.post("/generate")
def generate(req: GenerateRequest):
    return _run("cgan", req)


@router.post("/generate-vae")
def generate_vae(req: GenerateRequest):
    return _run("vae", req)


@router.get("/samples")
def samples():
    """One real cGAN image per class for the home-page gallery (no CLIP, fast)."""
    try:
        out = []
        for cid in range(10):
            r = gen.generate("cgan", cid, 1, evaluate=False, seed=7 + cid)
            out.append({"class_id": cid, "category": r["category"], "data_url": r["images"][0]["data_url"]})
        return {"samples": out}
    except ModelNotTrainedError as exc:
        raise HTTPException(503, str(exc))


@router.get("/compare")
def compare(class_id: int, count: int = 6, seed: int = 11):
    """Same class + same seed through both models, side by side."""
    if not 0 <= class_id <= 9 or not 1 <= count <= config.MAX_IMAGES:
        raise HTTPException(422, "class_id must be 0-9 and count 1-%d" % config.MAX_IMAGES)
    try:
        return {"cgan": gen.generate("cgan", class_id, count, True, seed),
                "vae": gen.generate("vae", class_id, count, True, seed)}
    except ModelNotTrainedError as exc:
        raise HTTPException(503, str(exc))


@router.get("/batch/{batch_id}/zip")
def download_zip(batch_id: str):
    data = gen.batch_zip(batch_id)
    if data is None:
        raise HTTPException(404, "Batch not found (it may have been cleaned up).")
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="fashiongen_{batch_id[:8]}.zip"'})


def _read_json(path, hint):
    if not path.exists():
        raise HTTPException(404, hint)
    return json.loads(path.read_text())


@router.get("/training-metrics")
def training_metrics():
    out = {}
    for key, path in (("cgan", config.GAN_METRICS), ("vae", config.VAE_METRICS)):
        out[key] = json.loads(path.read_text()) if path.exists() else None
    if out["cgan"] is None and out["vae"] is None:
        raise HTTPException(404, "No training metrics yet. Run: python -m training.train_gan && python -m training.train_vae")
    out["sample_base_url"] = "/files/samples/"
    return out


@router.get("/evaluation")
def evaluation():
    return _read_json(config.EVAL_RESULTS, "No evaluation results yet. Run: python -m services.evaluation_service")


@router.get("/model-info")
def model_info():
    def count(m): return sum(p.numel() for p in m.parameters())
    info = {"dataset": {"name": "Fashion-MNIST", "train": 60000, "test": 10000, "image_size": "28x28 grayscale",
                        "classes": config.CLASS_NAMES, "license": "MIT (Zalando Research)"},
            "device": str(registry.device), "clip": clip_evaluator.status(), "bert": bert_service.status(),
            "llm": {"provider": "Google Gemini", "model": config.GEMINI_MODEL,
                    "configured": bool(config.GEMINI_API_KEY), "fallback": "rule-based parser"}}
    for key, path, getter in (("cgan_generator", config.GAN_WEIGHTS, registry.generator),
                              ("cgan_discriminator", config.DISC_WEIGHTS, registry.discriminator),
                              ("cvae", config.VAE_WEIGHTS, registry.vae)):
        entry = {"weights_file": path.name, "weights_present": path.exists()}
        if path.exists():
            m = getter()
            entry.update(parameters=count(m), architecture=str(m), config=m.config(),
                         size_kb=round(path.stat().st_size / 1024, 1))
        info[key] = entry
    return info
