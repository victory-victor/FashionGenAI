"""UNIT 5: measurable experiment. Runs real generation + CLIP and writes results to disk.
    python -m services.evaluation_service
No number in the evaluation dashboard exists unless this script computed it."""
import json
import time
from datetime import datetime, timezone
from statistics import mean

import config
from services import clip_evaluator
from services.generation_service import sample_gan, sample_vae, tensor_to_pil
from services.llm_parser import parse_prompt, ParseError

# Ten prompts: (prompt, expected class id). Expected ids are written by hand = ground truth for the parser.
TEST_PROMPTS = [
    ("Show me four ankle boots", 9), ("Generate three sneakers", 7), ("Create five handbags", 8),
    ("Show me two dresses", 3), ("Make four t-shirts", 0), ("I need three coats", 4),
    ("Generate two trousers", 1), ("Show me four pullovers", 2), ("Create three sandals", 5),
    ("Show me two shirts", 6),
]
PER_CLASS_IMAGES = 16        # images per class for class-wise CLIP analysis
EVAL_SEED = 1234


def _model_scores(sampler, cid, n, seed):
    t0 = time.perf_counter()
    tensors = sampler(cid, n, seed)
    gen_ms = (time.perf_counter() - t0) * 1000
    scores = clip_evaluator.evaluate([tensor_to_pil(t) for t in tensors], cid)
    return scores, gen_ms


def run() -> dict:
    if not clip_evaluator._ensure_loaded():
        raise SystemExit(f"CLIP unavailable: {clip_evaluator.status()['error']}")
    prompt_rows, failures = [], []
    for prompt, expected in TEST_PROMPTS:
        t0 = time.perf_counter()
        try:
            parsed = parse_prompt(prompt, "few_shot")
        except ParseError as exc:
            prompt_rows.append({"prompt": prompt, "requested_class_id": expected, "error": str(exc)})
            failures.append({"prompt": prompt, "reason": f"parse error: {exc}"}); continue
        cid, n = parsed["class_id"], parsed["count"]
        g_scores, g_ms = _model_scores(sample_gan, cid, n, EVAL_SEED)
        v_scores, v_ms = _model_scores(sample_vae, cid, n, EVAL_SEED)
        total_ms = (time.perf_counter() - t0) * 1000
        g_sum, v_sum = clip_evaluator.summarize(g_scores), clip_evaluator.summarize(v_scores)
        row = {"prompt": prompt, "requested_category": config.CLASS_NAMES[expected],
               "generated_category": config.CLASS_NAMES[cid], "parser_source": parsed["source"],
               "requested_count": n, "generated_count": len(g_scores), "cgan": g_sum, "vae": v_sum,
               "cgan_generation_ms": round(g_ms, 1), "vae_generation_ms": round(v_ms, 1),
               "total_processing_ms": round(total_ms, 1)}
        prompt_rows.append(row)
        if cid != expected: failures.append({"prompt": prompt, "reason": "parser chose a different category"})
        if g_sum["top1_agreement"] < 0.5:
            failures.append({"prompt": prompt, "reason": f"CLIP top-1 agrees for only {g_sum['top1_agreement']:.0%} of cGAN images"})

    classwise = []
    for cid, name in enumerate(config.CLASS_NAMES):
        g_scores, g_ms = _model_scores(sample_gan, cid, PER_CLASS_IMAGES, EVAL_SEED + cid)
        v_scores, v_ms = _model_scores(sample_vae, cid, PER_CLASS_IMAGES, EVAL_SEED + cid)
        classwise.append({"class_id": cid, "category": name, "n_images": PER_CLASS_IMAGES,
                          "cgan": clip_evaluator.summarize(g_scores), "vae": clip_evaluator.summarize(v_scores),
                          "cgan_ms_per_image": round(g_ms / PER_CLASS_IMAGES, 2),
                          "vae_ms_per_image": round(v_ms / PER_CLASS_IMAGES, 2)})
    weakest = sorted(classwise, key=lambda r: r["cgan"]["mean_similarity"])[:3]
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "seed": EVAL_SEED,
        "clip_model": config.CLIP_MODEL,
        "note": "CLIP similarity measures image-text agreement, not verified classification accuracy.",
        "prompt_experiment": prompt_rows, "classwise": classwise, "failures": failures,
        "weakest_categories": [{"category": r["category"], "cgan_mean_similarity": r["cgan"]["mean_similarity"]} for r in weakest],
        "totals": {"images_evaluated": sum(r["n_images"] for r in classwise) * 2 + sum(r.get("generated_count", 0) * 2 for r in prompt_rows),
                   "cgan_mean_similarity": round(mean(r["cgan"]["mean_similarity"] for r in classwise), 4),
                   "vae_mean_similarity": round(mean(r["vae"]["mean_similarity"] for r in classwise), 4),
                   "cgan_mean_ms_per_image": round(mean(r["cgan_ms_per_image"] for r in classwise), 2),
                   "vae_mean_ms_per_image": round(mean(r["vae_ms_per_image"] for r in classwise), 2)},
    }
    config.EVAL_RESULTS.write_text(json.dumps(results, indent=2))
    print("Saved", config.EVAL_RESULTS)
    return results


if __name__ == "__main__":
    run()
