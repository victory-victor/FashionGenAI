"""Run from backend/:  pytest -v
Tests that need trained weights or downloaded Hugging Face models skip themselves if those are missing."""
import pytest
import torch
from fastapi.testclient import TestClient

import config
from main import app
from models.generator import Generator
from models.discriminator import Discriminator
from models.vae import ConditionalVAE
from services.llm_parser import rule_based_parse, parse_prompt, ParseError
from services import clip_evaluator

client = TestClient(app)
needs_gan = pytest.mark.skipif(not config.GAN_WEIGHTS.exists(), reason="cGAN not trained")
needs_vae = pytest.mark.skipif(not config.VAE_WEIGHTS.exists(), reason="VAE not trained")


# ---- health / categories ---------------------------------------------------------
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "GEMINI" not in r.text and "api_key" not in r.text.lower()


def test_categories_all_ten():
    cats = client.get("/api/categories").json()["categories"]
    assert [c["class_id"] for c in cats] == list(range(10)) and cats[9]["name"] == "Ankle boot"


# ---- prompt parsing (Unit 1) --------------------------------------------------------
@pytest.mark.parametrize("text,cid,count", [
    ("Show me four ankle boots.", 9, 4), ("Generate three sneakers", 7, 3), ("Create five handbags", 8, 5),
    ("Show me two dresses", 3, 2), ("make 2 t-shirts", 0, 2), ("a coat", 4, 1), ("six trousers", 1, 6),
    ("three pullovers please", 2, 3), ("one sandal", 5, 1), ("show two shirts", 6, 2)])
def test_rule_parser_all_ten_categories(text, cid, count):
    r = rule_based_parse(text)
    assert (r["class_id"], r["count"]) == (cid, count)


def test_parse_endpoint_ok(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    r = client.post("/api/parse-prompt", json={"prompt": "Show me four ankle boots"})
    assert r.status_code == 200
    j = r.json()
    assert (j["category"], j["class_id"], j["count"]) == ("ankle boot", 9, 4) and j["source"] == "rule-based"


@pytest.mark.parametrize("bad", ["draw a spaceship", "hello", "show me 50 dresses", "show me 0 dresses", "  "])
def test_invalid_requests_rejected(bad):
    r = client.post("/api/parse-prompt", json={"prompt": bad})
    assert r.status_code == 422 and "detail" in r.json()


def test_invalid_payload_types():
    assert client.post("/api/parse-prompt", json={"prompt": 123}).status_code == 422
    assert client.post("/api/generate", json={"class_id": 99}).status_code == 422
    assert client.post("/api/generate", json={}).status_code == 422
    assert client.post("/api/generate", json={"class_id": 1, "count": 1000}).status_code == 422


# ---- model shapes (no weights needed) ------------------------------------------------
def test_generator_output_shape_and_range():
    g = Generator().eval()
    out = g(torch.randn(5, config.LATENT_DIM), torch.tensor([9] * 5))
    assert out.shape == (5, 1, 28, 28) and out.min() >= -1 and out.max() <= 1


def test_discriminator_and_vae_shapes():
    d = Discriminator().eval()
    assert d(torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 2, 3])).shape == (4,)
    v = ConditionalVAE().eval()
    recon, mu, lv = v(torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 2, 3]))
    assert recon.shape == (4, 1, 28, 28) and mu.shape == (4, config.VAE_LATENT_DIM)


def test_label_conditioning_changes_output():
    g = Generator().eval()
    z = torch.randn(1, config.LATENT_DIM)
    assert not torch.allclose(g(z, torch.tensor([0])), g(z, torch.tensor([9])))


# ---- generation (need weights) ---------------------------------------------------------
@needs_gan
@pytest.mark.parametrize("cid", range(10))
def test_generate_every_category(cid):
    r = client.post("/api/generate", json={"class_id": cid, "count": 2, "evaluate": False})
    assert r.status_code == 200
    j = r.json()
    assert len(j["images"]) == 2 and j["images"][0]["data_url"].startswith("data:image/png;base64,")
    assert j["class_id"] == cid


@needs_gan
def test_generate_from_text_and_zip():
    r = client.post("/api/generate", json={"prompt": "Show me three sneakers", "evaluate": False})
    j = r.json()
    assert r.status_code == 200 and j["count"] == 3 and j["category"] == "Sneaker"
    z = client.get(f"/api/batch/{j['batch_id']}/zip")
    assert z.status_code == 200 and z.content[:2] == b"PK"
    assert client.get("/api/batch/../../etc/passwd/zip").status_code in (404, 422)
    assert client.get("/api/batch/notahexid/zip").status_code == 404


@needs_gan
def test_seed_is_reproducible():
    a = client.post("/api/generate", json={"class_id": 3, "count": 1, "seed": 5, "evaluate": False}).json()
    b = client.post("/api/generate", json={"class_id": 3, "count": 1, "seed": 5, "evaluate": False}).json()
    assert a["images"][0]["data_url"] == b["images"][0]["data_url"]


@needs_vae
def test_generate_vae():
    r = client.post("/api/generate-vae", json={"class_id": 7, "count": 3, "evaluate": False})
    assert r.status_code == 200 and len(r.json()["images"]) == 3 and r.json()["model"] == "vae"


def test_missing_weights_gives_503(monkeypatch, tmp_path):
    from models.model_loader import registry
    monkeypatch.setattr(config, "GAN_WEIGHTS", tmp_path / "nope.pt")
    registry.reload()
    r = client.post("/api/generate", json={"class_id": 1, "count": 1, "evaluate": False})
    assert r.status_code == 503 and "Train first" in r.json()["detail"]
    registry.reload()


def test_model_info_has_no_secrets():
    r = client.get("/api/model-info")
    assert r.status_code == 200 and "configured" in r.json()["llm"] and "key" not in r.text.lower().replace("keyword", "")


def test_metrics_and_evaluation_endpoints_status():
    assert client.get("/api/training-metrics").status_code in (200, 404)
    assert client.get("/api/evaluation").status_code in (200, 404)


# ---- CLIP (needs Hugging Face download; skipped when offline) -------------------------
def test_clip_evaluation_or_graceful_none():
    from PIL import Image
    res = clip_evaluator.evaluate([Image.new("RGB", (224, 224), "white")], 9)
    if res is None:
        pytest.skip("CLIP unavailable: " + str(clip_evaluator.status()["error"]))
    assert -1 <= res[0]["similarity"] <= 1 and 0 <= res[0]["class_match_prob"] <= 1
