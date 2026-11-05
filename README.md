# FashionGen AI – Text-to-Conditional-GAN Bridge (P14)

*Transform your words into AI-generated fashion images.*

Type "Show me four ankle boots" → an LLM extracts `{category, class_id, count}` → a Conditional GAN (PyTorch, trained on Fashion-MNIST) generates the images → CLIP scores the image–text match → a plain HTML/CSS/JS gallery shows and downloads them. A conditional VAE is included for comparison.

## 1. Quick start (local)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (optional) LLM parsing with Gemini – without a key the rule-based parser is used
cp .env.example .env     # then export GEMINI_API_KEY=... in your shell

# 1) Train (Fashion-MNIST downloads automatically). GPU/Colab: 30 epochs; CPU: expect ~2 min/epoch
python -m training.train_gan --epochs 30
python -m training.train_vae --epochs 30

# 2) Run the evaluation experiment (downloads CLIP the first time, ~600 MB)
python -m services.evaluation_service

# 3) Start the API + frontend
uvicorn main:app --reload --port 8000
```
Open **http://localhost:8000/app/** (the backend serves the frontend) · API docs: **http://localhost:8000/docs**

To serve the frontend separately: `cd frontend && python -m http.server 5500` (the backend already allows that origin).

> The repository ships with weights/metrics from a short CPU training run (see `backend/outputs/metrics/*.json` for the exact epochs). Retrain with more epochs for better images; the dashboards update automatically.

## 2. How the files connect

| Layer | File | Role |
|---|---|---|
| Frontend | `frontend/js/config.js` | API base URL (edit `PRODUCTION_API` after deploy) |
| | `js/api.js` | All `fetch()` calls, error handling |
| | `js/main.js` | Navbar, footer, backend status, home gallery |
| | `js/generator.js` | Generator page flow (parse → generate → gallery → downloads) |
| | `js/dashboard.js` | Comparison, Training and Evaluation pages (Chart.js) |
| Backend | `backend/main.py` | App, CORS (from `FRONTEND_ORIGINS`), error handlers, static samples |
| | `api/routes.py` | The 8 required endpoints + `/api/compare`, `/api/samples`, `/api/batch/{id}/zip` |
| | `services/llm_parser.py` | Gemini zero/few-shot + rule fallback + Pydantic validation |
| | `services/bert_service.py` | DistilBERT zero-shot category cross-check |
| | `services/generation_service.py` | cGAN/VAE inference, PNG encoding, zip, file pruning |
| | `services/clip_evaluator.py` | CLIP similarity, class-match probability, top-1 |
| | `services/evaluation_service.py` | Ten-prompt + class-wise experiment → JSON |
| | `models/*.py` | Generator, Discriminator, ConditionalVAE, cached loader |
| | `training/*.py` | Dataset loading, GAN and VAE training, metric/sample export |

## 3. API summary
`GET /api/health` · `POST /api/parse-prompt` · `POST /api/generate` · `POST /api/generate-vae` · `GET /api/categories` · `GET /api/training-metrics` · `GET /api/evaluation` · `GET /api/model-info` (plus `/api/compare`, `/api/samples`, `/api/batch/{id}/zip`). Errors: 422 invalid input, 404 missing metrics/batch, 503 model not trained, 500 unexpected.

## 4. Tests
```bash
cd backend && pytest -v
```
Covers health, all ten categories (parser and generation), invalid requests, shapes of all models, label conditioning, seed reproducibility, zip download, path-traversal safety, missing-weights error, and CLIP (skipped automatically if the Hugging Face model can't be downloaded).

## 5. Deployment

**Backend (Render).** Push to GitHub → Render → *New → Blueprint* → pick the repo (uses `render.yaml`). Set secrets: `GEMINI_API_KEY` (optional) and `FRONTEND_ORIGINS=https://<your-frontend-url>`. Weights are in the repo (`backend/outputs/saved_models`) so no separate download is needed; if you exclude them, upload them to a GitHub Release/Hugging Face Hub and download at build time.

**Memory warning.** CLIP (ViT-B/32) + DistilBERT + PyTorch need roughly 1.5 GB RAM. Render's 512 MB free tier will not fit that. Options: (a) Render paid instance; (b) **Hugging Face Spaces (Docker, free 16 GB)** using `backend/Dockerfile` (set port 7860, secrets in Space settings); (c) on free Render set `ENABLE_CLIP=false` and `ENABLE_BERT=false` – generation still works, scores are omitted and the UI says so. Free instances also sleep: the first request can take ~1 minute.

**Frontend (Netlify / Vercel / GitHub Pages).** Publish the `frontend/` folder as a static site (no build command). Then edit `PRODUCTION_API` in `frontend/js/config.js` to the backend URL and put the frontend URL into `FRONTEND_ORIGINS` on the backend (CORS is restricted to that list).

**Secrets.** API keys exist only as backend environment variables. No endpoint returns them (tested).

## 6. Syllabus coverage
Unit 1 prompts + JSON validation (`llm_parser.py`, `/api/parse-prompt` with `compare:true`) · Unit 2 cGAN (`models/`, `train_gan.py`) · Unit 3 LLM + BERT + CLIP + VAE · Unit 4 services, pipelines, API · Unit 5 licences, limitations, ten-prompt experiment, failure cases (`docs/`, About and Evaluation pages).

## 7. Viva cheat-sheet
- **Why "conditional"?** Both networks get the class label, so we choose what is generated.
- **Why a label embedding in G but a label map in D?** G needs a compact vector to join the noise; D needs the label aligned with image pixels, so it becomes an extra 28×28 channel.
- **Why does the VAE look blurrier?** Its loss averages pixels (reconstruction), while the GAN's adversarial loss rewards sharp, realistic detail.
- **Is CLIP score accuracy?** No. It's image–text similarity from a model not trained on 28×28 grayscale images.
- **What does the LLM add?** It turns free text into validated JSON; the rule-based parser guarantees the app still works offline.
- **What is the BERT model used for?** Zero-shot embedding cross-check of the category (no fine-tuning).

## 8. Licence & credits
Fashion-MNIST © Zalando SE, MIT. CLIP (OpenAI, MIT), DistilBERT (Apache-2.0). Add your own licence for the project code.
