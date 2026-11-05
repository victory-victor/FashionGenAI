# Architecture

```
Browser (HTML/CSS/JS)  --fetch-->  FastAPI (backend/main.py)
                                     |-- api/routes.py          thin endpoints, validation, HTTP codes
                                     |-- services/llm_parser    text -> {category, class_id, count} (Gemini or rules) + Pydantic check
                                     |-- services/bert_service  DistilBERT zero-shot cross-check
                                     |-- services/generation_service  cGAN / VAE inference, PNG encoding, zip
                                     |-- services/clip_evaluator      CLIP image-text similarity
                                     |-- services/evaluation_service  offline experiment -> outputs/metrics/evaluation_results.json
                                     '-- models/*               Generator, Discriminator, ConditionalVAE, cached loader
Training (offline): training/train_gan.py, train_vae.py -> outputs/saved_models/*.pt + outputs/metrics/*.json + outputs/samples/*.png
```

## Request flow for "Show me four ankle boots"
1. `generator.js` calls `POST /api/parse-prompt`.
2. `llm_parser.parse_prompt` asks Gemini (few-shot prompt) for JSON; if no key, the rule-based parser runs. `ParsedRequest` validates `class_id` 0-9 and `count` 1-8.
3. `generator.js` calls `POST /api/generate` with `class_id` and `count`.
4. `generation_service.sample_gan` draws noise, feeds noise + label to the Generator, rescales [-1,1] to [0,1].
5. Images are upscaled to 224x224, saved to `outputs/generated/<batch>/`, and returned as base64 PNG.
6. `clip_evaluator.evaluate` computes cosine similarity to "a photo of a ankle boot", a softmax over the ten class prompts, and CLIP's top-1 label.
7. The gallery renders images, scores and download buttons; "Download all" calls `GET /api/batch/{id}/zip`.

## Models
- **Generator**: z (100) + label embedding (50) -> Linear -> 128x7x7 -> ConvT 64x14x14 -> ConvT 1x28x28 -> Tanh.
- **Discriminator**: image + label embedded as a 28x28 channel -> Conv 64 -> Conv 128 -> Linear -> real/fake logit.
- **Conditional VAE**: MLP encoder (image + one-hot) -> mu, logvar (20) -> MLP decoder (z + one-hot) -> Sigmoid.
- Loss: GAN uses BCE with logits (one-sided label smoothing 0.9 for real); VAE uses BCE reconstruction + KL.

## Syllabus map
| Unit | Where |
|---|---|
| 1 Foundations | `llm_parser.py` (zero-/few-shot prompts, JSON validation), About page |
| 2 GANs + LLMs | `models/generator.py`, `discriminator.py`, `training/train_gan.py` |
| 3 GPT/BERT/multimodal | `llm_parser.py` (LLM), `bert_service.py`, `clip_evaluator.py`, `models/vae.py` |
| 4 Tools & techniques | `services/`, `training/`, `api/routes.py`, frontend |
| 5 Responsible AI | `docs/model_card.md`, `docs/evaluation.md`, About + Evaluation pages |
