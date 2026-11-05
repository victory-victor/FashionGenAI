"""Central configuration. Everything that changes between laptop and server
is read from environment variables (see .env.example). No secrets live in code."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Prevent HuggingFace transformers from attempting to import TensorFlow/Keras
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "saved_models"
METRICS_DIR = OUTPUT_DIR / "metrics"
SAMPLES_DIR = OUTPUT_DIR / "samples"
GENERATED_DIR = OUTPUT_DIR / "generated"
DATA_DIR = BASE_DIR / "data"          # Fashion-MNIST is downloaded here
for _d in (MODEL_DIR, METRICS_DIR, SAMPLES_DIR, GENERATED_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Fashion-MNIST class mapping (official)
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Natural wording used in CLIP text prompts
CLIP_LABELS = ["t-shirt", "trouser", "pullover", "dress", "coat",
               "sandal", "shirt", "sneaker", "handbag", "ankle boot"]

# Model hyper-parameters (stored inside checkpoints too, so loading never guesses)
LATENT_DIM = 100
VAE_LATENT_DIM = 20
NUM_CLASSES = 10
IMG_SIZE = 28
DISPLAY_SIZE = 224        # generated 28x28 images are upscaled for display/CLIP

GAN_WEIGHTS = MODEL_DIR / "cgan_generator.pt"
DISC_WEIGHTS = MODEL_DIR / "cgan_discriminator.pt"
VAE_WEIGHTS = MODEL_DIR / "cvae.pt"
GAN_METRICS = METRICS_DIR / "gan_metrics.json"
VAE_METRICS = METRICS_DIR / "vae_metrics.json"
EVAL_RESULTS = METRICS_DIR / "evaluation_results.json"

MAX_IMAGES = int(os.getenv("MAX_IMAGES", "8"))

# --- External / optional services -------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")          # never returned by any endpoint
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
BERT_MODEL = os.getenv("BERT_MODEL", "distilbert-base-uncased")
ENABLE_CLIP = os.getenv("ENABLE_CLIP", "true").lower() == "true"
ENABLE_BERT = os.getenv("ENABLE_BERT", "true").lower() == "true"
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))

# --- Web ---------------------------------------------------------------------
FRONTEND_ORIGINS = [o.strip() for o in os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000,http://127.0.0.1:8000,http://localhost:8080",
).split(",") if o.strip()]
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "true").lower() == "true"
