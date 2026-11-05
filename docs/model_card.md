# Model card - FashionGen AI

**Models:** Conditional GAN (generator ~ DCGAN-style, discriminator conditioned via label map) and Conditional VAE, both written in PyTorch and trained from scratch on Fashion-MNIST.
**Intended use:** classroom demonstration of conditional generation. Not for production or commercial use.
**Training data:** Fashion-MNIST, 60,000 training images, 10 classes, 28x28 grayscale. Licence: MIT, (c) Zalando SE. https://github.com/zalandoresearch/fashion-mnist
**Training budget:** see `backend/outputs/metrics/gan_metrics.json` and `vae_metrics.json` for the actual epochs, batch size, learning rate and device used. More epochs generally improve the GAN.
**Third-party models:** CLIP (`openai/clip-vit-base-patch32`, MIT), DistilBERT (`distilbert-base-uncased`, Apache-2.0), Google Gemini API (optional, Google terms).

## Limitations
- Low resolution and no colour; outputs are not product-quality images.
- Only ten classes; "Shirt", "T-shirt/top", "Pullover" and "Coat" are visually similar and easy to confuse.
- GAN failure modes: mode collapse, artefacts, unstable training. VAE failure mode: blur.
- CLIP is not trained on 28x28 grayscale images; its similarity is a weak proxy and is not classification accuracy.
- The BERT cross-check is zero-shot embedding similarity without fine-tuning, so it is a sanity check only.

## Bias and fairness
The dataset contains no people, but it reflects a single retailer's catalogue and Western clothing categories (no saris, kurtas, etc.). Class-wise results are reported on the Evaluation page to show which categories perform worse.

## Responsible use
Label generated images as synthetic. Do not present them as real photographs. The LLM sees only the user's request text.
