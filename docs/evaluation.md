# Evaluation protocol

Run: `cd backend && python -m services.evaluation_service` (needs trained weights and CLIP).
Output: `backend/outputs/metrics/evaluation_results.json`, which the Evaluation page reads. Nothing is typed in by hand.

## Experiment 1 - ten test prompts
Prompts (one per class, hand-written ground truth in `TEST_PROMPTS`) go through the real pipeline: parse -> cGAN -> CLIP, and the same class/count through the VAE. Recorded per prompt: requested category, generated category, requested vs generated image count, CLIP similarity, CLIP top-1 agreement, and processing time.

## Experiment 2 - class-wise analysis
16 images per class for each model (seeded, so reproducible). Reported: mean CLIP cosine similarity, mean class-match probability, top-1 agreement, ms per image. The three lowest cGAN similarities are listed as the weakest categories.

## Failure cases
A failure is logged when the parser picks a category different from the ground truth, or CLIP's top-1 agrees with the requested class for fewer than 50% of the cGAN images.

## How to read the numbers
- *CLIP similarity* = cosine between image and text embeddings. Values near 0.2-0.3 are normal for CLIP even for good matches.
- *Top-1 agreement* = how often CLIP's best label equals the requested class. It is a CLIP-based agreement rate, **not verified classification accuracy**, because CLIP is evaluated outside the domain it was trained on.
- Timings depend on hardware and include only CPU inference.
For a stricter measure, train a Fashion-MNIST classifier on the training set and test it on generated images (future work).
