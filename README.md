# DFN Generative Baselines

This project is a minimal, runnable generative baseline for 2D DFN (Discrete Fracture Network) binary images. Each DFN image is a single-channel 128 x 128 PNG where 0 is matrix/background and 255 is fracture.

The current scope is intentionally narrow: WGAN-GP, WAE, VQ-VAE, and pixel-space Flow Matching baselines. It does not include EDFM or flow validation, or real outcrop data processing.

## Project Layout

```text
dfn_gan/
  configs/wgan_gp_128.yaml
  configs/wae_mmd_128.yaml
  configs/wae_gan_128.yaml
  configs/vqvae_128.yaml
  configs/flow_matching_128.yaml
  data/synthetic_dfn_128/images/
  data/synthetic_dfn_128/metadata/
  src/datasets/dfn_dataset.py
  src/models/wgan_gp.py
  src/models/wae.py
  src/models/vqvae.py
  src/models/flow_matching.py
  src/training/train_wgan_gp.py
  src/training/train_wae.py
  src/training/train_vqvae.py
  src/training/train_flow_matching.py
  src/training/train_lightning.py
  src/utils/
  src/generate_synthetic_dfn.py
  D:/dfn_gan_outputs/dfn_gan_128/samples/
  D:/dfn_gan_outputs/dfn_gan_128/checkpoints/
  D:/dfn_gan_outputs/dfn_gan_128/logs/
  requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Generate Synthetic DFN Data

From the `dfn_gan` directory:

```bash
python src/generate_synthetic_dfn.py --num_samples 10000 --image_size 128 --out_dir data/synthetic_dfn_128
```

The script writes PNG images to `data/synthetic_dfn_128/images` and JSON metadata to `data/synthetic_dfn_128/metadata`. Metadata includes `sample_id`, `image_size`, `num_fractures`, and per-fracture `center_x`, `center_y`, `length`, `angle`, and `width`.

Optional controls include:

```bash
python src/generate_synthetic_dfn.py --orientation von_mises --von_mises_kappa 8.0 --length_distribution power_law
```

## Train WGAN-GP

```bash
python src/training/train_wgan_gp.py --config configs/wgan_gp_128.yaml
```

The trainer uses Wasserstein critic loss plus gradient penalty:

```text
critic_loss = fake_score.mean() - real_score.mean() + lambda_gp * gradient_penalty
generator_loss = -fake_score.mean()
```

The default config uses `device: auto`, which selects CUDA on Linux GPU hosts, MPS on Apple Silicon Macs, and CPU otherwise. You can also set `device` explicitly to `cuda`, `mps`, or `cpu`.

## Train WAE

```bash
python src/training/train_wae.py --config configs/wae_mmd_128.yaml
python src/training/train_wae.py --config configs/wae_gan_128.yaml
```

WAE sample grids use the same probability and binary PNG format as WGAN-GP, so they can be evaluated by the same `evaluate_dfn.py` script.

The WAE-MMD config uses a binary-image reconstruction preset:

```text
reconstruction_loss = bce_weight * BCE(probability, target) + (1 - bce_weight) * DiceLoss(probability, target)
```

The trainer maps decoder outputs and normalized dataset tensors from `[-1, 1]` back to `[0, 1]` before computing BCE and Dice. WAE-GAN keeps the previous L1 reconstruction loss unless its config opts into `regularizer.reconstruction_loss: bce_dice`.

## Train VQ-VAE

```bash
python src/training/train_vqvae.py --config configs/vqvae_128.yaml
```

For a quick smoke run:

```bash
python src/training/train_vqvae.py --config configs/vqvae_128.yaml --max_batches 1
```

The VQ-VAE baseline encodes each DFN image into a 16 x 16 grid of discrete codebook indices, applies straight-through vector quantization, and decodes the quantized features back to image space. It saves reconstruction grids and optional random-code decode grids in the same probability and binary PNG format used by the other trainers. Random-code grids are useful as a sanity check, but VQ-VAE does not learn a latent prior by itself.

## Train Flow Matching

```bash
python src/training/train_flow_matching.py --config configs/flow_matching_128.yaml
```

For a quick smoke run:

```bash
python src/training/train_flow_matching.py --config configs/flow_matching_128.yaml --max_batches 1
```

The Flow Matching baseline is an unconditional pixel-space Rectified Flow model. During training it samples Gaussian noise `x0`, real images `x1`, interpolates `x_t = (1 - t) * x0 + t * x1`, and trains a time-conditioned UNet to predict velocity `x1 - x0`. Sampling integrates the learned velocity field from noise at `t=0` to images at `t=1`; `sampler.solver` supports `euler`, `heun`, and `midpoint`.

Flow Matching sample grids use the same probability and binary PNG format as WGAN-GP and WAE.

By default Flow Matching samples training time `t` uniformly and keeps a fixed learning rate. To bias training time toward high-`t` denoising, set `training.time_sampling.distribution: beta` and tune `beta_alpha` / `beta_beta`. To enable warmup plus cosine learning-rate decay, set `training.scheduler.enabled: true`; older configs and checkpoints remain compatible when these fields are absent or disabled.

To resample from a trained Flow Matching `.pt` or Lightning `.ckpt` checkpoint, use:

```bash
python src/sampling/sample_flow_matching.py --config configs/flow_matching_128.yaml --checkpoint outputs/flow_matching/checkpoints/flow_matching_latest.pt --num_images 256 --batch_size 32
```

The default output is still a probability grid plus a binary grid. Add `--save_mode individual` to save one PNG per sample under separate `_prob` and `_binary` directories, or use `--save_mode both` to write both individual images and grids.

## Optional Lightning Training

Install dependencies from `requirements.txt`, then run:

```bash
python src/training/train_lightning.py --config configs/wgan_gp_128.yaml
python src/training/train_lightning.py --config configs/wae_mmd_128.yaml
python src/training/train_lightning.py --config configs/wae_gan_128.yaml
python src/training/train_lightning.py --config configs/vqvae_128.yaml --model vqvae
python src/training/train_lightning.py --config configs/flow_matching_128.yaml --model flow_matching
```

Lightning uses the same `device: auto` setting. It writes checkpoints, logs, and sample grids under `lightning/` subdirectories to avoid overwriting manual-training outputs.

Lightning mixed precision is controlled by `training.precision`. The default is `32-true`; on CUDA Linux you can try `16-mixed` or `bf16-mixed` for AMP. Keep full precision on MPS unless a specific local PyTorch/Lightning build validates mixed precision for your workload.

## Outputs

Generated sample grids are saved in `D:/dfn_gan_outputs/dfn_gan_128/samples` as both probability grids and thresholded binary grids. Checkpoints are saved in `D:/dfn_gan_outputs/dfn_gan_128/checkpoints`, and CSV logs are written to `D:/dfn_gan_outputs/dfn_gan_128/logs/train_log.csv`.

## Evaluate Generated DFNs

Use the synthetic DFN dataset as the reference distribution, then compare it with a generated binary grid:

```bash
python src/evaluation/evaluate_dfn.py ^
  --real_dir data/synthetic_dfn_128/images ^
  --generated_grid D:/dfn_gan_outputs/dfn_gan_128/samples/step_0010000_binary.png ^
  --out_dir D:/dfn_gan_outputs/dfn_gan_128/evaluation/step_0010000
```

On Linux/macOS, replace `^` with `\`.

The evaluator writes:

```text
metrics_reference.csv
metrics_generated.csv
comparison_metrics.csv
summary.json
comparison_plots.png
```

Current metrics include fracture pixel ratio, connected component count, largest component ratio, mean component area, skeleton length, endpoint count, junction count, Hough line count, and orientation histogram distance.

## Resume Training

```bash
python src/training/train_wgan_gp.py --config configs/wgan_gp_128.yaml --resume D:/dfn_gan_outputs/dfn_gan_128/checkpoints/wgan_gp_latest.pt
```

## Future Extensions

Useful DFN-specific evaluation metrics can be added later, such as fracture length distribution, orientation distribution, connected component counts, percolation probability, and MMD.
