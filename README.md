# DFN WGAN-GP Baseline

This project is a minimal, runnable generative baseline for 2D DFN (Discrete Fracture Network) binary images. Each DFN image is a single-channel 128 x 128 PNG where 0 is matrix/background and 255 is fracture.

The current scope is intentionally narrow: WGAN-GP and WAE baselines only. It does not include diffusion models, EDFM or flow validation, or real outcrop data processing.

## Project Layout

```text
dfn_gan/
  configs/wgan_gp_128.yaml
  configs/wae_mmd_128.yaml
  configs/wae_gan_128.yaml
  data/synthetic_dfn_128/images/
  data/synthetic_dfn_128/metadata/
  src/datasets/dfn_dataset.py
  src/models/wgan_gp.py
  src/models/wae.py
  src/training/train_wgan_gp.py
  src/training/train_wae.py
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

## Optional Lightning Training

Install dependencies from `requirements.txt`, then run:

```bash
python src/training/train_lightning.py --config configs/wgan_gp_128.yaml
python src/training/train_lightning.py --config configs/wae_mmd_128.yaml
python src/training/train_lightning.py --config configs/wae_gan_128.yaml
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
