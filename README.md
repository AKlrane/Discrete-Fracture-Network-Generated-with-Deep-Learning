# Discrete Fracture Network Generation with Deep Learning

This repository is part of course project done by Chaoran Liu and Zhi Xiang Koh, in Generation with Deep Learning course offered by Prof. Yizhou Wang in Peking University.

The repository contains reproducible code for generating and evaluating 2D
discrete fracture network (DFN) images with deep generative models. The project
reproduces a WGAN-GP baseline following

> Teng, Z., Wu, H., Zhang, J., Ju, X., & Qi, S. (2025). Generating high-fidelity discrete fracture networks from low-dimensional latent spaces using generative adversarial network. International Journal of Rock Mechanics and Mining Sciences, 196, 106301. <https://doi.org/10.1016/j.ijrmms.2025.106301>

adds several alternative generative models, and provides a common image-level evaluation
pipeline for the results.

## What Is Included

- Teng-style synthetic DFN dataset generation with fractal positions, truncated
  power-law lengths, and von Mises orientations.
- WGAN-GP training for low-dimensional latent-to-DFN generation.
- Alternative baselines: WAE-MMD, beta-VAE, VQ-VAE, Sphere Encoder, pixel-space
  Flow Matching, and latent-space Flow Matching.
- A shared evaluator that computes fracture pixel ratio, connected components,
  largest component ratio, skeleton length, Hough line statistics, orientation
  histograms, length histograms, occurrence overlays, and line-center heatmaps.
- A proof-of-concept latent inversion workflow with a deterministic pressure
  surrogate and an optional GEOS/EDFM adapter.

## Repository Layout

```text
configs/                         Experiment and dataset configuration files
configs/dataset/                 Synthetic DFN dataset presets
configs/inversion/               Latent inversion presets
src/generate_synthetic_dfn.py     Synthetic DFN generator
src/training/                    Training entry points
src/sampling/                    Post-training samplers
src/evaluation/evaluate_dfn.py    Shared image-level evaluator
src/inversion/                   Latent inversion and forward-model utilities
outputs/                         Generated artifacts and selected reference outputs
requirements.txt                 Python package requirements
```

## Environment

Use Python 3.10 or newer from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build matching the local CUDA or accelerator
stack before installing the remaining requirements. All configs use
`device: auto`, which selects CUDA, Apple MPS, or CPU according to availability.

## Reproduce the Synthetic Dataset

The main paper experiments use a 50-fracture, 128 x 128 Teng-style synthetic
dataset.

```bash
python src/generate_synthetic_dfn.py \
  --config configs/dataset/teng_unconditioned_50_128.yaml
```

Expected output:

```text
data/synthetic_dfn_teng_50_128/images/
data/synthetic_dfn_teng_50_128/metadata/
```

The default config generates 30,000 binary PNGs with a 20 m x 20 m physical
domain, 50 fractures per image, fractal center locations, truncated power-law
lengths, and a von Mises orientation distribution.

## Train the WGAN-GP Baseline

```bash
python src/training/train_wgan_gp.py \
  --config configs/wgan_gp_128.yaml
```

Default outputs:

```text
outputs/samples/
outputs/checkpoints/
outputs/logs/train_log.csv
```

The WGAN-GP config uses a 128-dimensional latent vector, `base_channels: 64`,
five critic updates per generator update, and gradient penalty
`lambda_gp: 10`.

To evaluate the selected WGAN-GP sample grid used for the Teng-style table in
the paper draft, run:

```bash
python src/evaluation/evaluate_dfn.py \
  --real_dir data/synthetic_dfn_teng_50_128/images \
  --generated_grid outputs/samples/step_0014500_prob.png \
  --out_dir outputs/evaluation/step_0014500_prob_vs_synthetic_dfn_teng_50_128 \
  --max_real_images 512 \
  --max_generated_images 64 \
  --grid_rows 8 \
  --grid_cols 8 \
  --grid_padding 2
```

If the exact step is not present because training was stopped early or run with
different sampling intervals, evaluate the nearest generated `*_binary.png` or
`*_prob.png` grid and report the matching output directory.

## Train Alternative Methods

The paper draft compares WGAN-GP with several alternative methods on the same
Teng-style 50-fracture dataset. The commands below reproduce the model outputs
without using shell wrappers.

```bash
python src/training/train_wae.py \
  --config configs/wae_mmd_16_corrected.yaml

python src/training/train_wae.py \
  --config configs/wae_mmd_32_structural_overdensity.yaml

python src/training/train_beta_vae.py \
  --config configs/beta_vae_16_capacity.yaml

python src/training/train_sphere_encoder.py \
  --config configs/sphere_encoder_16_structural.yaml

python src/training/train_vqvae.py \
  --config configs/vqvae_128.yaml

python src/training/train_flow_matching.py \
  --config configs/flow_matching_128.yaml
```

Notes for interpretation:

- WAE-MMD, beta-VAE, and Sphere Encoder produce both reconstruction grids and
  prior-sample grids. The paper table uses prior-sample grids for generative
  comparison.
- VQ-VAE is reconstruction-only in the default config because no learned prior
  over code indices is trained. Its reconstructions are useful diagnostics but
  are not directly comparable to WGAN-GP samples as unconditional generation.
- Pixel-space Flow Matching is a useful image generator baseline, but it does
  not provide a compact latent parameterization for inversion.

## Reproduce the Alternative-Method Metrics

After training, evaluate the generated grids against the same reference set.
The following commands reproduce the metrics table in the paper draft when run
with the default configs and final sample grids.

```bash
python src/evaluation/evaluate_dfn.py \
  --real_dir data/synthetic_dfn_teng_50_128/images \
  --generated_grid outputs/wae_mmd_16_corrected/samples/step_0093600_binary.png \
  --out_dir outputs/evaluation/alternative_methods/wae_mmd_16_corrected_sample \
  --max_real_images 512 \
  --max_generated_images 64 \
  --grid_rows 8 \
  --grid_cols 8 \
  --grid_padding 2

python src/evaluation/evaluate_dfn.py \
  --real_dir data/synthetic_dfn_teng_50_128/images \
  --generated_grid outputs/wae_mmd_32_structural_overdensity/samples/step_0093600_binary.png \
  --out_dir outputs/evaluation/alternative_methods/wae_mmd_32_overdensity_sample \
  --max_real_images 512 \
  --max_generated_images 64 \
  --grid_rows 8 \
  --grid_cols 8 \
  --grid_padding 2

python src/evaluation/evaluate_dfn.py \
  --real_dir data/synthetic_dfn_teng_50_128/images \
  --generated_grid outputs/beta_vae_16_capacity/samples/step_0093600_binary.png \
  --out_dir outputs/evaluation/alternative_methods/beta_vae_16_capacity_sample \
  --max_real_images 512 \
  --max_generated_images 64 \
  --grid_rows 8 \
  --grid_cols 8 \
  --grid_padding 2

python src/evaluation/evaluate_dfn.py \
  --real_dir data/synthetic_dfn_teng_50_128/images \
  --generated_grid outputs/sphere_encoder_16_structural/samples/step_0093600_binary.png \
  --out_dir outputs/evaluation/alternative_methods/sphere_encoder_16_structural_1step \
  --max_real_images 512 \
  --max_generated_images 64 \
  --grid_rows 8 \
  --grid_cols 8 \
  --grid_padding 2

python src/evaluation/evaluate_dfn.py \
  --real_dir data/synthetic_dfn_teng_50_128/images \
  --generated_grid outputs/flow_matching_teng_50H/samples/step_0093600_binary.png \
  --out_dir outputs/evaluation/alternative_methods/flow_matching_legacy_sample \
  --max_real_images 512 \
  --max_generated_images 64 \
  --grid_rows 8 \
  --grid_cols 8 \
  --grid_padding 2
```

To evaluate VQ-VAE reconstructions for the reconstruction figure:

```bash
python src/evaluation/evaluate_dfn.py \
  --real_dir data/synthetic_dfn_teng_50_128/images \
  --generated_grid outputs/vqvae_teng_50_128/samples/step_0093600_recon_binary.png \
  --out_dir outputs/evaluation/alternative_methods/vqvae_recon \
  --max_real_images 512 \
  --max_generated_images 64 \
  --grid_rows 8 \
  --grid_cols 8 \
  --grid_padding 2
```

Each evaluation directory contains:

```text
comparison_metrics.csv
metrics_reference.csv
metrics_generated.csv
summary.json
comparison_plots.png
overlay_comparison.png
overlay_reference.png
overlay_generated.png
overlay_abs_difference.png
center_heatmap_comparison.png
```

These files are the numerical and visual sources for the metric table,
statistical evaluation figure, occurrence overlay figure, and alternative-method
comparison in the paper draft.

## Optional Lightning Entry Point

Most models have standalone trainers. If a Lightning workflow is preferred, use
the tracked Lightning entry point directly:

```bash
python src/training/train_lightning.py \
  --config configs/wgan_gp_128.yaml \
  --model wgan_gp

python src/training/train_lightning.py \
  --config configs/vqvae_128.yaml \
  --model vqvae

python src/training/train_lightning.py \
  --config configs/flow_matching_128.yaml \
  --model flow_matching_legacy
```

Lightning writes sample grids, checkpoints, and logs under `lightning/`
subdirectories to avoid overwriting standalone-training outputs.

## Latent Flow Matching

Latent Flow Matching is a two-stage baseline: first train a compact WAE, then
freeze its encoder and decoder while training a latent velocity model.

```bash
python src/training/train_wae.py \
  --config configs/wae_mmd_16.yaml

python src/training/train_latent_flow_matching.py \
  --config configs/latent_flow_matching_16.yaml
```

To sample from a trained latent Flow Matching checkpoint:

```bash
python src/sampling/sample_latent_flow_matching.py \
  --config configs/latent_flow_matching_16.yaml \
  --checkpoint outputs/latent_flow_matching_16/checkpoints/latent_flow_matching_latest.pt \
  --num_images 64 \
  --batch_size 64 \
  --save_mode grid
```

## One-Step Inversion Workflow

The inversion code links a latent DFN generator to pressure-response comparison.
The default inversion config uses a deterministic mock pressure forward model,
so it can run without GEOS.

First train or provide the conditioned WGAN-GP prior expected by the inversion
config:

```bash
python src/training/train_wgan_gp.py \
  --config configs/wgan_gp_teng_conditioned_case1_ld16.yaml
```

Then generate a synthetic pressure-observation case:

```bash
python src/inversion/make_synthetic_pressure_case.py \
  --config configs/inversion/teng_pressure_ld16.yaml
```

Run latent-space inversion with the default sampler:

```bash
python src/inversion/run_mcmc.py \
  --config configs/inversion/teng_pressure_ld16.yaml \
  --sampler emcee
```

For a quick smoke test:

```bash
python src/inversion/run_mcmc.py \
  --config configs/inversion/teng_pressure_ld16.yaml \
  --sampler emcee \
  --max_steps 5
```

Inversion outputs are written under:

```text
outputs/inversion/teng_pressure_ld16/
```

The optional GEOS/EDFM path is configured by
`configs/inversion/teng_pressure_ld16_geos.yaml`. It requires a local GEOS
executable and template directory, so it is not part of the default
repository-only reproduction path.

## Rebuild Paper Figures and Tables

The paper draft figures are assembled from the artifacts above:

- Synthetic data examples: `data/synthetic_dfn_teng_50_128/images/`
- WGAN-GP sample grids: `outputs/samples/*_binary.png` and
  `outputs/samples/*_prob.png`
- Evaluation plots and overlays: `outputs/evaluation/**/comparison_plots.png`,
  `overlay_comparison.png`, and `center_heatmap_comparison.png`
- Alternative-method sample or reconstruction grids:
  `outputs/*/samples/*_binary.png` and `outputs/*/samples/*_recon_binary.png`
- Inversion artifacts:
  `outputs/inversion/teng_pressure_ld16/**/best_pressure_prediction.csv`,
  `best_binary.png`, and `best_summary.json`

The metric table is populated from each evaluation directory's
`comparison_metrics.csv`. Use the generated/reference means for scalar metrics
and the reported generated value for orientation `L1` distance.

## Fast Sanity Checks

Use small smoke runs before launching full 200-epoch jobs:

```bash
python src/generate_synthetic_dfn.py \
  --config configs/dataset/teng_unconditioned_50_128.yaml \
  --num_samples 16 \
  --out_dir /tmp/dfn_smoke

python src/training/train_wae.py \
  --config configs/wae_mmd_16_corrected.yaml \
  --max_batches 1

python src/training/train_beta_vae.py \
  --config configs/beta_vae_16_capacity.yaml \
  --max_batches 1

python src/training/train_sphere_encoder.py \
  --config configs/sphere_encoder_16_structural.yaml \
  --max_batches 1

python src/training/train_vqvae.py \
  --config configs/vqvae_128.yaml \
  --max_batches 1

python src/training/train_flow_matching.py \
  --config configs/flow_matching_128.yaml \
  --max_batches 1
```

## Reproducibility Notes

- Run commands from the repository root.
- Keep the YAML configs unchanged when reproducing paper numbers.
- Use the same generated grid when comparing metric values; different training
  steps or probability-versus-binary grids produce different statistics.
- VQ-VAE reconstructions should be discussed separately from generated samples
  unless a learned code prior is added.
- Image-level metrics are useful diagnostics, but they do not prove hydraulic
  equivalence. Use the inversion workflow for pressure-response comparisons.
