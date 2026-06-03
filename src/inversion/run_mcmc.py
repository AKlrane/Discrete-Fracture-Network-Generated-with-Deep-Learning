from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inversion.common import load_config, resolve_path, write_json
from src.inversion.forward_model import create_forward_model
from src.inversion.fracture_extract import write_segments_csv
from src.inversion.latent_prior import WGANLatentPrior, save_binary_image, save_probability_image
from src.inversion.likelihood import LatentPosterior, load_pressure_observations
from src.inversion.samplers import (
    run_dream_zs_sampler,
    run_emcee_sampler,
    save_sampler_result,
)


def write_pressure_prediction(path: Path, points: np.ndarray, observed: np.ndarray, predicted: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "observed_pressure", "predicted_pressure", "residual"])
        writer.writeheader()
        for point, obs, pred in zip(points, observed, predicted):
            writer.writerow(
                {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "observed_pressure": float(obs),
                    "predicted_pressure": float(pred),
                    "residual": float(pred - obs),
                }
            )


def flattened_best_sample(chain: np.ndarray, log_prob: np.ndarray, burn_in: int) -> tuple[np.ndarray, float]:
    chain_post = chain[int(burn_in) :]
    log_prob_post = log_prob[int(burn_in) :]
    if chain_post.size == 0:
        chain_post = chain
        log_prob_post = log_prob
    flat_chain = chain_post.reshape(-1, chain.shape[-1])
    flat_log_prob = log_prob_post.reshape(-1)
    index = int(np.argmax(flat_log_prob))
    return flat_chain[index], float(flat_log_prob[index])


def run(config: dict, sampler_override: str | None = None, max_steps: int | None = None) -> None:
    prior = WGANLatentPrior.from_config(config["prior"])
    observations = load_pressure_observations(config["observations"]["pressure_csv"])
    forward_model = create_forward_model(config, prior, observations.points)
    posterior = LatentPosterior(
        forward_model=forward_model,
        observations=observations,
        sigma=float(config["observations"].get("noise_sigma", 0.05)),
        prior_scale=float(config.get("posterior", {}).get("prior_scale", 1.0)),
        failure_log_prob=float(config.get("posterior", {}).get("failure_log_prob", -1.0e100)),
    )

    sampler_cfg = config["sampler"]
    backend = sampler_override or str(sampler_cfg.get("backend", "emcee"))
    latent_dim = int(config["prior"]["latent_dim"])
    num_steps = int(max_steps or sampler_cfg.get("num_steps", 200))
    burn_in = int(sampler_cfg.get("burn_in", 50))
    seed = int(sampler_cfg.get("seed", 42))
    init_scale = float(sampler_cfg.get("init_scale", 1.0))

    if backend == "emcee":
        result = run_emcee_sampler(
            posterior.log_prob,
            latent_dim=latent_dim,
            num_walkers=int(sampler_cfg.get("num_walkers", max(32, 2 * latent_dim))),
            num_steps=num_steps,
            seed=seed,
            init_scale=init_scale,
            progress=bool(sampler_cfg.get("progress", False)),
        )
    elif backend == "dream_zs":
        dream_cfg = sampler_cfg.get("dream_zs", {})
        result = run_dream_zs_sampler(
            posterior.log_prob,
            latent_dim=latent_dim,
            num_chains=int(dream_cfg.get("num_chains", sampler_cfg.get("num_chains", 32))),
            num_steps=num_steps,
            seed=seed,
            init_scale=init_scale,
            crossover_probability=float(dream_cfg.get("crossover_probability", 0.9)),
            noise_scale=float(dream_cfg.get("noise_scale", 1e-6)),
            gamma_scale=float(dream_cfg.get("gamma_scale", 1.0)),
        )
    else:
        raise ValueError("sampler.backend must be one of: emcee, dream_zs")

    out_dir = resolve_path(config["outputs"]["out_dir"]) / backend
    save_sampler_result(out_dir, result, burn_in=burn_in)
    best_z, best_log_prob = flattened_best_sample(result.chain, result.log_prob, burn_in)
    best = posterior.evaluate(best_z)
    np.save(out_dir / "best_z.npy", best_z)
    write_pressure_prediction(
        out_dir / "best_pressure_prediction.csv",
        observations.points,
        observations.pressures,
        best["pressures"],
    )
    write_segments_csv(out_dir / "best_segments.csv", best["segments"])
    if best["binary"] is not None:
        save_binary_image(best["binary"], out_dir / "best_binary.png")
    if best["probability"] is not None:
        save_probability_image(best["probability"], out_dir / "best_probability.png")
    write_json(
        out_dir / "best_summary.json",
        {
            "backend": backend,
            "best_log_prob": best_log_prob,
            "best_rmse": best["rmse"],
            "segment_count": len(best["segments"]),
            "failure_count": len(posterior.failures),
            "recent_failures": posterior.failures[-10:],
            "forward_metadata": best["metadata"],
        },
    )
    print(f"Wrote MCMC outputs to {out_dir}")
    print(f"Best log_prob={best_log_prob:.3f}, pressure RMSE={best['rmse']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent-space DFN pressure inversion.")
    parser.add_argument("--config", type=Path, default=Path("configs/inversion/teng_pressure_ld16.yaml"))
    parser.add_argument("--sampler", choices=("emcee", "dream_zs"), default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="Override sampler.num_steps for smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), sampler_override=args.sampler, max_steps=args.max_steps)
