from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import emcee
import numpy as np


LogProbFn = Callable[[np.ndarray], float]


@dataclass
class SamplerResult:
    backend: str
    chain: np.ndarray
    log_prob: np.ndarray
    acceptance_fraction: np.ndarray
    metadata: dict[str, float | int | str]


def initialize_chains(
    latent_dim: int,
    num_chains: int,
    init_scale: float,
    seed: int,
    initial_z: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = np.zeros(latent_dim, dtype=np.float64) if initial_z is None else np.asarray(initial_z, dtype=np.float64)
    if center.shape != (latent_dim,):
        raise ValueError(f"initial_z must have shape ({latent_dim},), got {center.shape}")
    return center[None, :] + rng.normal(0.0, float(init_scale), size=(num_chains, latent_dim))


def run_emcee_sampler(
    log_prob_fn: LogProbFn,
    latent_dim: int,
    num_walkers: int,
    num_steps: int,
    seed: int,
    init_scale: float = 1.0,
    initial_z: np.ndarray | None = None,
    progress: bool = False,
) -> SamplerResult:
    if num_walkers < 2 * latent_dim:
        raise ValueError("emcee requires at least 2 * latent_dim walkers")
    initial_state = initialize_chains(latent_dim, num_walkers, init_scale, seed, initial_z)
    sampler = emcee.EnsembleSampler(num_walkers, latent_dim, log_prob_fn)
    sampler.run_mcmc(initial_state, int(num_steps), progress=progress)
    return SamplerResult(
        backend="emcee",
        chain=sampler.get_chain(),
        log_prob=sampler.get_log_prob(),
        acceptance_fraction=np.asarray(sampler.acceptance_fraction, dtype=np.float64),
        metadata={
            "latent_dim": latent_dim,
            "num_walkers": num_walkers,
            "num_steps": num_steps,
            "seed": seed,
            "init_scale": init_scale,
        },
    )


def run_dream_zs_sampler(
    log_prob_fn: LogProbFn,
    latent_dim: int,
    num_chains: int,
    num_steps: int,
    seed: int,
    init_scale: float = 1.0,
    initial_z: np.ndarray | None = None,
    crossover_probability: float = 0.9,
    noise_scale: float = 1e-6,
    gamma_scale: float = 1.0,
) -> SamplerResult:
    if num_chains < 3:
        raise ValueError("dream_zs requires at least 3 chains")
    rng = np.random.default_rng(seed)
    current = initialize_chains(latent_dim, num_chains, init_scale, seed, initial_z)
    current_log_prob = np.array([log_prob_fn(state) for state in current], dtype=np.float64)
    chain = np.zeros((num_steps, num_chains, latent_dim), dtype=np.float64)
    log_prob = np.zeros((num_steps, num_chains), dtype=np.float64)
    accepted = np.zeros(num_chains, dtype=np.float64)
    history = [state.copy() for state in current]

    for step in range(num_steps):
        archive = np.asarray(history, dtype=np.float64)
        for chain_index in range(num_chains):
            mask = rng.random(latent_dim) < crossover_probability
            if not np.any(mask):
                mask[rng.integers(0, latent_dim)] = True
            active_dims = int(mask.sum())

            if archive.shape[0] >= 2:
                r1, r2 = rng.choice(archive.shape[0], size=2, replace=False)
                gamma = gamma_scale * 2.38 / np.sqrt(2.0 * active_dims)
                proposal = current[chain_index].copy()
                proposal[mask] += gamma * (archive[r1, mask] - archive[r2, mask])
                proposal[mask] += rng.normal(0.0, noise_scale, size=active_dims)
            else:
                proposal = current[chain_index] + rng.normal(0.0, init_scale, size=latent_dim)

            proposal_log_prob = float(log_prob_fn(proposal))
            if np.log(rng.random()) < proposal_log_prob - current_log_prob[chain_index]:
                current[chain_index] = proposal
                current_log_prob[chain_index] = proposal_log_prob
                accepted[chain_index] += 1.0

            chain[step, chain_index] = current[chain_index]
            log_prob[step, chain_index] = current_log_prob[chain_index]

        history.extend(state.copy() for state in current)

    return SamplerResult(
        backend="dream_zs",
        chain=chain,
        log_prob=log_prob,
        acceptance_fraction=accepted / max(1, num_steps),
        metadata={
            "latent_dim": latent_dim,
            "num_chains": num_chains,
            "num_steps": num_steps,
            "seed": seed,
            "init_scale": init_scale,
            "crossover_probability": crossover_probability,
            "noise_scale": noise_scale,
            "gamma_scale": gamma_scale,
        },
    )


def save_sampler_result(path: str | Path, result: SamplerResult, burn_in: int) -> None:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "chain.npz",
        chain=result.chain,
        log_prob=result.log_prob,
        acceptance_fraction=result.acceptance_fraction,
        burn_in=int(burn_in),
    )
    summary = {
        "backend": result.backend,
        "chain_shape": list(result.chain.shape),
        "log_prob_max": float(np.max(result.log_prob)),
        "log_prob_mean": float(np.mean(result.log_prob)),
        "acceptance_fraction_mean": float(np.mean(result.acceptance_fraction)),
        "acceptance_fraction_min": float(np.min(result.acceptance_fraction)),
        "acceptance_fraction_max": float(np.max(result.acceptance_fraction)),
        "metadata": result.metadata,
    }
    with (out_dir / "sampler_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
