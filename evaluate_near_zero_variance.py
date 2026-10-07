"""Evaluate the final model near the degenerate variance boundary.

The same (s, tau, Heston parameters) points are used at every variance level.
Each point lies in the predicted continuation region for every level, so the
PDE residuals are compared on identical states apart from v.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import (
    K_ref,
    S_BOUNDS,
    T_s_ref,
    kappa_ref,
    rho_ref,
    sigma_v_ref,
    theta_ref,
)
from diagnose_clamp import load_trained_models
from evaluate_chapter4_boundary import (
    normalized_parameters,
    sample_admissible_parameters,
    summarize,
    uniform,
)
from loss import get_PDE_residual
from param_sampling import normalize_params


SEED = 59
V_LEVELS = (0.0, 0.001, 0.01, 0.05, 0.1)
CONTINUATION_MARGIN = 0.01


def evaluate_pde(price_net, boundary_net, n: int, batch_size: int) -> dict:
    torch.manual_seed(SEED)
    params = sample_admissible_parameters(n)
    params_hat = normalized_parameters(params)
    tau = uniform((n, 1), 1.0e-5, 1.0)

    with torch.no_grad():
        boundary_by_level = []
        for level in V_LEVELS:
            v = torch.full((n, 1), level)
            x_bound = torch.cat([v, tau, *params_hat], dim=1)
            boundary_by_level.append(boundary_net(x_bound))
        largest_boundary = torch.stack(boundary_by_level).max(dim=0).values
        lower_s = largest_boundary + CONTINUATION_MARGIN
        if bool((lower_s >= S_BOUNDS[1]).any()):
            raise ValueError("A sampled boundary leaves no common continuation interval")
        s = lower_s + (S_BOUNDS[1] - lower_s) * torch.rand((n, 1))
        min_distance = min(
            float((s - level_boundary).min()) for level_boundary in boundary_by_level
        )

    rows = []
    for level in V_LEVELS:
        residual_batches = []
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            s_batch = s[start:stop].detach().requires_grad_(True)
            v_batch = torch.full_like(s_batch, level).requires_grad_(True)
            tau_batch = tau[start:stop].detach().requires_grad_(True)
            phys_batch = (*[x[start:stop] for x in params], torch.ones_like(s_batch))
            norm_batch = (*[x[start:stop] for x in params_hat], torch.ones_like(s_batch))
            residual = get_PDE_residual(
                s_batch,
                v_batch,
                tau_batch,
                phys_batch,
                norm_batch,
                torch.zeros_like(s_batch),
                price_net,
            )
            residual_batches.append(residual.detach().abs().flatten())
        stats = summarize(torch.cat(residual_batches))
        rows.append({"v": level, **stats})
        print(
            f"v={level:.3f}  mean={stats['mean']:.6f}  "
            f"p95={stats['p95']:.6f}  max={stats['max']:.6f}",
            flush=True,
        )

    return {
        "n_common_continuation_points": n,
        "seed": SEED,
        "tau_range": [1.0e-5, 1.0],
        "normalized_s_range": list(S_BOUNDS),
        "continuation_margin": CONTINUATION_MARGIN,
        "minimum_distance_to_predicted_boundary": min_distance,
        "parameters": "uniform rejection sample over Feller admissible set",
        "statistics": rows,
    }


def evaluate_benchmark_boundary(boundary_net) -> dict:
    levels = torch.tensor(V_LEVELS, dtype=torch.float32).reshape(-1, 1)
    tau = torch.full_like(levels, T_s_ref)
    kappa = torch.full_like(levels, kappa_ref)
    theta = torch.full_like(levels, theta_ref)
    sigma_v = torch.full_like(levels, sigma_v_ref)
    rho = torch.full_like(levels, rho_ref)
    params_hat = normalize_params(kappa, theta, sigma_v, rho, tau)[:4]

    with torch.no_grad():
        x_bound = torch.cat([levels, tau, *params_hat], dim=1)
        boundary_values = K_ref * boundary_net(x_bound).flatten()

        fine_v = torch.linspace(0.0, 0.1, 501).reshape(-1, 1)
        fine_tau = torch.full_like(fine_v, T_s_ref)
        fine_hat = normalize_params(
            torch.full_like(fine_v, kappa_ref),
            torch.full_like(fine_v, theta_ref),
            torch.full_like(fine_v, sigma_v_ref),
            torch.full_like(fine_v, rho_ref),
            fine_tau,
        )[:4]
        fine_boundary = K_ref * boundary_net(
            torch.cat([fine_v, fine_tau, *fine_hat], dim=1)
        ).flatten()
        first_differences = torch.diff(fine_boundary)

    return {
        "benchmark_parameters": {
            "K": K_ref,
            "tau": T_s_ref,
            "kappa": kappa_ref,
            "theta": theta_ref,
            "sigma_v": sigma_v_ref,
            "rho": rho_ref,
        },
        "rows": [
            {"v": level, "predicted_S_star": float(boundary_values[i])}
            for i, level in enumerate(V_LEVELS)
        ],
        "fine_grid": {
            "v_range": [0.0, 0.1],
            "n": len(fine_v),
            "positive_steps": int((first_differences > 0).sum()),
            "negative_steps": int((first_differences < 0).sum()),
            "max_absolute_step": float(first_differences.abs().max()),
            "max_absolute_second_difference": float(torch.diff(first_differences).abs().max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="saved_models")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output", default="output/diagnostics/near_zero_variance.json"
    )
    args = parser.parse_args()
    if args.n <= 0 or args.batch_size <= 0:
        parser.error("--n and --batch-size must be positive")

    price_net, boundary_net = load_trained_models(args.checkpoint_dir)
    # Keep this post-training diagnostic on CPU for reproducible batching.
    price_net = price_net.cpu()
    boundary_net = boundary_net.cpu()
    results = {
        "checkpoint_directory": args.checkpoint_dir,
        "pde_residual": evaluate_pde(price_net, boundary_net, args.n, args.batch_size),
        "benchmark_boundary": evaluate_benchmark_boundary(boundary_net),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
