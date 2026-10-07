"""Evaluate the final model near the upper variance boundary.

The same ``(s, tau, mu)`` points are used at every variance level. Each
point lies in the predicted continuation region for all levels, so changes
in the diagnostics are not caused by different evaluation samples.
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
V_LEVELS = (0.80, 0.90, 0.95, 1.00)
CONTINUATION_MARGIN = 0.01


def evaluate_common_points(price_net, boundary_net, n: int, batch_size: int) -> dict:
    torch.manual_seed(SEED)
    params = sample_admissible_parameters(n)
    params_hat = normalized_parameters(params)
    tau = uniform((n, 1), 1.0e-5, 1.0)

    with torch.no_grad():
        boundaries = []
        for level in V_LEVELS:
            v = torch.full((n, 1), level)
            boundaries.append(boundary_net(torch.cat([v, tau, *params_hat], dim=1)))
        largest_boundary = torch.stack(boundaries).max(dim=0).values
        lower_s = largest_boundary + CONTINUATION_MARGIN
        if bool((lower_s >= S_BOUNDS[1]).any()):
            raise ValueError("A sampled boundary leaves no common continuation interval")
        s = lower_s + (S_BOUNDS[1] - lower_s) * torch.rand((n, 1))
        minimum_distance = min(
            float((s - boundary).min()) for boundary in boundaries
        )

    rows: list[dict[str, object]] = []
    for level in V_LEVELS:
        residual_batches: list[torch.Tensor] = []
        derivative_batches: list[torch.Tensor] = []

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            s_batch = s[start:stop].detach().requires_grad_(True)
            v_batch = torch.full_like(s_batch, level).requires_grad_(True)
            tau_batch = tau[start:stop].detach().requires_grad_(True)
            physical_batch = tuple(x[start:stop] for x in params)
            normalized_batch = tuple(x[start:stop] for x in params_hat)
            dummy_horizon = torch.ones_like(s_batch)

            residual = get_PDE_residual(
                s_batch,
                v_batch,
                tau_batch,
                (*physical_batch, dummy_horizon),
                (*normalized_batch, dummy_horizon),
                torch.zeros_like(s_batch),
                price_net,
            )
            residual_batches.append(residual.detach().abs().flatten())

            v_first = torch.full_like(s_batch, level).requires_grad_(True)
            price = price_net(
                torch.cat(
                    [s_batch.detach(), v_first, tau_batch.detach(), *normalized_batch],
                    dim=1,
                )
            )
            p_v = torch.autograd.grad(
                price,
                v_first,
                grad_outputs=torch.ones_like(price),
                create_graph=False,
            )[0]
            derivative_batches.append(p_v.detach().abs().flatten())

        residual_stats = summarize(torch.cat(residual_batches))
        derivative_stats = summarize(torch.cat(derivative_batches))
        rows.append(
            {
                "v": level,
                "absolute_pde_residual": residual_stats,
                "absolute_normalized_price_derivative": derivative_stats,
            }
        )
        print(
            f"v={level:.2f}  residual mean={residual_stats['mean']:.6f}  "
            f"|p_v| mean={derivative_stats['mean']:.6f}  "
            f"|p_v| p95={derivative_stats['p95']:.6f}",
            flush=True,
        )

    return {
        "n_common_continuation_points": n,
        "seed": SEED,
        "tau_range": [1.0e-5, 1.0],
        "normalized_s_range": list(S_BOUNDS),
        "continuation_margin": CONTINUATION_MARGIN,
        "minimum_distance_to_predicted_boundary": minimum_distance,
        "parameters": "uniform rejection sample over Feller admissible set",
        "price_derivative": "derivative of normalized price p with respect to v",
        "rows": rows,
    }


def evaluate_benchmark_boundary(boundary_net, n_grid: int = 201) -> dict:
    v = torch.linspace(0.8, 1.0, n_grid).reshape(-1, 1).requires_grad_(True)
    tau = torch.full_like(v, T_s_ref)
    kappa = torch.full_like(v, kappa_ref)
    theta = torch.full_like(v, theta_ref)
    sigma_v = torch.full_like(v, sigma_v_ref)
    rho = torch.full_like(v, rho_ref)
    params_hat = normalize_params(kappa, theta, sigma_v, rho, tau)[:4]

    boundary = K_ref * boundary_net(torch.cat([v, tau, *params_hat], dim=1))
    derivative = torch.autograd.grad(
        boundary,
        v,
        grad_outputs=torch.ones_like(boundary),
        create_graph=False,
    )[0]

    boundary_values = boundary.detach().flatten()
    derivative_values = derivative.detach().flatten()
    steps = torch.diff(boundary_values)

    return {
        "benchmark_parameters": {
            "K": K_ref,
            "tau": T_s_ref,
            "kappa": kappa_ref,
            "theta": theta_ref,
            "sigma_v": sigma_v_ref,
            "rho": rho_ref,
        },
        "v_range": [0.8, 1.0],
        "n_grid": n_grid,
        "S_star_at_v_0_8": float(boundary_values[0]),
        "S_star_at_v_1_0": float(boundary_values[-1]),
        "positive_steps": int((steps > 0).sum()),
        "negative_steps": int((steps < 0).sum()),
        "max_absolute_step": float(steps.abs().max()),
        "dS_star_dv_min": float(derivative_values.min()),
        "dS_star_dv_max": float(derivative_values.max()),
        "dS_star_dv_mean": float(derivative_values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="saved_models")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output", default="output/diagnostics/upper_variance.json"
    )
    args = parser.parse_args()
    if args.n <= 0 or args.batch_size <= 0:
        parser.error("--n and --batch-size must be positive")

    price_net, boundary_net = load_trained_models(args.checkpoint_dir)
    price_net = price_net.cpu()
    boundary_net = boundary_net.cpu()
    results = {
        "checkpoint_directory": args.checkpoint_dir,
        "common_point_diagnostics": evaluate_common_points(
            price_net, boundary_net, args.n, args.batch_size
        ),
        "benchmark_boundary": evaluate_benchmark_boundary(boundary_net),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
