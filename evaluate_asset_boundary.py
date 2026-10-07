"""Evaluate the trained model near the upper asset-price boundary.

The same ``(v, tau, mu)`` sample is used at every asset-price level, so
changes between rows are caused only by ``s``. Parameters are sampled
uniformly from the Feller-admissible domain, following the other Chapter 4
diagnostics. No retraining is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import S_BOUNDS
from diagnose_clamp import load_trained_models
from evaluate_chapter4_boundary import (
    normalized_parameters,
    sample_admissible_parameters,
    summarize,
    uniform,
)
from loss import get_PDE_residual


SEED = 59
S_LEVELS = (1.60, 1.80, 1.90, 1.95, 2.00)
V_MIN, V_MAX = 0.01, 1.00
TAU_MIN, TAU_MAX = 1.0e-5, 1.00


def evaluate_common_points(
    price_net, boundary_net, n: int, batch_size: int
) -> dict[str, object]:
    """Evaluate raw price, asset slope, and PDE residual at fixed s levels."""
    torch.manual_seed(SEED)
    params = sample_admissible_parameters(n)
    params_hat = normalized_parameters(params)
    v = uniform((n, 1), V_MIN, V_MAX)
    tau = uniform((n, 1), TAU_MIN, TAU_MAX)

    with torch.no_grad():
        boundary = boundary_net(torch.cat([v, tau, *params_hat], dim=1))
    maximum_boundary = float(boundary.max())
    if maximum_boundary >= min(S_LEVELS):
        raise ValueError(
            "The lowest asset level is not in the continuation region for all points"
        )

    rows: list[dict[str, object]] = []
    for s_level in S_LEVELS:
        price_batches: list[torch.Tensor] = []
        slope_batches: list[torch.Tensor] = []
        positive_slope_batches: list[torch.Tensor] = []
        residual_batches: list[torch.Tensor] = []

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            v_batch = v[start:stop]
            tau_batch = tau[start:stop]
            physical_batch = tuple(x[start:stop] for x in params)
            normalized_batch = tuple(x[start:stop] for x in params_hat)

            # Raw price and its untrained far-field slope condition.
            s_first = torch.full_like(v_batch, s_level).requires_grad_(True)
            price = price_net(
                torch.cat(
                    [s_first, v_batch, tau_batch, *normalized_batch], dim=1
                )
            )
            p_s = torch.autograd.grad(
                price,
                s_first,
                grad_outputs=torch.ones_like(price),
                create_graph=False,
            )[0]
            price_batches.append(price.detach().abs().flatten())
            slope_batches.append(p_s.detach().abs().flatten())
            positive_slope_batches.append((p_s.detach() > 0).flatten())

            # The PDE is an interior condition, so it is not evaluated at s_max.
            if s_level < S_BOUNDS[1]:
                s_pde = torch.full_like(v_batch, s_level).requires_grad_(True)
                v_pde = v_batch.detach().requires_grad_(True)
                tau_pde = tau_batch.detach().requires_grad_(True)
                dummy_horizon = torch.ones_like(s_pde)
                residual = get_PDE_residual(
                    s_pde,
                    v_pde,
                    tau_pde,
                    (*physical_batch, dummy_horizon),
                    (*normalized_batch, dummy_horizon),
                    torch.zeros_like(s_pde),
                    price_net,
                )
                residual_batches.append(residual.detach().abs().flatten())

        price_stats = summarize(torch.cat(price_batches))
        slope_stats = summarize(torch.cat(slope_batches))
        positive_slope_rate = float(
            torch.cat(positive_slope_batches).float().mean()
        )
        residual_stats = (
            summarize(torch.cat(residual_batches)) if residual_batches else None
        )
        rows.append(
            {
                "s": s_level,
                "absolute_raw_normalized_price": price_stats,
                "absolute_asset_derivative": slope_stats,
                "positive_slope_rate": positive_slope_rate,
                "absolute_pde_residual": residual_stats,
            }
        )

        residual_text = (
            f"{residual_stats['mean']:.6f}  {residual_stats['p95']:.6f}"
            if residual_stats is not None
            else "       --          --"
        )
        print(
            f"s={s_level:.2f}  "
            f"|p| mean={price_stats['mean']:.6f}  p95={price_stats['p95']:.6f}  "
            f"|p_s| mean={slope_stats['mean']:.6f}  p95={slope_stats['p95']:.6f}  "
            f"positive slope={positive_slope_rate:.2%}  "
            f"|PDE| mean/p95={residual_text}",
            flush=True,
        )

    return {
        "seed": SEED,
        "sample_size": n,
        "s_levels": list(S_LEVELS),
        "v_range": [V_MIN, V_MAX],
        "tau_range": [TAU_MIN, TAU_MAX],
        "parameters": "uniform rejection sample over the Feller-admissible box",
        "same_points_at_each_s": True,
        "maximum_predicted_boundary": maximum_boundary,
        "price_quantity": "raw normalized price p",
        "pde_at_s_max": "not evaluated because the PDE is an interior condition",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="saved_models")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output", default="output/diagnostics/upper_asset_boundary.json"
    )
    args = parser.parse_args()
    if args.n <= 0 or args.batch_size <= 0:
        parser.error("--n and --batch-size must be positive")

    price_net, boundary_net = load_trained_models(args.checkpoint_dir)
    price_net = price_net.cpu()
    boundary_net = boundary_net.cpu()
    results = evaluate_common_points(price_net, boundary_net, args.n, args.batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
