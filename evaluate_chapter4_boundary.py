"""Independent Chapter 4 diagnostics for the final proposed model.

The script reports:
  1. value matching, smooth pasting, and boundary maturity residuals;
  2. raw boundary output range;
  3. raw price network error inside the predicted exercise region; and
  4. intrinsic value lower bound violations before the final projection.

All statistics use fresh samples and seed 59. Heston parameter tuples are
sampled uniformly from the stated box and rejected unless they satisfy the
Feller margin used in the thesis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from config import (
    FELLER_MARGIN,
    KAPPA_BOUNDS,
    RHO_BOUNDS,
    SIGMA_V_BOUNDS,
    THETA_BOUNDS,
)
from models import create_models
from param_sampling import normalize_params


SEED = 59
N_TEST = 200_000
BATCH_SIZE = 4_096
V_MIN, V_MAX = 0.01, 1.0
TAU_MIN, TAU_MAX = 0.0, 1.0
S_MIN, S_MAX = 0.0, 2.0
NEAR_BOUNDARY_BAND = 0.05


def uniform(shape: tuple[int, ...], lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, dtype=torch.float32)


def sample_admissible_parameters(n: int) -> tuple[torch.Tensor, ...]:
    """Uniform rejection sample from the admissible Heston parameter box."""
    accepted: list[torch.Tensor] = []
    remaining = n
    while remaining > 0:
        proposal_n = max(50_000, int(remaining * 1.8))
        kappa = uniform((proposal_n, 1), *KAPPA_BOUNDS)
        theta = uniform((proposal_n, 1), *THETA_BOUNDS)
        sigma_v = uniform((proposal_n, 1), *SIGMA_V_BOUNDS)
        rho = uniform((proposal_n, 1), *RHO_BOUNDS)
        keep = sigma_v <= FELLER_MARGIN * torch.sqrt(2.0 * kappa * theta)
        block = torch.cat([kappa, theta, sigma_v, rho], dim=1)[keep.flatten()]
        if block.numel() == 0:
            continue
        block = block[:remaining]
        accepted.append(block)
        remaining -= block.shape[0]
    values = torch.cat(accepted, dim=0)
    return tuple(values[:, j : j + 1] for j in range(4))


def normalized_parameters(params: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    kappa, theta, sigma_v, rho = params
    dummy_horizon = torch.full_like(kappa, 0.25)
    return normalize_params(kappa, theta, sigma_v, rho, dummy_horizon)[:4]


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().flatten().cpu()
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "p95": torch.quantile(values, 0.95).item(),
        "max": values.max().item(),
    }


def violation_summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().flatten().cpu()
    active = values > 0
    triggered = values[active]
    if triggered.numel() == 0:
        return {
            "n": values.numel(),
            "n_violated": 0,
            "violation_rate": 0.0,
            "mean_when_violated": 0.0,
            "p95_when_violated": 0.0,
            "max": 0.0,
        }
    return {
        "n": values.numel(),
        "n_violated": triggered.numel(),
        "violation_rate": active.float().mean().item(),
        "mean_when_violated": triggered.mean().item(),
        "p95_when_violated": torch.quantile(triggered, 0.95).item(),
        "max": triggered.max().item(),
    }


def evaluate_boundary_conditions(price_net, boundary_net) -> dict[str, object]:
    params = sample_admissible_parameters(N_TEST)
    params_hat = normalized_parameters(params)
    v = uniform((N_TEST, 1), V_MIN, V_MAX)
    tau = uniform((N_TEST, 1), 1.0e-6, TAU_MAX)

    vm_values: list[torch.Tensor] = []
    sp_values: list[torch.Tensor] = []
    boundary_values: list[torch.Tensor] = []

    for start in range(0, N_TEST, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_TEST)
        v_b = v[start:stop]
        tau_b = tau[start:stop]
        p_hat_b = tuple(x[start:stop] for x in params_hat)

        with torch.no_grad():
            x_boundary = torch.cat([v_b, tau_b, *p_hat_b], dim=1)
            s_star = boundary_net(x_boundary)
        s_eval = s_star.detach().requires_grad_(True)
        x_price = torch.cat([s_eval, v_b, tau_b, *p_hat_b], dim=1)
        price = price_net(x_price)
        price_s = torch.autograd.grad(
            price,
            s_eval,
            grad_outputs=torch.ones_like(price),
            create_graph=False,
        )[0]

        vm_values.append((price.detach() - (1.0 - s_star)).abs())
        sp_values.append((price_s.detach() + 1.0).abs())
        boundary_values.append(s_star.detach())

    vm = torch.cat(vm_values)
    sp = torch.cat(sp_values)
    s_star_all = torch.cat(boundary_values)

    # Boundary maturity condition on a separate, independently sampled set.
    params_t = sample_admissible_parameters(N_TEST)
    params_t_hat = normalized_parameters(params_t)
    v_t = uniform((N_TEST, 1), V_MIN, V_MAX)
    tau_t = torch.zeros((N_TEST, 1), dtype=torch.float32)
    bt_values: list[torch.Tensor] = []
    for start in range(0, N_TEST, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_TEST)
        x_boundary = torch.cat(
            [v_t[start:stop], tau_t[start:stop], *(x[start:stop] for x in params_t_hat)],
            dim=1,
        )
        with torch.no_grad():
            bt_values.append((boundary_net(x_boundary) - 1.0).abs())
    bt = torch.cat(bt_values)

    return {
        "n": N_TEST,
        "value_matching": summarize(vm),
        "smooth_pasting": summarize(sp),
        "boundary_maturity": summarize(bt),
        "raw_boundary": {
            "min": s_star_all.min().item(),
            "max": s_star_all.max().item(),
            "mean": s_star_all.mean().item(),
            "fraction_nonpositive": (s_star_all <= 0).float().mean().item(),
            "fraction_above_one": (s_star_all > 1).float().mean().item(),
        },
    }


def evaluate_exercise_region(price_net, boundary_net) -> dict[str, object]:
    params = sample_admissible_parameters(N_TEST)
    params_hat = normalized_parameters(params)
    v = uniform((N_TEST, 1), V_MIN, V_MAX)
    tau = uniform((N_TEST, 1), TAU_MIN, TAU_MAX)

    with torch.no_grad():
        x_boundary = torch.cat([v, tau, *params_hat], dim=1)
        s_star = boundary_net(x_boundary)

    # Sample from the region defined by the raw boundary output. The separate
    # boundary range diagnostic records any nonpositive predictions.
    s_star_for_sampling = s_star.clamp_min(0.0)
    s = torch.rand((N_TEST, 1), dtype=torch.float32) * s_star_for_sampling
    payoff = torch.relu(1.0 - s)

    errors: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, N_TEST, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, N_TEST)
            x_price = torch.cat(
                [s[start:stop], v[start:stop], tau[start:stop], *(x[start:stop] for x in params_hat)],
                dim=1,
            )
            errors.append((price_net(x_price) - payoff[start:stop]).abs())

    return {
        "n": N_TEST,
        "payoff_error": summarize(torch.cat(errors)),
    }


def evaluate_lower_bound(price_net, boundary_net) -> dict[str, object]:
    params = sample_admissible_parameters(N_TEST)
    params_hat = normalized_parameters(params)
    s = uniform((N_TEST, 1), S_MIN, S_MAX)
    v = uniform((N_TEST, 1), V_MIN, V_MAX)
    tau = uniform((N_TEST, 1), TAU_MIN, TAU_MAX)

    violations: list[torch.Tensor] = []
    near_violations: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, N_TEST, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, N_TEST)
            s_b = s[start:stop]
            v_b = v[start:stop]
            tau_b = tau[start:stop]
            p_hat_b = tuple(x[start:stop] for x in params_hat)
            s_star = boundary_net(torch.cat([v_b, tau_b, *p_hat_b], dim=1))
            p_raw = price_net(torch.cat([s_b, v_b, tau_b, *p_hat_b], dim=1))
            payoff = torch.relu(1.0 - s_b)
            assembled = torch.where(s_b < s_star, payoff, p_raw)
            violation = torch.relu(payoff - assembled)
            violations.append(violation)
            near = (s_b - s_star).abs() < NEAR_BOUNDARY_BAND
            if near.any():
                near_violations.append(violation[near])

    all_values = torch.cat(violations)
    near_values = torch.cat(near_violations) if near_violations else torch.empty(0)
    return {
        "all_domain": violation_summary(all_values),
        "near_boundary_band": NEAR_BOUNDARY_BAND,
        "near_boundary": violation_summary(near_values),
    }


def format_metric_row(name: str, values: dict[str, float]) -> str:
    return (
        f"{name:<24} {values['mean']:>12.6f} {values['median']:>12.6f} "
        f"{values['p95']:>12.6f} {values['max']:>12.6f}"
    )


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Build and load the networks on CPU so the random stream (weight
    # initialisation consumes it) and the sampled states are identical on
    # every machine, whatever config.device is.
    price_net, boundary_net = create_models(torch.device("cpu"))
    price_net.load_state_dict(torch.load("saved_models/solutionNN.pth", map_location="cpu"))
    boundary_net.load_state_dict(torch.load("saved_models/boundaryNN.pth", map_location="cpu"))
    price_net.eval()
    boundary_net.eval()
    boundary = evaluate_boundary_conditions(price_net, boundary_net)
    exercise = evaluate_exercise_region(price_net, boundary_net)
    lower_bound = evaluate_lower_bound(price_net, boundary_net)

    results = {
        "seed": SEED,
        "checkpoint_directory": "saved_models",
        "sampling": {
            "state": "uniform",
            "parameters": "uniform rejection sample from the Feller admissible box",
            "v_range": [V_MIN, V_MAX],
            "tau_range": [TAU_MIN, TAU_MAX],
            "s_range": [S_MIN, S_MAX],
        },
        "boundary_conditions": boundary,
        "exercise_region": exercise,
        "intrinsic_lower_bound": lower_bound,
    }

    output_json = Path("chapter4_boundary_results.json")
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Boundary condition residuals (normalized units)")
    print(f"{'Metric':<24} {'Mean':>12} {'Median':>12} {'P95':>12} {'Maximum':>12}")
    print(format_metric_row("Value matching", boundary["value_matching"]))
    print(format_metric_row("Smooth pasting", boundary["smooth_pasting"]))
    print(format_metric_row("Boundary at maturity", boundary["boundary_maturity"]))
    print()
    print("Raw boundary output")
    print(json.dumps(boundary["raw_boundary"], indent=2))
    print()
    print("Exercise region payoff error (normalized units)")
    print(format_metric_row("Payoff error", exercise["payoff_error"]))
    print()
    print("Intrinsic value lower bound violation before final projection")
    print(json.dumps(lower_bound, indent=2))
    print(f"\nSaved full precision results to {output_json}")


if __name__ == "__main__":
    main()
