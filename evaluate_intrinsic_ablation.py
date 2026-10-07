"""Evaluate intrinsic-loss ablation checkpoints on common test samples.

The four configurations differ only in the weight assigned to the intrinsic
value loss.  All domain-wide statistics use the same random seed and the same
base samples for every checkpoint.  Residuals are reported in normalized
units; benchmark price differences are reported in asset-price units.
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from benchmark_data import IKONEN_TOIVANEN, S_VALS
from config import (
    FELLER_MARGIN,
    KAPPA_BOUNDS,
    K_ref,
    RHO_BOUNDS,
    SIGMA_V_BOUNDS,
    THETA_BOUNDS,
    kappa_ref,
    rho_ref,
    sigma_v_ref,
    theta_ref,
)
from loss import get_PDE_residual
from models import create_models
from param_sampling import normalize_params


SEED = 59
N_TEST = 200_000
N_PDE = 50_000
BATCH_SIZE = 4_096
V_MIN, V_MAX = 0.01, 1.0
TAU_MIN, TAU_MAX = 0.0, 1.0
S_MIN, S_MAX = 0.0, 2.0
NEAR_BOUNDARY_BAND = 0.05
DEVICE = torch.device("cpu")

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = OrderedDict(
    [
        (0, ROOT / "saved_models_ablation_no_intrinsic"),
        (5, ROOT / "saved_models_ablation_intrinsic_w5"),
        (10, ROOT / "saved_models"),
        (20, ROOT / "saved_models_ablation_intrinsic_w20"),
    ]
)
OUTPUT_DIR = ROOT / "output" / "ablation"


def uniform(shape: tuple[int, ...], lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, dtype=torch.float32, device=DEVICE)


def sample_admissible_parameters(n: int) -> tuple[torch.Tensor, ...]:
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


def make_samples() -> dict[str, object]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    samples: dict[str, object] = {}
    # Keep the draw order used by evaluate_chapter4_boundary.py.  This makes
    # the weight-10 row directly reproducible from the diagnostics already
    # reported in Tables 4.6--4.8.
    params = sample_admissible_parameters(N_TEST)
    samples["boundary_params"] = params
    samples["boundary_params_hat"] = normalized_parameters(params)
    samples["boundary_v"] = uniform((N_TEST, 1), V_MIN, V_MAX)
    samples["boundary_tau"] = uniform((N_TEST, 1), 1.0e-6, TAU_MAX)

    params = sample_admissible_parameters(N_TEST)
    samples["maturity_params"] = params
    samples["maturity_params_hat"] = normalized_parameters(params)
    samples["maturity_v"] = uniform((N_TEST, 1), V_MIN, V_MAX)

    params = sample_admissible_parameters(N_TEST)
    samples["exercise_params"] = params
    samples["exercise_params_hat"] = normalized_parameters(params)
    samples["exercise_v"] = uniform((N_TEST, 1), V_MIN, V_MAX)
    samples["exercise_tau"] = uniform((N_TEST, 1), TAU_MIN, TAU_MAX)
    samples["exercise_u"] = torch.rand((N_TEST, 1), dtype=torch.float32)

    params = sample_admissible_parameters(N_TEST)
    samples["lower_params"] = params
    samples["lower_params_hat"] = normalized_parameters(params)
    samples["lower_s"] = uniform((N_TEST, 1), S_MIN, S_MAX)
    samples["lower_v"] = uniform((N_TEST, 1), V_MIN, V_MAX)
    samples["lower_tau"] = uniform((N_TEST, 1), TAU_MIN, TAU_MAX)

    params = sample_admissible_parameters(N_PDE)
    samples["pde_params"] = params
    samples["pde_params_hat"] = normalized_parameters(params)
    samples["pde_v"] = uniform((N_PDE, 1), V_MIN, V_MAX)
    samples["pde_tau"] = uniform((N_PDE, 1), 1.0e-6, TAU_MAX)
    samples["pde_u"] = torch.rand((N_PDE, 1), dtype=torch.float32)
    return samples


def load_models(checkpoint_dir: Path):
    price_net, boundary_net = create_models(DEVICE)
    price_state = torch.load(
        checkpoint_dir / "solutionNN.pth", map_location=DEVICE, weights_only=True
    )
    boundary_state = torch.load(
        checkpoint_dir / "boundaryNN.pth", map_location=DEVICE, weights_only=True
    )
    price_net.load_state_dict(price_state)
    boundary_net.load_state_dict(boundary_state)
    price_net.eval()
    boundary_net.eval()
    return price_net, boundary_net


def evaluate_boundary(price_net, boundary_net, samples: dict[str, object]) -> dict[str, object]:
    v = samples["boundary_v"]
    tau = samples["boundary_tau"]
    params_hat = samples["boundary_params_hat"]
    vm_values: list[torch.Tensor] = []
    sp_values: list[torch.Tensor] = []
    raw_boundary: list[torch.Tensor] = []

    for start in range(0, N_TEST, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_TEST)
        v_b, tau_b = v[start:stop], tau[start:stop]
        p_hat_b = tuple(x[start:stop] for x in params_hat)
        with torch.no_grad():
            s_star = boundary_net(torch.cat([v_b, tau_b, *p_hat_b], dim=1))
        s_eval = s_star.detach().requires_grad_(True)
        price = price_net(torch.cat([s_eval, v_b, tau_b, *p_hat_b], dim=1))
        price_s = torch.autograd.grad(
            price, s_eval, grad_outputs=torch.ones_like(price), create_graph=False
        )[0]
        vm_values.append((price.detach() - (1.0 - s_star)).abs())
        sp_values.append((price_s.detach() + 1.0).abs())
        raw_boundary.append(s_star.detach())

    v_t = samples["maturity_v"]
    params_t_hat = samples["maturity_params_hat"]
    bt_values: list[torch.Tensor] = []
    for start in range(0, N_TEST, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_TEST)
        tau_t = torch.zeros((stop - start, 1), dtype=torch.float32)
        with torch.no_grad():
            value = boundary_net(
                torch.cat([v_t[start:stop], tau_t, *(x[start:stop] for x in params_t_hat)], dim=1)
            )
        bt_values.append((value - 1.0).abs())

    boundary_all = torch.cat(raw_boundary)
    return {
        "n": N_TEST,
        "value_matching": summarize(torch.cat(vm_values)),
        "smooth_pasting": summarize(torch.cat(sp_values)),
        "boundary_at_maturity": summarize(torch.cat(bt_values)),
        "raw_boundary": {
            "min": boundary_all.min().item(),
            "max": boundary_all.max().item(),
            "fraction_nonpositive": (boundary_all <= 0).float().mean().item(),
            "fraction_above_one": (boundary_all > 1).float().mean().item(),
        },
    }


def evaluate_exercise(price_net, boundary_net, samples: dict[str, object]) -> dict[str, object]:
    v = samples["exercise_v"]
    tau = samples["exercise_tau"]
    u = samples["exercise_u"]
    params_hat = samples["exercise_params_hat"]
    errors: list[torch.Tensor] = []

    for start in range(0, N_TEST, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_TEST)
        v_b, tau_b, u_b = v[start:stop], tau[start:stop], u[start:stop]
        p_hat_b = tuple(x[start:stop] for x in params_hat)
        with torch.no_grad():
            s_star = boundary_net(torch.cat([v_b, tau_b, *p_hat_b], dim=1))
            s = u_b * s_star.clamp_min(0.0)
            payoff = torch.relu(1.0 - s)
            raw_price = price_net(torch.cat([s, v_b, tau_b, *p_hat_b], dim=1))
        errors.append((raw_price - payoff).abs())
    return {"n": N_TEST, "payoff_residual": summarize(torch.cat(errors))}


def evaluate_lower_bound(price_net, boundary_net, samples: dict[str, object]) -> dict[str, object]:
    s = samples["lower_s"]
    v = samples["lower_v"]
    tau = samples["lower_tau"]
    params_hat = samples["lower_params_hat"]
    all_values: list[torch.Tensor] = []
    near_values: list[torch.Tensor] = []

    for start in range(0, N_TEST, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_TEST)
        s_b, v_b, tau_b = s[start:stop], v[start:stop], tau[start:stop]
        p_hat_b = tuple(x[start:stop] for x in params_hat)
        with torch.no_grad():
            s_star = boundary_net(torch.cat([v_b, tau_b, *p_hat_b], dim=1))
            raw_price = price_net(torch.cat([s_b, v_b, tau_b, *p_hat_b], dim=1))
            payoff = torch.relu(1.0 - s_b)
            violation = torch.where(
                s_b < s_star, torch.zeros_like(s_b), torch.relu(payoff - raw_price)
            )
        all_values.append(violation)
        near = (s_b - s_star).abs() < NEAR_BOUNDARY_BAND
        if near.any():
            near_values.append(violation[near])

    return {
        "all_domain": violation_summary(torch.cat(all_values)),
        "near_boundary_band": NEAR_BOUNDARY_BAND,
        "near_boundary": violation_summary(torch.cat(near_values)),
    }


def evaluate_pde(price_net, boundary_net, samples: dict[str, object]) -> dict[str, object]:
    v = samples["pde_v"]
    tau = samples["pde_tau"]
    u = samples["pde_u"]
    params = samples["pde_params"]
    params_hat = samples["pde_params_hat"]
    values: list[torch.Tensor] = []

    for start in range(0, N_PDE, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_PDE)
        v_base, tau_base, u_b = v[start:stop], tau[start:stop], u[start:stop]
        p_b = tuple(x[start:stop] for x in params)
        p_hat_b = tuple(x[start:stop] for x in params_hat)
        with torch.no_grad():
            s_star = boundary_net(torch.cat([v_base, tau_base, *p_hat_b], dim=1))
            s_base = s_star.clamp(S_MIN, S_MAX * 0.99)
            s_base = s_base + (S_MAX - s_base) * (u_b ** 3)
        s_b = s_base.detach().requires_grad_(True)
        v_b = v_base.detach().requires_grad_(True)
        tau_b = tau_base.detach().requires_grad_(True)
        dummy_horizon = torch.full_like(s_b, 0.25)
        residual = get_PDE_residual(
            s_b,
            v_b,
            tau_b,
            (*p_b, dummy_horizon),
            (*p_hat_b, dummy_horizon),
            torch.zeros_like(s_b),
            price_net,
        )
        values.append(residual.detach().abs())
    return {"n": N_PDE, "absolute_residual": summarize(torch.cat(values))}


def evaluate_benchmark(price_net, boundary_net) -> dict[str, object]:
    prices: list[float] = []
    references: list[float] = []
    for variance in (0.0625, 0.25):
        n = len(S_VALS)
        s = torch.tensor(S_VALS, dtype=torch.float32).reshape(-1, 1) / K_ref
        v = torch.full((n, 1), variance, dtype=torch.float32)
        tau = torch.full((n, 1), 0.25, dtype=torch.float32)
        kappa = torch.full((n, 1), kappa_ref, dtype=torch.float32)
        theta = torch.full((n, 1), theta_ref, dtype=torch.float32)
        sigma_v = torch.full((n, 1), sigma_v_ref, dtype=torch.float32)
        rho = torch.full((n, 1), rho_ref, dtype=torch.float32)
        horizon = torch.full((n, 1), 0.25, dtype=torch.float32)
        params_hat = normalize_params(kappa, theta, sigma_v, rho, horizon)[:4]
        with torch.no_grad():
            s_star = boundary_net(torch.cat([v, tau, *params_hat], dim=1))
            raw_price = price_net(torch.cat([s, v, tau, *params_hat], dim=1))
            payoff = torch.relu(1.0 - s)
            assembled = torch.where(s < s_star, payoff, raw_price)
            reported = assembled.clamp(min=payoff).clamp(max=1.0) * K_ref
        prices.extend(reported.flatten().tolist())
        references.extend(IKONEN_TOIVANEN[variance])

    price_tensor = torch.tensor(prices)
    reference_tensor = torch.tensor(references)
    differences = (price_tensor - reference_tensor).abs()
    return {
        "n": len(prices),
        "reference_precision": "four-decimal values stored in benchmark_data.py",
        "prices": prices,
        "mean_absolute_difference": differences.mean().item(),
        "maximum_absolute_difference": differences.max().item(),
    }


def csv_row(weight: int, result: dict[str, object]) -> dict[str, float | int]:
    exercise = result["exercise_region"]["payoff_residual"]
    lower = result["intrinsic_lower_bound"]["all_domain"]
    boundary = result["boundary_conditions"]
    pde = result["pde"]["absolute_residual"]
    benchmark = result["benchmark"]
    return {
        "weight": weight,
        "exercise_mean": exercise["mean"],
        "exercise_p95": exercise["p95"],
        "exercise_max": exercise["max"],
        "violation_rate": lower["violation_rate"],
        "shortfall_mean_conditional": lower["mean_when_violated"],
        "shortfall_p95_conditional": lower["p95_when_violated"],
        "shortfall_max": lower["max"],
        "benchmark_mean_abs_difference": benchmark["mean_absolute_difference"],
        "benchmark_max_abs_difference": benchmark["maximum_absolute_difference"],
        "pde_mean_abs_residual": pde["mean"],
        "pde_p95_abs_residual": pde["p95"],
        "value_matching_mean": boundary["value_matching"]["mean"],
        "value_matching_p95": boundary["value_matching"]["p95"],
        "smooth_pasting_mean": boundary["smooth_pasting"]["mean"],
        "smooth_pasting_p95": boundary["smooth_pasting"]["p95"],
        "boundary_maturity_mean": boundary["boundary_at_maturity"]["mean"],
    }


def main() -> None:
    for checkpoint_dir in CHECKPOINTS.values():
        for filename in ("solutionNN.pth", "boundaryNN.pth"):
            if not (checkpoint_dir / filename).is_file():
                raise FileNotFoundError(checkpoint_dir / filename)

    samples = make_samples()
    results: dict[str, object] = {
        "seed": SEED,
        "sample_sizes": {"main": N_TEST, "pde": N_PDE},
        "state_domain": {"s": [S_MIN, S_MAX], "v": [V_MIN, V_MAX], "tau": [TAU_MIN, TAU_MAX]},
        "near_boundary_band": NEAR_BOUNDARY_BAND,
        "models": {},
    }
    rows: list[dict[str, float | int]] = []

    for weight, checkpoint_dir in CHECKPOINTS.items():
        print(f"Evaluating intrinsic loss weight {weight} from {checkpoint_dir}", flush=True)
        price_net, boundary_net = load_models(checkpoint_dir)
        model_result = {
            "checkpoint_directory": str(checkpoint_dir.relative_to(ROOT)),
            "benchmark": evaluate_benchmark(price_net, boundary_net),
            "exercise_region": evaluate_exercise(price_net, boundary_net, samples),
            "intrinsic_lower_bound": evaluate_lower_bound(price_net, boundary_net, samples),
            "boundary_conditions": evaluate_boundary(price_net, boundary_net, samples),
            "pde": evaluate_pde(price_net, boundary_net, samples),
        }
        results["models"][str(weight)] = model_result
        rows.append(csv_row(weight, model_result))
        print(f"Completed intrinsic loss weight {weight}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "intrinsic_ablation_results.json"
    csv_path = OUTPUT_DIR / "intrinsic_ablation_summary.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    for row in rows:
        print(json.dumps(row, sort_keys=False))


if __name__ == "__main__":
    main()
