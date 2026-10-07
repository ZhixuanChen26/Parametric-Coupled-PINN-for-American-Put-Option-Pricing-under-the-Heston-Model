"""
Diagnoses whether the V=0 far-field (Dirichlet) approximation at
s=S_BOUNDS[1] is justified, on the ALREADY-TRAINED model -- no
retraining needed.

IMPORTANT -- checkpoint/config consistency: a saved checkpoint's weights
carry no record of what S_BOUNDS it was trained under. This script always
tests against config.py's CURRENT S_BOUNDS, so before trusting the
output, make sure that value actually matches the checkpoint you're
loading (the same problem does not affect kappa/theta/sigma_v/rho/T_s,
since those are pulled from config.py consistently on both the training
and diagnostic side -- it's specifically S_BOUNDS that isn't recorded
anywhere in the .pth files).

Fixes vs. the first version of this script:
  - `american_put_price` already returns dollar prices (price_norm * K,
    see pricer.py). Everything below prints and reasons about dollar
    prices directly -- do NOT multiply by K again.
  - The "extreme corner" test point is chosen to satisfy the Feller
    condition actually enforced by param_sampling.sample_heston_params;
    a Feller-violating point would never be seen in training and tells
    you nothing about in-domain behavior.
  - Grid points are fractions of the CURRENT S_BOUNDS[1], not hardcoded
    absolute values, so this stays correct if S_BOUNDS changes.
  - The PDE-residual check draws ONE batch of (kappa,theta,sigma_v,rho,T_s,
    v,tau) and re-evaluates that SAME batch at each s level, so only s
    varies -- the earlier version resampled everything at each s level,
    confounding the effect of s with ordinary batch-to-batch noise.
  - The Black-Scholes comparison is a rough heuristic only (uses an ad
    hoc effective-vol proxy, sigma=sqrt(theta) or sqrt(avg v)). European
    <= American in general, but that does NOT make this BS number a
    rigorous lower bound for the Heston American price -- treat it as a
    sanity-check ballpark, not a proof.
"""
import math
import torch
from config import (
    device, S_BOUNDS, K_ref, T_s_ref as T_ref, kappa_ref, theta_ref, sigma_v_ref, rho_ref, r,
    FELLER_MARGIN,
)
from param_sampling import sample_heston_params, normalize_params
from loss import get_PDE_residual
from pricer import american_put_price
from diagnose_clamp import load_trained_models

N = 20_000


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _feller_admissible_sigma_v(kappa, theta, sigma_v_requested):
    """Clamp to what sample_heston_params would actually have allowed."""
    limit = FELLER_MARGIN * math.sqrt(2.0 * kappa * theta)
    return min(sigma_v_requested, limit), limit


def check_price_decay_to_smax(price_net, free_boundary_net):
    s_min, s_max = S_BOUNDS
    print(f"(Using CURRENT config.S_BOUNDS = {S_BOUNDS} -- verify this matches the "
          f"checkpoint you loaded before trusting these numbers.)\n")

    s_fracs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.0]
    s_grid = [f * s_max for f in s_fracs]

    print("=== Price decay approaching s_max, reference params ===")
    print(f"(K={K_ref}, T={T_ref}, kappa={kappa_ref}, theta={theta_ref}, sigma_v={sigma_v_ref}, "
          f"rho={rho_ref}, v=theta; all prices in dollars, already K-scaled)")
    for s_val in s_grid:
        S_val = s_val * K_ref
        price, _ = american_put_price(S_val, theta_ref, 0.0, price_net, free_boundary_net,
                                       K=K_ref, T=T_ref, kappa=kappa_ref, theta=theta_ref,
                                       sigma_v=sigma_v_ref, rho=rho_ref)
        bs_ref = black_scholes_put(S_val, K_ref, T_ref, r, math.sqrt(theta_ref))
        print(f"  s={s_val:.3f} (S={S_val:>6.2f}): model_price=${price.item():.6f} "
              f"(p_norm={price.item()/K_ref:.6f}) | BS_heuristic=${bs_ref:.6f}")
    print()

    # Extreme corner: high vol-of-vol / long maturity, but clamped to stay
    # Feller-admissible for this (kappa, theta) so it's an in-domain point
    # sample_heston_params could actually have produced.
    kappa_ex, theta_ex, rho_ex, T_ex, v_ex = 1.5, 0.5, -0.9, 1.0, 1.0
    sigma_v_ex, feller_limit = _feller_admissible_sigma_v(kappa_ex, theta_ex, 1.0)
    print("=== Price decay approaching s_max, EXTREME corner (high vol-of-vol, long T) ===")
    print(f"(K={K_ref}, T={T_ex}, kappa={kappa_ex}, theta={theta_ex}, sigma_v={sigma_v_ex:.4f} "
          f"[Feller limit={feller_limit:.4f}, admissible], rho={rho_ex}, v={v_ex})")
    for s_val in s_grid:
        S_val = s_val * K_ref
        price, _ = american_put_price(S_val, v_ex, 0.0, price_net, free_boundary_net,
                                       K=K_ref, T=T_ex, kappa=kappa_ex, theta=theta_ex,
                                       sigma_v=sigma_v_ex, rho=rho_ex)
        sigma_eff = math.sqrt(0.5 * (v_ex + theta_ex))
        bs_ref = black_scholes_put(S_val, K_ref, T_ex, r, sigma_eff)
        print(f"  s={s_val:.3f} (S={S_val:>6.2f}): model_price=${price.item():.6f} "
              f"(p_norm={price.item()/K_ref:.6f}) | BS_heuristic=${bs_ref:.6f}")
    print()


def check_pde_residual_near_smax(price_net, free_boundary_net):
    s_min, s_max = S_BOUNDS
    s_fracs = [0.8, 0.9, 0.95, 0.975, 1.0]
    s_levels = [f * s_max for f in s_fracs]

    # Draw ONE batch and hold (kappa,theta,sigma_v,rho,T_s,v,tau) fixed
    # across all s levels -- only s varies below, so any trend is
    # attributable to s alone, not to a fresh random draw at each step.
    kappa, theta, sigma_v, rho, T_s = sample_heston_params(N, width_fraction=1.0)
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa, theta, sigma_v, rho, T_s)
    v = torch.rand((N, 1), device=device)
    tau = torch.rand((N, 1), device=device) * T_s
    params_phys = (kappa, theta, sigma_v, rho, T_s)
    params_norm = (kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n)

    print("=== PDE residual approaching s_max (SAME batch of params/v/tau at every s) ===")
    print(f"{'s':>8} | {'PDE |residual|':>16}")
    print("-" * 30)
    for s_val in s_levels:
        s_g = torch.full((N, 1), s_val, device=device).requires_grad_(True)
        v_g = v.detach().requires_grad_(True)
        tau_g = tau.detach().requires_grad_(True)
        target_zero = torch.zeros_like(s_g)
        res = get_PDE_residual(s_g, v_g, tau_g, params_phys, params_norm, target_zero, price_net)
        print(f"{s_val:>8.3f} | {res.detach().abs().mean().item():>16.6f}")
    print()


if __name__ == '__main__':
    priceNN, freeBoundaryNN = load_trained_models()
    check_price_decay_to_smax(priceNN, freeBoundaryNN)
    check_pde_residual_near_smax(priceNN, freeBoundaryNN)
