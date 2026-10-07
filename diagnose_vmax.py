"""
Diagnoses whether the V_v=0 far-field approximation at v=V_BOUNDS[1] (=1.0)
is actually justified for this model, BEFORE deciding whether to impose it
as an explicit Neumann loss term. Runs entirely on the already-trained
model -- no retraining needed.

Checks, near v = 0.8, 0.9, 0.95, 1.0, in the continuation region (s > s_star),
sampled across the full trained Heston parameter range:
  1. PDE residual magnitude -- is it elevated near v_max relative to the
     rest of the domain, suggesting the model is already struggling there?
  2. |dp/dv| -- has price sensitivity to v actually decayed toward 0, or
     is the model still learning a meaningfully v-dependent continuation
     price out near v_max (in which case forcing V_v=0 would be wrong)?
  3. Smoothness of the free boundary S_f(v) at the reference Heston
     parameters, across a dense v grid -- oscillations would indicate
     instability near v_max independent of the Neumann-condition question.
"""
import torch
from config import (
    device, S_BOUNDS, K_ref, T_s_ref as T_ref, kappa_ref, theta_ref, sigma_v_ref, rho_ref,
)
from param_sampling import sample_heston_params, normalize_params
from loss import get_PDE_residual
from diagnose_clamp import load_trained_models

V_LEVELS = [0.8, 0.9, 0.95, 1.0]
N = 20_000


def check_residual_and_v_sensitivity(price_net, free_boundary_net):
    s_min, s_max = S_BOUNDS

    print(f"{'v':>6} | {'PDE |residual|':>16} | {'|dp/dv| mean':>13} | {'|dp/dv| max':>12} | {'|dp/dv| p95':>12}")
    print("-" * 72)

    for v_val in V_LEVELS:
        with torch.no_grad():
            kappa, theta, sigma_v, rho, T_s = sample_heston_params(N, width_fraction=1.0)
            kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa, theta, sigma_v, rho, T_s)
            v_fixed = torch.full((N, 1), v_val, device=device)
            tau = torch.rand((N, 1), device=device) * T_s

            X_bound = torch.cat([v_fixed, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
            s_star_bar = free_boundary_net(X_bound).clamp(s_min, s_max * 0.99)

            u_s = torch.rand((N, 1), device=device)
            s_vals = s_star_bar + (s_max - s_star_bar) * u_s  # continuation region only

        # --- PDE residual at these points ---
        s_g = s_vals.detach().requires_grad_(True)
        v_g = v_fixed.detach().requires_grad_(True)
        tau_g = tau.detach().requires_grad_(True)
        params_phys = (kappa, theta, sigma_v, rho, T_s)
        params_norm = (kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n)
        target_zero = torch.zeros_like(s_g)
        res = get_PDE_residual(s_g, v_g, tau_g, params_phys, params_norm, target_zero, price_net)
        res_abs = res.detach().abs()

        # --- dp/dv directly (separate, first-order-only pass) ---
        v_g2 = v_fixed.detach().requires_grad_(True)
        X_sol = torch.cat([s_vals, v_g2, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        p = price_net(X_sol)
        p_v = torch.autograd.grad(p, v_g2, grad_outputs=torch.ones_like(p), create_graph=False)[0]
        p_v_abs = p_v.detach().abs()

        p95 = torch.quantile(p_v_abs.flatten(), 0.95).item()
        print(f"{v_val:>6.2f} | {res_abs.mean().item():>16.6f} | {p_v_abs.mean().item():>13.6f} | "
              f"{p_v_abs.max().item():>12.6f} | {p95:>12.6f}")
    print()


def check_free_boundary_smoothness(free_boundary_net, K=K_ref, T=T_ref,
                                    kappa=kappa_ref, theta=theta_ref,
                                    sigma_v=sigma_v_ref, rho=rho_ref,
                                    n_points=200):
    v_grid = torch.linspace(0.0, 1.0, n_points, device=device).reshape(-1, 1)
    tau_fixed = torch.full_like(v_grid, T)  # t=0 slice (full time remaining)

    kappa_t = torch.full_like(v_grid, kappa)
    theta_t = torch.full_like(v_grid, theta)
    sigma_v_t = torch.full_like(v_grid, sigma_v)
    rho_t = torch.full_like(v_grid, rho)
    T_t = torch.full_like(v_grid, T)
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa_t, theta_t, sigma_v_t, rho_t, T_t)

    with torch.no_grad():
        X_bound = torch.cat([v_grid, tau_fixed, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        S_star = (free_boundary_net(X_bound) * K).flatten().cpu()

    d1 = S_star[1:] - S_star[:-1]
    d2 = d1[1:] - d1[:-1]

    print(f"Free boundary S_f(v) smoothness at reference params (t=0, K={K}, T={T}):")
    print(f"  max |first difference|  = {d1.abs().max().item():.6f}")
    print(f"  max |second difference| = {d2.abs().max().item():.6f}  (large spikes here indicate kinks/oscillation)")
    print("  S_f(v) near v_max:")
    for i in range(n_points - 5, n_points):
        print(f"    v={v_grid[i].item():.4f}  S_f={S_star[i].item():.6f}")
    print()


if __name__ == '__main__':
    priceNN, freeBoundaryNN = load_trained_models()
    check_residual_and_v_sensitivity(priceNN, freeBoundaryNN)
    check_free_boundary_smoothness(freeBoundaryNN)
