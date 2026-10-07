import torch
from config import (
    device, S_BOUNDS, V_BOUNDS,
    KAPPA_BOUNDS, THETA_BOUNDS, SIGMA_V_BOUNDS, RHO_BOUNDS, T_s_BOUNDS,
)
from models import create_models
from param_sampling import sample_heston_params, normalize_params


def load_trained_models(save_dir="./saved_models"):
    priceNN, freeBoundaryNN = create_models(device)
    priceNN.load_state_dict(torch.load(f"{save_dir}/solutionNN.pth", map_location=device))
    freeBoundaryNN.load_state_dict(torch.load(f"{save_dir}/boundaryNN.pth", map_location=device))
    priceNN.eval()
    freeBoundaryNN.eval()
    return priceNN, freeBoundaryNN


def report_by_bins(name, values_phys, violation, bounds, n_bins=5):
    lo, hi = bounds
    edges = torch.linspace(lo, hi, n_bins + 1)
    flat_vals = values_phys.flatten().cpu()
    flat_viol = violation.flatten().cpu()

    print(f"-- Trigger rate by {name} bin --")
    for i in range(n_bins):
        b_lo, b_hi = edges[i].item(), edges[i + 1].item()
        if i == n_bins - 1:
            mask = (flat_vals >= b_lo) & (flat_vals <= b_hi)
        else:
            mask = (flat_vals >= b_lo) & (flat_vals < b_hi)
        n = mask.sum().item()
        if n == 0:
            print(f"  [{b_lo:.3f}, {b_hi:.3f}): no points sampled")
            continue
        v_sub = flat_viol[mask]
        triggered = v_sub > 0
        trig_rate = triggered.float().mean().item()
        mean_v = v_sub[triggered].mean().item() if triggered.any() else 0.0
        print(f"  [{b_lo:.3f}, {b_hi:.3f}): N={n} | trigger_rate={trig_rate:.2%} | "
              f"mean_violation_when_triggered={mean_v:.6f}")
    print()


def diagnose_exercise_region_fit(price_net, free_boundary_net, N=200_000):
    """
    Directly tests whether L_int worked: samples fresh points in the
    exercise region (s < s_star_bar, never seen during training) and
    compares price_net's RAW output (bypassing torch.where / the clamp)
    against the true intrinsic value. This is the only way to check L_int's
    effect, since diagnose_clamp_violations's `raw_price` is forced to
    p_intrinsic exactly whenever s < s_star and therefore never reflects
    what price_net itself predicts there.
    """
    s_min, s_max = S_BOUNDS
    v_min, v_max = V_BOUNDS

    with torch.no_grad():
        v   = torch.rand((N, 1), device=device) * (v_max - v_min) + v_min
        kappa, theta, sigma_v, rho, T_s = sample_heston_params(N, width_fraction=1.0)
        tau = torch.rand((N, 1), device=device) * T_s  # tau = remaining time, range [0, T_s]
        kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa, theta, sigma_v, rho, T_s)

        X_bound = torch.cat([v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        s_star_bar = free_boundary_net(X_bound).clamp(s_min, s_max * 0.99)

        # fresh points strictly inside the exercise region, uniformly this
        # time (not power-law biased toward s_star_bar) to test the whole region
        u = torch.rand((N, 1), device=device)
        s = s_min + (s_star_bar - s_min) * u

        X_sol = torch.cat([s, v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        p_raw = price_net(X_sol)

        p_intrinsic = torch.relu(1.0 - s)
        error = (p_raw - p_intrinsic).abs()

        error_cpu = error.flatten().cpu()
        p95 = torch.quantile(error_cpu, 0.95).item()

        print(f"Exercise-region fit (fresh points, s < s_star_bar, N={N}):")
        print(f"  mean_abs_error={error_cpu.mean().item():.6f} | "
              f"median_abs_error={error_cpu.median().item():.6f} | "
              f"max_abs_error={error_cpu.max().item():.6f} | "
              f"95th_pct_error={p95:.6f}")
        print()

        # Locate the worst offenders: print full coordinates of the top-5
        # largest errors, to see whether they are isolated freak points or
        # cluster in one systematic weak region of the input space.
        top_err, top_idx = torch.topk(error.flatten(), k=5)
        print("Top-5 worst exercise-region points:")
        print(f"  {'error':>10} {'s':>8} {'s_star_bar':>8} {'v':>8} {'tau':>8} "
              f"{'kappa':>8} {'theta':>8} {'sigma_v':>8} {'rho':>8} {'T_s':>8} "
              f"{'p_raw':>9} {'intrinsic':>9}")
        for rank in range(5):
            j = top_idx[rank].item()
            print(f"  {top_err[rank].item():>10.6f} {s.flatten()[j].item():>8.4f} "
                  f"{s_star_bar.flatten()[j].item():>8.4f} {v.flatten()[j].item():>8.4f} "
                  f"{tau.flatten()[j].item():>8.4f} {kappa.flatten()[j].item():>8.4f} "
                  f"{theta.flatten()[j].item():>8.4f} {sigma_v.flatten()[j].item():>8.4f} "
                  f"{rho.flatten()[j].item():>8.4f} {T_s.flatten()[j].item():>8.4f} "
                  f"{p_raw.flatten()[j].item():>9.4f} {p_intrinsic.flatten()[j].item():>9.4f}")
        print()


def diagnose_clamp_violations(price_net, free_boundary_net, N=200_000, near_boundary_band=0.05):
    s_min, s_max = S_BOUNDS
    v_min, v_max = V_BOUNDS

    with torch.no_grad():
        s   = torch.rand((N, 1), device=device) * (s_max - s_min) + s_min
        v   = torch.rand((N, 1), device=device) * (v_max - v_min) + v_min

        # width_fraction=1.0 samples over the full trained parameter range,
        # with the same Feller-condition adjustment to sigma_v used during
        # training, so this diagnostic never probes combinations the model
        # was never exposed to.
        kappa, theta, sigma_v, rho, T_s = sample_heston_params(N, width_fraction=1.0)
        tau = torch.rand((N, 1), device=device) * T_s  # tau = remaining time, range [0, T_s]

        kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa, theta, sigma_v, rho, T_s)

        X_bound = torch.cat([v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        s_star = free_boundary_net(X_bound)

        X_sol = torch.cat([s, v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        p_continuation = price_net(X_sol)

        p_intrinsic = torch.relu(1.0 - s)
        raw_price = torch.where(s < s_star, p_intrinsic, p_continuation)

        violation = torch.relu(p_intrinsic - raw_price)

        all_mask  = torch.ones_like(s, dtype=torch.bool).flatten()
        near_mask = (torch.abs(s - s_star) < near_boundary_band).flatten()

        def report(mask, label):
            n = mask.sum().item()
            if n == 0:
                print(f"{label}: no points sampled in this band")
                return
            v_sub = violation.flatten()[mask]
            triggered = v_sub > 0
            trig_rate = triggered.float().mean().item()
            mean_v = v_sub[triggered].mean().item() if triggered.any() else 0.0
            max_v = v_sub.max().item()
            print(f"{label}: N={n} | trigger_rate={trig_rate:.4%} | "
                  f"mean_violation_when_triggered={mean_v:.6f} | max_violation={max_v:.6f}")

        report(all_mask, "All points (full parameter range)")
        report(near_mask, f"Near boundary only (|s - s_star| < {near_boundary_band})")

        print()
        report_by_bins("v (state variable)", v,     violation, V_BOUNDS)
        report_by_bins("kappa",              kappa, violation, KAPPA_BOUNDS)
        report_by_bins("theta",              theta, violation, THETA_BOUNDS)
        report_by_bins("sigma_v",            sigma_v, violation, SIGMA_V_BOUNDS)
        report_by_bins("rho",                rho,   violation, RHO_BOUNDS)
        report_by_bins("T_s",                T_s,   violation, T_s_BOUNDS)


if __name__ == '__main__':
    priceNN, freeBoundaryNN = load_trained_models()
    diagnose_exercise_region_fit(priceNN, freeBoundaryNN)
    diagnose_clamp_violations(priceNN, freeBoundaryNN)
