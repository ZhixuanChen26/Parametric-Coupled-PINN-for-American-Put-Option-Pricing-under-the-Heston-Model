import torch
from config import device
from param_sampling import sample_heston_params, normalize_params

_beta_dist = torch.distributions.Beta(
    torch.tensor(0.6, dtype=torch.float32),
    torch.tensor(0.9, dtype=torch.float32),
)

def _sample_v(N, v_min, v_max):
    u = _beta_dist.sample((N, 1)).to(device)
    return u * (v_max - v_min) + v_min


def _sample_params(N, width_fraction):
    kappa, theta, sigma_v, rho, T_s = sample_heston_params(N, width_fraction)
    params_phys = (kappa, theta, sigma_v, rho, T_s)
    params_norm = normalize_params(kappa, theta, sigma_v, rho, T_s)
    return params_phys, params_norm


def generate_pde_data(N, free_boundary_net, s_bounds, v_bounds, tau_bounds,
                       power_s=3.0, power_tau=2.0, width_fraction=1.0):
    s_min, s_max = s_bounds
    v_min, v_max = v_bounds

    params_phys, params_norm = _sample_params(N, width_fraction)
    kappa, theta, sigma_v, rho, T_s = params_phys
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = params_norm

    v = _sample_v(N, v_min, v_max)

    # tau = remaining time to maturity = T - t (see loss.py). Power-law
    # biased toward tau -> 0 (near maturity), mirroring the old bias toward
    # t/T -> 1 in the fractional-time parameterization. Range depends on
    # each sample's own T_s, not a fixed [0,1].
    u_tau = torch.rand((N, 1), device=device)
    tau = T_s * (u_tau ** power_tau)

    free_boundary_net.eval()
    with torch.no_grad():
        X_bound = torch.cat([v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        s_star_bar = free_boundary_net(X_bound).clamp(s_min, s_max * 0.99)

    u_s = torch.rand((N, 1), device=device)
    s = s_star_bar + (s_max - s_star_bar) * (u_s ** power_s)

    target = torch.zeros_like(s, device=device)

    indices = torch.randperm(N)
    params_phys_shuf = tuple(p[indices] for p in params_phys)
    params_norm_shuf = tuple(p[indices] for p in params_norm)

    return (s[indices], v[indices], tau[indices],
            params_phys_shuf, params_norm_shuf, target[indices])


def generate_terminal_data(N, free_boundary_net, s_bounds, v_bounds, tau_bounds,
                            power_s=3.0, width_fraction=1.0):
    s_min, s_max = s_bounds
    v_min, v_max = v_bounds

    params_phys, params_norm = _sample_params(N, width_fraction)
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = params_norm

    # Maturity is tau = 0 (regardless of each sample's own T_s).
    tau = torch.zeros((N, 1), device=device)
    v = _sample_v(N, v_min, v_max)

    free_boundary_net.eval()
    with torch.no_grad():
        X_bound = torch.cat([v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        s_star_bar = free_boundary_net(X_bound).clamp(s_min, s_max * 0.99)

    # The terminal condition p(s, v, tau=0) = max(1-s, 0) holds on the
    # WHOLE s domain, not just the continuation side. Mix 20% uniform
    # samples over [s_min, s_max] into the power-law draw so the
    # exercise-side corner (s -> 0 at maturity) is anchored too --
    # diagnostics located the single worst exercise-region error
    # (|err| ~ 1.3) exactly in that corner, which no loss term covered
    # before.
    u_s = torch.rand((N, 1), device=device)
    s_powerlaw = s_star_bar + (s_max - s_star_bar) * (u_s ** power_s)
    s_uniform = torch.rand((N, 1), device=device) * (s_max - s_min) + s_min
    mix_mask = torch.rand((N, 1), device=device) < 0.2
    s = torch.where(mix_mask, s_uniform, s_powerlaw)

    payoff = torch.relu(1.0 - s)

    indices = torch.randperm(N)
    params_phys_shuf = tuple(p[indices] for p in params_phys)
    params_norm_shuf = tuple(p[indices] for p in params_norm)

    return (s[indices], v[indices], tau[indices],
            params_phys_shuf, params_norm_shuf, payoff[indices])


def generate_exercise_region_data(N, free_boundary_net, s_bounds, v_bounds, tau_bounds,
                                   power_s=3.0, width_fraction=1.0):
    s_min, s_max = s_bounds
    v_min, v_max = v_bounds

    params_phys, params_norm = _sample_params(N, width_fraction)
    kappa, theta, sigma_v, rho, T_s = params_phys
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = params_norm

    v = _sample_v(N, v_min, v_max)
    tau = torch.rand((N, 1), device=device) * T_s

    free_boundary_net.eval()
    with torch.no_grad():
        X_bound = torch.cat([v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        s_star_bar = free_boundary_net(X_bound).clamp(s_min, s_max * 0.99)

    # Mirrors the generate_terminal_data fix: the power-law draw biases s
    # toward s_star_bar and essentially never reaches s_min, so L_intrinsic
    # never supervised the far side of the exercise region (diagnose_clamp.py's
    # worst exercise-region points cluster at s->0). Mix in 20% uniform
    # sampling over the full [s_min, s_star_bar] exercise range to cover it.
    u_s = torch.rand((N, 1), device=device)
    s_powerlaw = s_star_bar - (s_star_bar - s_min) * (u_s ** power_s)
    s_uniform = torch.rand((N, 1), device=device) * (s_star_bar - s_min) + s_min
    mix_mask = torch.rand((N, 1), device=device) < 0.2
    s = torch.where(mix_mask, s_uniform, s_powerlaw)

    target = torch.relu(1.0 - s)

    indices = torch.randperm(N)
    params_phys_shuf = tuple(p[indices] for p in params_phys)
    params_norm_shuf = tuple(p[indices] for p in params_norm)

    return (s[indices], v[indices], tau[indices],
            params_phys_shuf, params_norm_shuf, target[indices])


def generate_spatial_boundary_data(N, s_bounds, v_bounds, tau_bounds, width_fraction=1.0):
    s_min, s_max = s_bounds
    v_min, v_max = v_bounds

    params_phys, params_norm = _sample_params(N, width_fraction)
    kappa, theta, sigma_v, rho, T_s = params_phys

    s = torch.ones((N, 1), device=device) * s_max
    v = _sample_v(N, v_min, v_max)
    tau = torch.rand((N, 1), device=device) * T_s

    target = torch.zeros_like(s, device=device)

    indices = torch.randperm(N)
    params_phys_shuf = tuple(p[indices] for p in params_phys)
    params_norm_shuf = tuple(p[indices] for p in params_norm)

    return (s[indices], v[indices], tau[indices],
            params_phys_shuf, params_norm_shuf, target[indices])


def generate_free_boundary_data(N, v_bounds, tau_bounds, width_fraction=1.0):
    v_min, v_max = v_bounds

    params_phys, params_norm = _sample_params(N, width_fraction)
    kappa, theta, sigma_v, rho, T_s = params_phys

    v = _sample_v(N, v_min, v_max)
    tau = torch.rand((N, 1), device=device) * T_s
    target = torch.zeros((N, 1), device=device)

    return v, tau, params_phys, params_norm, target


def generate_free_boundary_terminal_data(N, v_bounds, width_fraction=1.0):
    v_min, v_max = v_bounds

    params_phys, params_norm = _sample_params(N, width_fraction)

    v = _sample_v(N, v_min, v_max)
    # Maturity is tau = 0. Target s_star_norm stays 1.0 (s* = K at maturity,
    # independent of T_s).
    tau = torch.zeros((N, 1), device=device)
    target = torch.ones((N, 1), device=device)

    return v, tau, params_phys, params_norm, target


def generate_all_training_data(free_boundary_net, N_pde, N_term, N_spat, N_free, N_free_term, N_intrinsic,
                               s_bounds, v_bounds, tau_bounds, power_s=3.0, power_tau=2.0,
                               width_fraction=1.0):
    print(f'Generating dimensionless training data (param width_fraction={width_fraction})...')
    data = {}
    data['pde']       = generate_pde_data(N_pde, free_boundary_net, s_bounds, v_bounds, tau_bounds,
                                           power_s, power_tau, width_fraction)
    data['term']      = generate_terminal_data(N_term, free_boundary_net, s_bounds, v_bounds, tau_bounds,
                                                power_s, width_fraction)
    data['spat']      = generate_spatial_boundary_data(N_spat, s_bounds, v_bounds, tau_bounds, width_fraction)
    data['free']      = generate_free_boundary_data(N_free, v_bounds, tau_bounds, width_fraction)
    data['free_term'] = generate_free_boundary_terminal_data(N_free_term, v_bounds, width_fraction)
    data['intrinsic'] = generate_exercise_region_data(N_intrinsic, free_boundary_net, s_bounds, v_bounds, tau_bounds,
                                                        power_s, width_fraction)
    return data
