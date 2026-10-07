import torch
from config import (
    device, KAPPA_BOUNDS, THETA_BOUNDS, SIGMA_V_BOUNDS, RHO_BOUNDS, T_s_BOUNDS,
    FELLER_MARGIN, kappa_ref, theta_ref, sigma_v_ref, rho_ref, T_s_ref,
)


def _lerp_bounds(ref, full_bounds, width_fraction):
    lo_full, hi_full = full_bounds
    window_width = width_fraction * (hi_full - lo_full)

    lo = ref - 0.5 * window_width
    hi = ref + 0.5 * window_width

    # Shift the window (instead of clipping it) when it spills past a true
    # bound, so the window always keeps its full requested width -- clipping
    # would silently shrink it whenever ref sits off-center, leaving the far
    # side permanently under-covered even at width_fraction=1.0.
    if lo < lo_full:
        shift = lo_full - lo
        lo += shift
        hi += shift
    if hi > hi_full:
        shift = hi - hi_full
        lo -= shift
        hi -= shift

    lo = max(lo, lo_full)
    hi = min(hi, hi_full)
    return lo, hi


def get_curriculum_bounds(width_fraction):
    return {
        'kappa':   _lerp_bounds(kappa_ref,   KAPPA_BOUNDS,   width_fraction),
        'theta':   _lerp_bounds(theta_ref,   THETA_BOUNDS,   width_fraction),
        'sigma_v': _lerp_bounds(sigma_v_ref, SIGMA_V_BOUNDS, width_fraction),
        'rho':     _lerp_bounds(rho_ref,     RHO_BOUNDS,     width_fraction),
        'T_s':     _lerp_bounds(T_s_ref,     T_s_BOUNDS,     width_fraction),
    }


def sample_heston_params(N, width_fraction=1.0):
    bounds = get_curriculum_bounds(width_fraction)

    k_lo,   k_hi   = bounds['kappa']
    th_lo,  th_hi  = bounds['theta']
    sv_lo,  sv_hi  = bounds['sigma_v']
    rho_lo, rho_hi = bounds['rho']
    Ts_lo,  Ts_hi  = bounds['T_s']

    kappa = torch.rand((N, 1), device=device) * (k_hi - k_lo) + k_lo

    # Power-law bias toward th_lo: low theta is empirically harder to fit
    # (diagnose_clamp.py showed a ~3.27x higher violation rate there), so
    # more collocation points are allocated to that end of the range.
    u_theta = torch.rand((N, 1), device=device)
    theta = th_lo + (th_hi - th_lo) * (u_theta ** 1.75)

    feller_limit = FELLER_MARGIN * torch.sqrt(2.0 * kappa * theta)
    sv_hi_tensor = torch.full_like(kappa, sv_hi)
    effective_sv_hi = torch.minimum(sv_hi_tensor, feller_limit)
    sv_lo_tensor = torch.full_like(kappa, sv_lo)
    effective_sv_lo = torch.where(
        sv_lo_tensor > effective_sv_hi,
        torch.clamp(effective_sv_hi - 1e-3, min=1e-3),
        sv_lo_tensor,
    )
    u = torch.rand((N, 1), device=device)
    sigma_v = effective_sv_lo + u * (effective_sv_hi - effective_sv_lo)

    rho = torch.rand((N, 1), device=device) * (rho_hi - rho_lo) + rho_lo
    T_s = torch.rand((N, 1), device=device) * (Ts_hi - Ts_lo) + Ts_lo

    return kappa, theta, sigma_v, rho, T_s


def normalize_params(kappa, theta, sigma_v, rho, T_s):
    def _norm(x, bounds):
        lo, hi = bounds
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    kappa_hat   = _norm(kappa,   KAPPA_BOUNDS)
    theta_hat   = _norm(theta,   THETA_BOUNDS)
    sigma_v_hat = _norm(sigma_v, SIGMA_V_BOUNDS)
    rho_hat     = _norm(rho,     RHO_BOUNDS)
    T_s_n       = _norm(T_s,     T_s_BOUNDS)
    return kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n
